#include "lcms/io/mzxml_reader.hpp"
#include "lcms/io/base64.hpp"
#include <fstream>
#include <sstream>
#include <regex>
#include <cstring>

#ifdef MASSKIT_HAS_PUGIXML
#include <pugixml.hpp>
#endif

namespace lcms {
namespace io {

class MzXMLReader::Impl {
public:
    Impl() = default;

    MSExperiment read(const std::string& filename, const MzXMLReaderOptions& options) {
#ifdef MASSKIT_HAS_PUGIXML
        return readWithPugixml(filename, options);
#else
        return readFallback(filename, options);
#endif
    }

    MSExperiment parseString(const std::string& content,
                             const MzXMLReaderOptions& options) {
#ifdef MASSKIT_HAS_PUGIXML
        return parseStringWithPugixml(content, options);
#else
        return parseStringFallback(content, options);
#endif
    }

    std::size_t countSpectra(const std::string& filename) {
#ifdef MASSKIT_HAS_PUGIXML
        pugi::xml_document doc;
        if (!doc.load_file(filename.c_str())) {
            throw MzXMLParseError("Failed to load file: " + filename);
        }
        auto ms_run = doc.select_node("//msRun");
        if (ms_run) {
            return static_cast<std::size_t>(
                ms_run.node().attribute("scanCount").as_uint());
        }
        // Count scan elements manually
        std::size_t count = 0;
        for (auto scan : doc.select_nodes("//scan")) {
            (void)scan;
            ++count;
        }
        return count;
#else
        std::ifstream file(filename);
        if (!file) {
            throw MzXMLParseError("Failed to open file: " + filename);
        }
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        std::regex count_regex("scanCount=\"(\\d+)\"");
        std::smatch match;
        if (std::regex_search(content, match, count_regex)) {
            return std::stoull(match[1].str());
        }
        // Count <scan> tags
        std::size_t count = 0;
        std::size_t pos = 0;
        while ((pos = content.find("<scan ", pos)) != std::string::npos) {
            ++count;
            ++pos;
        }
        return count;
#endif
    }

private:
#ifdef MASSKIT_HAS_PUGIXML
    MSExperiment readWithPugixml(const std::string& filename,
                                  const MzXMLReaderOptions& options) {
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(filename.c_str());

        if (!result) {
            throw MzXMLParseError("Failed to parse file: " + std::string(result.description()));
        }

        MSExperiment exp;
        exp.setSourceFile(filename);

        parseDocument(doc, exp, options);
        return exp;
    }

    MSExperiment parseStringWithPugixml(const std::string& content,
                                         const MzXMLReaderOptions& options) {
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_string(content.c_str());

        if (!result) {
            throw MzXMLParseError("Failed to parse content: " + std::string(result.description()));
        }

        MSExperiment exp;
        parseDocument(doc, exp, options);
        return exp;
    }

    void parseDocument(pugi::xml_document& doc, MSExperiment& exp,
                        const MzXMLReaderOptions& options) {
        // Find msRun element
        auto mzxml = doc.child("mzXML");
        if (!mzxml) {
            throw MzXMLParseError("No mzXML element found");
        }

        auto ms_run = mzxml.child("msRun");
        if (!ms_run) {
            throw MzXMLParseError("No msRun element found");
        }

        // Parse instrument
        auto instrument = ms_run.child("msInstrument");
        if (instrument) {
            if (auto model = instrument.child("msModel")) {
                exp.setInstrumentModel(model.attribute("value").value());
            }
            if (auto serial = instrument.child("msSerialNumber")) {
                exp.setInstrumentSerial(serial.attribute("value").value());
            }
        }

        // Parse start/end time
        if (auto start = ms_run.attribute("startTime")) {
            exp.metadata()["startTime"] = start.value();
        }

        // Reserve space for spectra
        std::size_t scan_count = ms_run.attribute("scanCount").as_uint(0);
        if (scan_count > 0) {
            exp.reserveSpectra(scan_count);
        }

        // Parse scans recursively (mzXML uses nested scans for MS/MS)
        parseScans(ms_run, exp, options, 0);
    }

    void parseScans(pugi::xml_node parent, MSExperiment& exp,
                    const MzXMLReaderOptions& options, int depth) {
        for (auto scan : parent.children("scan")) {
            if (options.max_spectra > 0 && exp.spectrumCount() >= options.max_spectra) {
                return;
            }

            Spectrum spec = parseScan(scan, options);

            // Apply filters
            if (!options.ms_levels.empty()) {
                auto it = std::find(options.ms_levels.begin(),
                                    options.ms_levels.end(),
                                    spec.msLevel());
                if (it == options.ms_levels.end()) {
                    // Still parse nested scans
                    parseScans(scan, exp, options, depth + 1);
                    continue;
                }
            }

            if (options.rt_range) {
                if (!options.rt_range->contains(spec.retentionTime())) {
                    parseScans(scan, exp, options, depth + 1);
                    continue;
                }
            }

            exp.addSpectrum(std::move(spec));

            // Progress callback
            if (options.progress_callback) {
                if (!options.progress_callback(static_cast<int>(exp.spectrumCount()), -1)) {
                    return;  // User cancelled
                }
            }

            // Parse nested scans (MS/MS)
            parseScans(scan, exp, options, depth + 1);
        }
    }

    Spectrum parseScan(pugi::xml_node node, const MzXMLReaderOptions& options) {
        Spectrum spec;

        // Parse attributes
        spec.setMsLevel(static_cast<MSLevel>(node.attribute("msLevel").as_int(1)));

        // Retention time (stored in various formats)
        if (auto rt_attr = node.attribute("retentionTime")) {
            std::string rt_str = rt_attr.value();
            // Format: PT123.456S (ISO 8601 duration) or just a number
            double rt = 0.0;
            if (rt_str.substr(0, 2) == "PT" && rt_str.back() == 'S') {
                rt = std::stod(rt_str.substr(2, rt_str.size() - 3));
            } else if (rt_str.substr(0, 2) == "PT" && rt_str.back() == 'M') {
                rt = std::stod(rt_str.substr(2, rt_str.size() - 3)) * 60.0;
            } else {
                rt = std::stod(rt_str);
            }
            spec.setRetentionTime(rt);
        }

        // Polarity
        std::string polarity = node.attribute("polarity").value();
        if (polarity == "+" || polarity == "positive") {
            spec.setPolarity(Polarity::POSITIVE);
        } else if (polarity == "-" || polarity == "negative") {
            spec.setPolarity(Polarity::NEGATIVE);
        }

        // Centroided
        if (node.attribute("centroided").as_bool(false)) {
            spec.setType(SpectrumType::CENTROID);
        } else {
            spec.setType(SpectrumType::PROFILE);
        }

        // TIC and base peak
        if (auto tic = node.attribute("totIonCurrent")) {
            spec.setTic(tic.as_double());
        }

        // Precursor info (for MS/MS)
        if (spec.msLevel() > 1) {
            for (auto prec_node : node.children("precursorMz")) {
                Precursor prec;
                prec.mz = std::stod(prec_node.child_value());

                if (auto intensity = prec_node.attribute("precursorIntensity")) {
                    prec.intensity = intensity.as_double();
                }
                if (auto charge = prec_node.attribute("precursorCharge")) {
                    prec.charge = static_cast<ChargeState>(charge.as_int());
                }
                if (auto window = prec_node.attribute("windowWideness")) {
                    double half_width = window.as_double() / 2.0;
                    prec.isolation_window_lower = prec.mz - half_width;
                    prec.isolation_window_upper = prec.mz + half_width;
                }
                if (auto activation = prec_node.attribute("activationMethod")) {
                    std::string method = activation.value();
                    if (method == "CID") prec.activation = ActivationMethod::CID;
                    else if (method == "HCD") prec.activation = ActivationMethod::HCD;
                    else if (method == "ETD") prec.activation = ActivationMethod::ETD;
                }

                spec.addPrecursor(prec);
            }
        }

        // Parse peaks
        auto peaks_node = node.child("peaks");
        if (peaks_node) {
            int precision = peaks_node.attribute("precision").as_int(32);
            std::string byte_order = peaks_node.attribute("byteOrder").value();
            std::string compression = peaks_node.attribute("compressionType").value();
            std::string pair_order = peaks_node.attribute("pairOrder").value();

            bool is_64bit = (precision == 64);
            bool is_little_endian = (byte_order != "network");  // mzXML typically uses big-endian
            bool is_compressed = (compression == "zlib");

            // Default pair order is m/z-int
            bool mz_first = (pair_order != "int-mz");

            std::string base64_data = peaks_node.child_value();
            base64_data.erase(
                std::remove_if(base64_data.begin(), base64_data.end(), ::isspace),
                base64_data.end());

            if (!base64_data.empty()) {
                std::vector<double> values;

                try {
                    if (is_compressed) {
#ifdef MASSKIT_HAS_ZLIB
                        if (is_64bit) {
                            values = Zlib::decompressFloat64(base64_data, is_little_endian);
                        } else {
                            auto floats = Zlib::decompressFloat32(base64_data, is_little_endian);
                            values.assign(floats.begin(), floats.end());
                        }
#else
                        throw MzXMLParseError("Compressed data requires zlib support");
#endif
                    } else {
                        if (is_64bit) {
                            values = Base64::decodeFloat64(base64_data, is_little_endian);
                        } else {
                            auto floats = Base64::decodeFloat32(base64_data, is_little_endian);
                            values.assign(floats.begin(), floats.end());
                        }
                    }
                } catch (const std::exception& e) {
                    throw MzXMLParseError("Failed to decode peaks: " + std::string(e.what()));
                }

                // Values are interleaved: m/z, intensity, m/z, intensity, ...
                std::vector<MZ> mz_data;
                std::vector<Intensity> intensity_data;
                mz_data.reserve(values.size() / 2);
                intensity_data.reserve(values.size() / 2);

                for (std::size_t i = 0; i + 1 < values.size(); i += 2) {
                    if (mz_first) {
                        mz_data.push_back(values[i]);
                        intensity_data.push_back(values[i + 1]);
                    } else {
                        intensity_data.push_back(values[i]);
                        mz_data.push_back(values[i + 1]);
                    }
                }

                // Apply intensity threshold
                if (options.intensity_threshold > 0) {
                    std::vector<MZ> filtered_mz;
                    std::vector<Intensity> filtered_int;
                    for (std::size_t i = 0; i < mz_data.size(); ++i) {
                        if (intensity_data[i] >= options.intensity_threshold) {
                            filtered_mz.push_back(mz_data[i]);
                            filtered_int.push_back(intensity_data[i]);
                        }
                    }
                    spec.setData(std::move(filtered_mz), std::move(filtered_int));
                } else {
                    spec.setData(std::move(mz_data), std::move(intensity_data));
                }

                if (options.sort_by_mz && !spec.isSortedByMz()) {
                    spec.sortByMz();
                }
            }
        }

        return spec;
    }
#endif // MASSKIT_HAS_PUGIXML

    // Fallback implementation
    MSExperiment readFallback(const std::string& filename,
                               const MzXMLReaderOptions& options) {
        std::ifstream file(filename);
        if (!file) {
            throw MzXMLParseError("Failed to open file: " + filename);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return parseStringFallback(buffer.str(), options);
    }

    MSExperiment parseStringFallback(const std::string& content,
                                      [[maybe_unused]] const MzXMLReaderOptions& options) {
        MSExperiment exp;

        if (content.find("<mzXML") == std::string::npos) {
            throw MzXMLParseError("Not a valid mzXML file");
        }

        exp.metadata()["warning"] = "Loaded with fallback parser - limited functionality";
        return exp;
    }
};

MzXMLReader::MzXMLReader() : impl_(std::make_unique<Impl>()) {}
MzXMLReader::~MzXMLReader() = default;
MzXMLReader::MzXMLReader(MzXMLReader&&) noexcept = default;
MzXMLReader& MzXMLReader::operator=(MzXMLReader&&) noexcept = default;

MSExperiment MzXMLReader::read(const std::string& filename) {
    return read(filename, default_options_);
}

MSExperiment MzXMLReader::read(const std::string& filename,
                               const MzXMLReaderOptions& options) {
    try {
        return impl_->read(filename, options);
    } catch (const MzXMLParseError&) {
        throw;
    } catch (const std::exception& e) {
        last_error_ = e.what();
        throw MzXMLParseError(e.what());
    }
}

MSExperiment MzXMLReader::parseString(const std::string& content) {
    return parseString(content, default_options_);
}

MSExperiment MzXMLReader::parseString(const std::string& content,
                                      const MzXMLReaderOptions& options) {
    try {
        return impl_->parseString(content, options);
    } catch (const MzXMLParseError&) {
        throw;
    } catch (const std::exception& e) {
        last_error_ = e.what();
        throw MzXMLParseError(e.what());
    }
}

std::size_t MzXMLReader::countSpectra(const std::string& filename) {
    return impl_->countSpectra(filename);
}

bool MzXMLReader::isValidMzXML(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return false;

    char buffer[1024];
    file.read(buffer, sizeof(buffer));
    std::string content(buffer, static_cast<std::size_t>(file.gcount()));

    return content.find("<mzXML") != std::string::npos;
}

} // namespace io
} // namespace lcms
