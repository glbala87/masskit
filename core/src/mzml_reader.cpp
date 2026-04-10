#include "lcms/io/mzml_reader.hpp"
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

// CV term accessions for common parameters
namespace cv {
    // MS level
    constexpr const char* MS_LEVEL = "MS:1000511";

    // Spectrum type
    constexpr const char* PROFILE_SPECTRUM = "MS:1000128";
    constexpr const char* CENTROID_SPECTRUM = "MS:1000127";

    // Polarity
    constexpr const char* POSITIVE_SCAN = "MS:1000130";
    constexpr const char* NEGATIVE_SCAN = "MS:1000129";

    // Binary data type
    constexpr const char* MZ_ARRAY = "MS:1000514";
    constexpr const char* INTENSITY_ARRAY = "MS:1000515";
    constexpr const char* TIME_ARRAY = "MS:1000595";

    // Binary encoding
    constexpr const char* FLOAT_64_BIT = "MS:1000523";
    constexpr const char* FLOAT_32_BIT = "MS:1000521";
    constexpr const char* ZLIB_COMPRESSION = "MS:1000574";
    constexpr const char* NO_COMPRESSION = "MS:1000576";

    // Scan time
    constexpr const char* SCAN_START_TIME = "MS:1000016";

    // TIC
    constexpr const char* TIC = "MS:1000285";
    constexpr const char* BASE_PEAK_MZ = "MS:1000504";
    constexpr const char* BASE_PEAK_INTENSITY = "MS:1000505";

    // Precursor
    constexpr const char* SELECTED_ION_MZ = "MS:1000744";
    constexpr const char* PEAK_INTENSITY = "MS:1000042";
    constexpr const char* CHARGE_STATE = "MS:1000041";
    constexpr const char* ISOLATION_WINDOW_TARGET = "MS:1000827";
    constexpr const char* ISOLATION_WINDOW_LOWER = "MS:1000828";
    constexpr const char* ISOLATION_WINDOW_UPPER = "MS:1000829";
    constexpr const char* COLLISION_ENERGY = "MS:1000045";

    // Activation methods
    constexpr const char* CID = "MS:1000133";
    constexpr const char* HCD = "MS:1000422";
    constexpr const char* ETD = "MS:1000598";

    // Chromatogram types
    constexpr const char* TIC_CHROMATOGRAM = "MS:1000235";
    constexpr const char* BPC_CHROMATOGRAM = "MS:1000628";
    constexpr const char* SRM_CHROMATOGRAM = "MS:1001473";

    // Time units
    constexpr const char* MINUTE = "UO:0000031";
    constexpr const char* SECOND = "UO:0000010";
}

class MzMLReader::Impl {
public:
    Impl() = default;

    MSExperiment read(const std::string& filename, const MzMLReaderOptions& options) {
#ifdef MASSKIT_HAS_PUGIXML
        return readWithPugixml(filename, options);
#else
        return readFallback(filename, options);
#endif
    }

    MSExperiment parseString(const std::string& content,
                             const MzMLReaderOptions& options) {
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
            throw MzMLParseError("Failed to load file: " + filename);
        }
        auto spectrum_list = doc.select_node("//spectrumList");
        if (spectrum_list) {
            return static_cast<std::size_t>(
                spectrum_list.node().attribute("count").as_uint());
        }
        return 0;
#else
        // Simple regex-based counting
        std::ifstream file(filename);
        if (!file) {
            throw MzMLParseError("Failed to open file: " + filename);
        }
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        std::regex count_regex("spectrumList[^>]*count=\"(\\d+)\"");
        std::smatch match;
        if (std::regex_search(content, match, count_regex)) {
            return std::stoull(match[1].str());
        }
        return 0;
#endif
    }

    std::size_t countChromatograms(const std::string& filename) {
#ifdef MASSKIT_HAS_PUGIXML
        pugi::xml_document doc;
        if (!doc.load_file(filename.c_str())) {
            throw MzMLParseError("Failed to load file: " + filename);
        }
        auto chrom_list = doc.select_node("//chromatogramList");
        if (chrom_list) {
            return static_cast<std::size_t>(
                chrom_list.node().attribute("count").as_uint());
        }
        return 0;
#else
        std::ifstream file(filename);
        if (!file) {
            throw MzMLParseError("Failed to open file: " + filename);
        }
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        std::regex count_regex("chromatogramList[^>]*count=\"(\\d+)\"");
        std::smatch match;
        if (std::regex_search(content, match, count_regex)) {
            return std::stoull(match[1].str());
        }
        return 0;
#endif
    }

private:
#ifdef MASSKIT_HAS_PUGIXML
    MSExperiment readWithPugixml(const std::string& filename,
                                  const MzMLReaderOptions& options) {
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(filename.c_str());

        if (!result) {
            throw MzMLParseError("Failed to parse file: " + std::string(result.description()));
        }

        MSExperiment exp;
        exp.setSourceFile(filename);

        parseDocument(doc, exp, options);
        return exp;
    }

    MSExperiment parseStringWithPugixml(const std::string& content,
                                         const MzMLReaderOptions& options) {
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_string(content.c_str());

        if (!result) {
            throw MzMLParseError("Failed to parse content: " + std::string(result.description()));
        }

        MSExperiment exp;
        parseDocument(doc, exp, options);
        return exp;
    }

    void parseDocument(pugi::xml_document& doc, MSExperiment& exp,
                        const MzMLReaderOptions& options) {
        // Find the mzML root element
        auto mzml = doc.child("indexedmzML").child("mzML");
        if (!mzml) {
            mzml = doc.child("mzML");
        }
        if (!mzml) {
            throw MzMLParseError("No mzML element found");
        }

        // Parse run metadata
        auto run = mzml.child("run");
        if (run) {
            if (auto id = run.attribute("id")) {
                exp.metadata()["run_id"] = id.value();
            }
            if (auto start_time = run.attribute("startTimeStamp")) {
                exp.setDateTime(start_time.value());
            }
        }

        // Parse instrument configuration
        for (auto inst : mzml.child("instrumentConfigurationList").children("instrumentConfiguration")) {
            for (auto cv : inst.children("cvParam")) {
                std::string acc = cv.attribute("accession").value();
                if (acc.substr(0, 3) == "MS:") {
                    if (auto name = cv.attribute("name")) {
                        exp.setInstrumentModel(name.value());
                        break;
                    }
                }
            }
        }

        // Parse spectra
        if (!options.skip_spectra) {
            auto spectrum_list = run.child("spectrumList");
            if (spectrum_list) {
                std::size_t count = spectrum_list.attribute("count").as_uint();
                exp.reserveSpectra(count);

                int progress = 0;
                for (auto spectrum_node : spectrum_list.children("spectrum")) {
                    if (options.max_spectra > 0 && exp.spectrumCount() >= options.max_spectra) {
                        break;
                    }

                    Spectrum spec = parseSpectrum(spectrum_node, options);

                    // Apply filters
                    if (!options.ms_levels.empty()) {
                        auto it = std::find(options.ms_levels.begin(),
                                            options.ms_levels.end(),
                                            spec.msLevel());
                        if (it == options.ms_levels.end()) continue;
                    }

                    if (options.rt_range) {
                        if (!options.rt_range->contains(spec.retentionTime())) continue;
                    }

                    exp.addSpectrum(std::move(spec));

                    // Progress callback
                    if (options.progress_callback) {
                        if (!options.progress_callback(++progress, static_cast<int>(count))) {
                            break;  // User cancelled
                        }
                    }
                }
            }
        }

        // Parse chromatograms
        if (!options.skip_chromatograms) {
            auto chrom_list = run.child("chromatogramList");
            if (chrom_list) {
                for (auto chrom_node : chrom_list.children("chromatogram")) {
                    Chromatogram chrom = parseChromatogram(chrom_node, options);
                    exp.addChromatogram(std::move(chrom));
                }
            }
        }
    }

    Spectrum parseSpectrum(pugi::xml_node node, const MzMLReaderOptions& options) {
        Spectrum spec;

        // Parse attributes
        spec.setNativeId(node.attribute("id").value());

        // Parse cvParams
        for (auto cv : node.children("cvParam")) {
            std::string acc = cv.attribute("accession").value();

            if (acc == cv::MS_LEVEL) {
                spec.setMsLevel(static_cast<MSLevel>(cv.attribute("value").as_int()));
            } else if (acc == cv::PROFILE_SPECTRUM) {
                spec.setType(SpectrumType::PROFILE);
            } else if (acc == cv::CENTROID_SPECTRUM) {
                spec.setType(SpectrumType::CENTROID);
            } else if (acc == cv::POSITIVE_SCAN) {
                spec.setPolarity(Polarity::POSITIVE);
            } else if (acc == cv::NEGATIVE_SCAN) {
                spec.setPolarity(Polarity::NEGATIVE);
            } else if (acc == cv::TIC) {
                spec.setTic(cv.attribute("value").as_double());
            }
        }

        // Parse scan list for RT
        auto scan_list = node.child("scanList");
        if (scan_list) {
            auto scan = scan_list.child("scan");
            if (scan) {
                for (auto cv : scan.children("cvParam")) {
                    std::string acc = cv.attribute("accession").value();
                    if (acc == cv::SCAN_START_TIME) {
                        double rt = cv.attribute("value").as_double();
                        // Check unit (convert minutes to seconds)
                        std::string unit = cv.attribute("unitAccession").value();
                        if (unit == cv::MINUTE) {
                            rt *= 60.0;
                        }
                        spec.setRetentionTime(rt);
                    }
                }
            }
        }

        // Parse precursor info
        auto precursor_list = node.child("precursorList");
        if (precursor_list) {
            for (auto prec_node : precursor_list.children("precursor")) {
                Precursor prec;

                // Isolation window
                auto isolation = prec_node.child("isolationWindow");
                if (isolation) {
                    for (auto cv : isolation.children("cvParam")) {
                        std::string acc = cv.attribute("accession").value();
                        if (acc == cv::ISOLATION_WINDOW_TARGET) {
                            prec.mz = cv.attribute("value").as_double();
                        } else if (acc == cv::ISOLATION_WINDOW_LOWER) {
                            prec.isolation_window_lower = cv.attribute("value").as_double();
                        } else if (acc == cv::ISOLATION_WINDOW_UPPER) {
                            prec.isolation_window_upper = cv.attribute("value").as_double();
                        }
                    }
                }

                // Selected ion
                auto selected = prec_node.child("selectedIonList").child("selectedIon");
                if (selected) {
                    for (auto cv : selected.children("cvParam")) {
                        std::string acc = cv.attribute("accession").value();
                        if (acc == cv::SELECTED_ION_MZ) {
                            prec.mz = cv.attribute("value").as_double();
                        } else if (acc == cv::PEAK_INTENSITY) {
                            prec.intensity = cv.attribute("value").as_double();
                        } else if (acc == cv::CHARGE_STATE) {
                            prec.charge = static_cast<ChargeState>(cv.attribute("value").as_int());
                        }
                    }
                }

                // Activation
                auto activation = prec_node.child("activation");
                if (activation) {
                    for (auto cv : activation.children("cvParam")) {
                        std::string acc = cv.attribute("accession").value();
                        if (acc == cv::CID) {
                            prec.activation = ActivationMethod::CID;
                        } else if (acc == cv::HCD) {
                            prec.activation = ActivationMethod::HCD;
                        } else if (acc == cv::ETD) {
                            prec.activation = ActivationMethod::ETD;
                        } else if (acc == cv::COLLISION_ENERGY) {
                            prec.collision_energy = cv.attribute("value").as_double();
                        }
                    }
                }

                spec.addPrecursor(prec);
            }
        }

        // Parse binary data
        auto binary_list = node.child("binaryDataArrayList");
        if (binary_list) {
            std::vector<double> mz_data, intensity_data;

            for (auto binary : binary_list.children("binaryDataArray")) {
                bool is_mz = false;
                bool is_intensity = false;
                bool is_64bit = false;
                bool is_compressed = false;

                for (auto cv : binary.children("cvParam")) {
                    std::string acc = cv.attribute("accession").value();
                    if (acc == cv::MZ_ARRAY) is_mz = true;
                    else if (acc == cv::INTENSITY_ARRAY) is_intensity = true;
                    else if (acc == cv::FLOAT_64_BIT) is_64bit = true;
                    else if (acc == cv::FLOAT_32_BIT) is_64bit = false;
                    else if (acc == cv::ZLIB_COMPRESSION) is_compressed = true;
                }

                std::string base64_data = binary.child("binary").child_value();

                // Remove whitespace
                base64_data.erase(
                    std::remove_if(base64_data.begin(), base64_data.end(), ::isspace),
                    base64_data.end());

                if (base64_data.empty()) continue;

                std::vector<double> decoded;
                try {
                    if (is_compressed) {
#ifdef MASSKIT_HAS_ZLIB
                        if (is_64bit) {
                            decoded = Zlib::decompressFloat64(base64_data, true);
                        } else {
                            auto floats = Zlib::decompressFloat32(base64_data, true);
                            decoded.assign(floats.begin(), floats.end());
                        }
#else
                        throw MzMLParseError("Compressed data requires zlib support");
#endif
                    } else {
                        if (is_64bit) {
                            decoded = Base64::decodeFloat64(base64_data, true);
                        } else {
                            auto floats = Base64::decodeFloat32(base64_data, true);
                            decoded.assign(floats.begin(), floats.end());
                        }
                    }
                } catch (const std::exception& e) {
                    throw MzMLParseError("Failed to decode binary data: " + std::string(e.what()));
                }

                if (is_mz) {
                    mz_data = std::move(decoded);
                } else if (is_intensity && options.load_intensity) {
                    intensity_data = std::move(decoded);
                }
            }

            // Set spectrum data
            if (!mz_data.empty()) {
                if (intensity_data.empty()) {
                    intensity_data.resize(mz_data.size(), 0.0);
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

    Chromatogram parseChromatogram(pugi::xml_node node,
                                    [[maybe_unused]] const MzMLReaderOptions& options) {
        Chromatogram chrom;

        chrom.setNativeId(node.attribute("id").value());

        // Parse cvParams
        for (auto cv : node.children("cvParam")) {
            std::string acc = cv.attribute("accession").value();
            if (acc == cv::TIC_CHROMATOGRAM) {
                chrom.setType(ChromatogramType::TIC);
            } else if (acc == cv::BPC_CHROMATOGRAM) {
                chrom.setType(ChromatogramType::BPC);
            } else if (acc == cv::SRM_CHROMATOGRAM) {
                chrom.setType(ChromatogramType::SRM);
            }
        }

        // Parse binary data
        auto binary_list = node.child("binaryDataArrayList");
        if (binary_list) {
            std::vector<double> rt_data, intensity_data;

            for (auto binary : binary_list.children("binaryDataArray")) {
                bool is_time = false;
                bool is_intensity = false;
                bool is_64bit = false;
                bool is_compressed = false;
                bool is_minutes = false;

                for (auto cv : binary.children("cvParam")) {
                    std::string acc = cv.attribute("accession").value();
                    if (acc == cv::TIME_ARRAY) is_time = true;
                    else if (acc == cv::INTENSITY_ARRAY) is_intensity = true;
                    else if (acc == cv::FLOAT_64_BIT) is_64bit = true;
                    else if (acc == cv::FLOAT_32_BIT) is_64bit = false;
                    else if (acc == cv::ZLIB_COMPRESSION) is_compressed = true;

                    std::string unit = cv.attribute("unitAccession").value();
                    if (unit == cv::MINUTE) is_minutes = true;
                }

                std::string base64_data = binary.child("binary").child_value();
                base64_data.erase(
                    std::remove_if(base64_data.begin(), base64_data.end(), ::isspace),
                    base64_data.end());

                if (base64_data.empty()) continue;

                std::vector<double> decoded;
                if (is_compressed) {
#ifdef MASSKIT_HAS_ZLIB
                    if (is_64bit) {
                        decoded = Zlib::decompressFloat64(base64_data, true);
                    } else {
                        auto floats = Zlib::decompressFloat32(base64_data, true);
                        decoded.assign(floats.begin(), floats.end());
                    }
#else
                    throw MzMLParseError("Compressed data requires zlib support");
#endif
                } else {
                    if (is_64bit) {
                        decoded = Base64::decodeFloat64(base64_data, true);
                    } else {
                        auto floats = Base64::decodeFloat32(base64_data, true);
                        decoded.assign(floats.begin(), floats.end());
                    }
                }

                if (is_time) {
                    if (is_minutes) {
                        for (auto& v : decoded) v *= 60.0;
                    }
                    rt_data = std::move(decoded);
                } else if (is_intensity) {
                    intensity_data = std::move(decoded);
                }
            }

            if (!rt_data.empty() && !intensity_data.empty()) {
                chrom.setData(std::move(rt_data), std::move(intensity_data));
            }
        }

        return chrom;
    }
#endif // MASSKIT_HAS_PUGIXML

    // Fallback implementation using simple regex parsing
    MSExperiment readFallback(const std::string& filename,
                               const MzMLReaderOptions& options) {
        std::ifstream file(filename);
        if (!file) {
            throw MzMLParseError("Failed to open file: " + filename);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return parseStringFallback(buffer.str(), options);
    }

    MSExperiment parseStringFallback(const std::string& content,
                                      [[maybe_unused]] const MzMLReaderOptions& options) {
        // Very basic fallback parser - only extracts minimal info
        MSExperiment exp;

        // This is a placeholder - real implementation would need proper parsing
        // For production use, pugixml should be available
        if (content.find("<mzML") == std::string::npos &&
            content.find("<indexedmzML") == std::string::npos) {
            throw MzMLParseError("Not a valid mzML file");
        }

        exp.metadata()["warning"] = "Loaded with fallback parser - limited functionality";
        return exp;
    }
};

MzMLReader::MzMLReader() : impl_(std::make_unique<Impl>()) {}
MzMLReader::~MzMLReader() = default;
MzMLReader::MzMLReader(MzMLReader&&) noexcept = default;
MzMLReader& MzMLReader::operator=(MzMLReader&&) noexcept = default;

MSExperiment MzMLReader::read(const std::string& filename) {
    return read(filename, default_options_);
}

MSExperiment MzMLReader::read(const std::string& filename,
                              const MzMLReaderOptions& options) {
    try {
        return impl_->read(filename, options);
    } catch (const MzMLParseError&) {
        throw;
    } catch (const std::exception& e) {
        last_error_ = e.what();
        throw MzMLParseError(e.what());
    }
}

MSExperiment MzMLReader::parseString(const std::string& content) {
    return parseString(content, default_options_);
}

MSExperiment MzMLReader::parseString(const std::string& content,
                                     const MzMLReaderOptions& options) {
    try {
        return impl_->parseString(content, options);
    } catch (const MzMLParseError&) {
        throw;
    } catch (const std::exception& e) {
        last_error_ = e.what();
        throw MzMLParseError(e.what());
    }
}

std::size_t MzMLReader::countSpectra(const std::string& filename) {
    return impl_->countSpectra(filename);
}

std::size_t MzMLReader::countChromatograms(const std::string& filename) {
    return impl_->countChromatograms(filename);
}

bool MzMLReader::isValidMzML(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return false;

    // Read first 1KB and check for mzML markers
    char buffer[1024];
    file.read(buffer, sizeof(buffer));
    std::string content(buffer, static_cast<std::size_t>(file.gcount()));

    return content.find("<mzML") != std::string::npos ||
           content.find("<indexedmzML") != std::string::npos;
}

} // namespace io
} // namespace lcms
