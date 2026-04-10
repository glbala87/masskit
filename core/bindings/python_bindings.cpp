/**
 * @file python_bindings.cpp
 * @brief pybind11 bindings for the MassKit C++ core library.
 *
 * Build with:
 *   pip install pybind11
 *   cmake -DMASSKIT_BUILD_PYTHON=ON ..
 *
 * Provides Python access to the high-performance C++ implementations
 * of spectrum processing, peak picking, isotope detection, and more.
 */

#ifdef MASSKIT_BUILD_PYTHON

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "lcms/spectrum.hpp"
#include "lcms/peak.hpp"
#include "lcms/chromatogram.hpp"
#include "lcms/feature_map.hpp"
#include "lcms/ms_experiment.hpp"
#include "lcms/types.hpp"
#include "lcms/algorithms/peak_picking.hpp"
#include "lcms/algorithms/isotope_detection.hpp"
#include "lcms/algorithms/spectral_matching.hpp"
#include "lcms/algorithms/quantification.hpp"
#include "lcms/parallel.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_masskit_core, m) {
    m.doc() = "MassKit C++ core library Python bindings";

    // =====================================================================
    // Enums
    // =====================================================================

    py::enum_<lcms::Polarity>(m, "Polarity")
        .value("UNKNOWN", lcms::Polarity::UNKNOWN)
        .value("POSITIVE", lcms::Polarity::POSITIVE)
        .value("NEGATIVE", lcms::Polarity::NEGATIVE);

    py::enum_<lcms::SpectrumType>(m, "SpectrumType")
        .value("UNKNOWN", lcms::SpectrumType::UNKNOWN)
        .value("PROFILE", lcms::SpectrumType::PROFILE)
        .value("CENTROID", lcms::SpectrumType::CENTROID);

    py::enum_<lcms::ActivationMethod>(m, "ActivationMethod")
        .value("UNKNOWN", lcms::ActivationMethod::UNKNOWN)
        .value("CID", lcms::ActivationMethod::CID)
        .value("HCD", lcms::ActivationMethod::HCD)
        .value("ETD", lcms::ActivationMethod::ETD);

    // =====================================================================
    // MZTolerance
    // =====================================================================

    py::class_<lcms::MZTolerance>(m, "MZTolerance")
        .def(py::init<>())
        .def(py::init<double, bool>(), py::arg("value"), py::arg("is_ppm") = false)
        .def_static("Da", &lcms::MZTolerance::Da)
        .def_static("PPM", &lcms::MZTolerance::PPM)
        .def("absolute_at", &lcms::MZTolerance::absoluteAt)
        .def("matches", &lcms::MZTolerance::matches);

    // =====================================================================
    // Spectrum
    // =====================================================================

    py::class_<lcms::Spectrum>(m, "Spectrum")
        .def(py::init<>())
        .def("size", &lcms::Spectrum::size)
        .def("empty", &lcms::Spectrum::empty)
        .def("mz_data", [](const lcms::Spectrum& s) {
            const auto& data = s.mzData();
            return py::array_t<double>(data.size(), data.data());
        })
        .def("intensity_data", [](const lcms::Spectrum& s) {
            const auto& data = s.intensityData();
            return py::array_t<double>(data.size(), data.data());
        })
        .def("set_data", [](lcms::Spectrum& s, py::array_t<double> mz, py::array_t<double> intensity) {
            auto mz_buf = mz.request();
            auto int_buf = intensity.request();
            std::vector<double> mz_vec(static_cast<double*>(mz_buf.ptr),
                                        static_cast<double*>(mz_buf.ptr) + mz_buf.size);
            std::vector<double> int_vec(static_cast<double*>(int_buf.ptr),
                                         static_cast<double*>(int_buf.ptr) + int_buf.size);
            s.setData(mz_vec, int_vec);
        })
        .def_property("ms_level", &lcms::Spectrum::msLevel, &lcms::Spectrum::setMsLevel)
        .def_property("rt", &lcms::Spectrum::rt, &lcms::Spectrum::setRt)
        .def_property("index", &lcms::Spectrum::index, &lcms::Spectrum::setIndex)
        .def("tic", &lcms::Spectrum::tic)
        .def("base_peak_mz", &lcms::Spectrum::basePeakMz)
        .def("base_peak_intensity", &lcms::Spectrum::basePeakIntensity)
        .def("mz_range", &lcms::Spectrum::mzRange);

    // =====================================================================
    // Peak
    // =====================================================================

    py::class_<lcms::Peak>(m, "Peak")
        .def(py::init<>())
        .def(py::init<lcms::MZ, lcms::Intensity, lcms::RetentionTime>(),
             py::arg("mz"), py::arg("intensity"), py::arg("rt") = 0.0)
        .def_property("mz", &lcms::Peak::mz, &lcms::Peak::setMz)
        .def_property("rt", &lcms::Peak::rt, &lcms::Peak::setRt)
        .def_property("intensity", &lcms::Peak::intensity, &lcms::Peak::setIntensity)
        .def_property("area", &lcms::Peak::area, &lcms::Peak::setArea)
        .def("neutral_mass", &lcms::Peak::neutralMass);

    // =====================================================================
    // PeakList
    // =====================================================================

    py::class_<lcms::PeakList>(m, "PeakList")
        .def(py::init<>())
        .def("size", &lcms::PeakList::size)
        .def("empty", &lcms::PeakList::empty)
        .def("add", &lcms::PeakList::add)
        .def("__len__", &lcms::PeakList::size)
        .def("__getitem__", [](const lcms::PeakList& pl, size_t i) {
            return pl[i];
        })
        .def("sort_by_mz", &lcms::PeakList::sortByMz)
        .def("sort_by_intensity", &lcms::PeakList::sortByIntensity);

    // =====================================================================
    // Peak Picking
    // =====================================================================

    py::class_<lcms::algorithms::PeakPickingOptions>(m, "PeakPickingOptions")
        .def(py::init<>())
        .def_readwrite("min_snr", &lcms::algorithms::PeakPickingOptions::min_snr)
        .def_readwrite("min_intensity", &lcms::algorithms::PeakPickingOptions::min_intensity)
        .def_readwrite("window_size", &lcms::algorithms::PeakPickingOptions::window_size)
        .def_readwrite("fit_peaks", &lcms::algorithms::PeakPickingOptions::fit_peaks);

    py::class_<lcms::algorithms::PeakPicker>(m, "PeakPicker")
        .def(py::init<>())
        .def(py::init<const lcms::algorithms::PeakPickingOptions&>())
        .def("pick", &lcms::algorithms::PeakPicker::pick)
        .def("centroid", &lcms::algorithms::PeakPicker::centroid)
        .def("estimate_noise", &lcms::algorithms::PeakPicker::estimateNoise);

    m.def("pick_peaks", &lcms::algorithms::pickPeaks,
          py::arg("spectrum"), py::arg("options") = lcms::algorithms::PeakPickingOptions{},
          "Pick peaks from a spectrum");

    m.def("centroid_spectrum", &lcms::algorithms::centroidSpectrum,
          py::arg("spectrum"), py::arg("options") = lcms::algorithms::PeakPickingOptions{},
          "Convert profile spectrum to centroided");

    // =====================================================================
    // Isotope Detection
    // =====================================================================

    py::class_<lcms::algorithms::IsotopePattern>(m, "IsotopePattern")
        .def(py::init<>())
        .def_readwrite("monoisotopic_mz", &lcms::algorithms::IsotopePattern::monoisotopic_mz)
        .def_readwrite("charge", &lcms::algorithms::IsotopePattern::charge)
        .def_readwrite("score", &lcms::algorithms::IsotopePattern::score)
        .def_readwrite("rt", &lcms::algorithms::IsotopePattern::rt)
        .def("neutral_mass", &lcms::algorithms::IsotopePattern::neutralMass)
        .def("total_intensity", &lcms::algorithms::IsotopePattern::totalIntensity)
        .def("num_peaks", &lcms::algorithms::IsotopePattern::numPeaks);

    py::class_<lcms::algorithms::DeconvolutedMass>(m, "DeconvolutedMass")
        .def(py::init<>())
        .def_readwrite("neutral_mass", &lcms::algorithms::DeconvolutedMass::neutral_mass)
        .def_readwrite("intensity", &lcms::algorithms::DeconvolutedMass::intensity)
        .def_readwrite("quality_score", &lcms::algorithms::DeconvolutedMass::quality_score)
        .def("num_charge_states", &lcms::algorithms::DeconvolutedMass::numChargeStates);

    py::class_<lcms::algorithms::IsotopeDetectionOptions>(m, "IsotopeDetectionOptions")
        .def(py::init<>())
        .def_readwrite("min_charge", &lcms::algorithms::IsotopeDetectionOptions::min_charge)
        .def_readwrite("max_charge", &lcms::algorithms::IsotopeDetectionOptions::max_charge)
        .def_readwrite("mz_tolerance", &lcms::algorithms::IsotopeDetectionOptions::mz_tolerance)
        .def_readwrite("min_peaks", &lcms::algorithms::IsotopeDetectionOptions::min_peaks)
        .def_readwrite("min_score", &lcms::algorithms::IsotopeDetectionOptions::min_score);

    m.def("averagine_distribution", &lcms::algorithms::averagineDistribution,
          py::arg("mass"), py::arg("num_peaks") = 6);

    m.def("detect_isotope_patterns", &lcms::algorithms::detectIsotopePatterns,
          py::arg("spectrum"), py::arg("options") = lcms::algorithms::IsotopeDetectionOptions{});

    m.def("deconvolute_spectrum", &lcms::algorithms::deconvoluteSpectrum,
          py::arg("spectrum"), py::arg("options") = lcms::algorithms::IsotopeDetectionOptions{});

    // =====================================================================
    // Spectral Matching
    // =====================================================================

    m.def("cosine_similarity", &lcms::algorithms::cosineSimilarity,
          py::arg("mz1"), py::arg("int1"), py::arg("mz2"), py::arg("int2"),
          py::arg("tolerance") = 0.01);

    m.def("modified_cosine_similarity", &lcms::algorithms::modifiedCosineSimilarity,
          py::arg("mz1"), py::arg("int1"), py::arg("precursor1"),
          py::arg("mz2"), py::arg("int2"), py::arg("precursor2"),
          py::arg("tolerance") = 0.01);

    // =====================================================================
    // Parallel Processing
    // =====================================================================

    m.def("parallel_peak_picking", &lcms::parallelPeakPicking,
          py::arg("spectra"), py::arg("min_snr") = 3.0, py::arg("num_threads") = 0,
          "Pick peaks from multiple spectra in parallel using C++ threads");

    // =====================================================================
    // Version
    // =====================================================================

    m.attr("__version__") = "1.0.0";
}

#endif // MASSKIT_BUILD_PYTHON
