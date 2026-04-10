#pragma once

/**
 * @file lcms.hpp
 * @brief Main header for the MassKit Core library.
 *
 * Include this header to get access to all MassKit core functionality.
 *
 * @example
 * @code
 * #include <lcms/lcms.hpp>
 *
 * int main() {
 *     // Load an mzML file
 *     auto exp = lcms::io::loadMzML("sample.mzML");
 *
 *     // Access spectra
 *     for (const auto& spectrum : exp.spectra()) {
 *         std::cout << "RT: " << spectrum.retentionTime() << "s\n";
 *     }
 *
 *     return 0;
 * }
 * @endcode
 */

// Core types
#include "types.hpp"

// Data structures
#include "spectrum.hpp"
#include "chromatogram.hpp"
#include "peak.hpp"
#include "feature_map.hpp"
#include "ms_experiment.hpp"

// I/O
#include "io/mzml_reader.hpp"
#include "io/mzxml_reader.hpp"
#include "io/base64.hpp"

// Algorithms
#include "algorithms/peak_picking.hpp"
#include "algorithms/baseline_correction.hpp"
#include "algorithms/smoothing.hpp"

/**
 * @namespace lcms
 * @brief Root namespace for the MassKit library.
 */

/**
 * @namespace lcms::io
 * @brief I/O utilities for reading and writing LC-MS data files.
 */

/**
 * @namespace lcms::algorithms
 * @brief Signal processing and analysis algorithms.
 */
