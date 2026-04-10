/**
 * Example usage of the MassKit C++ library.
 *
 * Compile with:
 *   g++ -std=c++17 -I../core/include cpp_example.cpp -o example
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "lcms/lcms.hpp"

using namespace lcms;

/**
 * Create a synthetic spectrum for demonstration.
 */
Spectrum createSyntheticSpectrum(double rt) {
    std::vector<MZ> mz;
    std::vector<Intensity> intensity;

    // Generate m/z values
    for (double m = 100.0; m < 1000.0; m += 0.1) {
        mz.push_back(m);
    }

    intensity.resize(mz.size(), 0.0);

    // Add peaks with Gaussian shape
    std::vector<std::pair<double, double>> peaks = {
        {150.0, 1000.0},
        {250.5, 5000.0},
        {350.2, 3000.0},
        {500.7, 8000.0},
        {750.3, 2000.0}
    };

    for (const auto& [peak_mz, peak_int] : peaks) {
        double sigma = 0.1;
        for (size_t i = 0; i < mz.size(); ++i) {
            double diff = mz[i] - peak_mz;
            intensity[i] += peak_int * std::exp(-0.5 * diff * diff / (sigma * sigma));
        }
    }

    // Add noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 50);

    for (auto& val : intensity) {
        val = std::max(0.0, val + noise(gen));
    }

    Spectrum spec(std::move(mz), std::move(intensity));
    spec.setRetentionTime(rt);
    spec.setMsLevel(1);
    spec.setType(SpectrumType::PROFILE);
    spec.setPolarity(Polarity::POSITIVE);

    return spec;
}

void exampleSpectrumOperations() {
    std::cout << "========================================\n";
    std::cout << "Spectrum Operations\n";
    std::cout << "========================================\n";

    // Create a spectrum
    auto spec = createSyntheticSpectrum(60.0);

    std::cout << "Created spectrum:\n";
    std::cout << "  Size: " << spec.size() << " points\n";
    std::cout << "  m/z range: [" << spec.mzRange().min_value
              << ", " << spec.mzRange().max_value << "]\n";
    std::cout << "  Base peak: m/z " << spec.basePeakMz()
              << ", intensity " << spec.basePeakIntensity() << "\n";
    std::cout << "  TIC: " << spec.tic() << "\n";

    // Check if sorted
    std::cout << "  Sorted: " << (spec.isSortedByMz() ? "yes" : "no") << "\n";

    // Extract a range
    auto extracted = spec.extractRange(200.0, 400.0);
    std::cout << "\nExtracted m/z 200-400: " << extracted.size() << " points\n";

    // Find nearest peak
    Index nearest = spec.findNearestMz(500.0);
    std::cout << "Nearest to m/z 500: index " << nearest
              << ", m/z " << spec.mzAt(nearest) << "\n";
}

void exampleMSExperiment() {
    std::cout << "\n========================================\n";
    std::cout << "MSExperiment Container\n";
    std::cout << "========================================\n";

    MSExperiment exp;
    exp.setSourceFile("synthetic_data.mzML");
    exp.setInstrumentModel("Synthetic Mass Spectrometer");

    // Add spectra at different retention times
    for (double rt : {60.0, 120.0, 180.0, 240.0, 300.0}) {
        exp.addSpectrum(createSyntheticSpectrum(rt));
    }

    std::cout << "Created experiment:\n";
    std::cout << "  Source: " << exp.sourceFile() << "\n";
    std::cout << "  Spectra: " << exp.spectrumCount() << "\n";
    std::cout << "  m/z range: [" << exp.mzRange().min_value
              << ", " << exp.mzRange().max_value << "]\n";
    std::cout << "  RT range: [" << exp.rtRange().min_value
              << ", " << exp.rtRange().max_value << "] s\n";
    std::cout << "  Total data points: " << exp.totalDataPoints() << "\n";
    std::cout << "  Avg spectrum size: " << exp.averageSpectrumSize() << "\n";

    // Generate TIC
    auto tic = exp.generateTIC(1);
    std::cout << "\nGenerated TIC:\n";
    std::cout << "  Points: " << tic.size() << "\n";
    std::cout << "  Max intensity: " << tic.maxIntensity() << "\n";
    std::cout << "  Apex RT: " << tic.rtAtMaxIntensity() << " s\n";

    // Generate XIC
    auto xic = exp.generateXIC(500.7, MZTolerance::Da(0.5), 1);
    std::cout << "\nGenerated XIC @ m/z 500.7:\n";
    std::cout << "  Points: " << xic.size() << "\n";
    std::cout << "  Max intensity: " << xic.maxIntensity() << "\n";

    // Find spectrum by RT
    auto* nearest_spec = exp.findSpectrumByRt(150.0, 1);
    if (nearest_spec) {
        std::cout << "\nNearest spectrum to RT 150s: RT "
                  << nearest_spec->retentionTime() << "s\n";
    }

    // Get spectra by MS level
    auto ms1_spectra = exp.getSpectraByLevel(1);
    std::cout << "MS1 spectra: " << ms1_spectra.size() << "\n";
}

void examplePeakList() {
    std::cout << "\n========================================\n";
    std::cout << "PeakList Operations\n";
    std::cout << "========================================\n";

    PeakList peaks;

    // Add some peaks
    peaks.add(Peak(150.0, 1000.0, 60.0));
    peaks.add(Peak(250.5, 5000.0, 60.0));
    peaks.add(Peak(350.2, 3000.0, 60.0));
    peaks.add(Peak(500.7, 8000.0, 60.0));

    std::cout << "Added " << peaks.size() << " peaks\n";

    // Sort by intensity
    peaks.sortByIntensity();
    std::cout << "\nTop peaks by intensity:\n";
    for (size_t i = 0; i < peaks.size(); ++i) {
        std::cout << "  " << (i+1) << ". m/z " << peaks[i].mz()
                  << ", intensity " << peaks[i].intensity() << "\n";
    }

    // Find peaks in range
    auto filtered = peaks.findInMzRange(200.0, 400.0);
    std::cout << "\nPeaks in m/z 200-400: " << filtered.size() << "\n";

    // Find nearest
    auto* nearest = peaks.findNearestMz(300.0);
    if (nearest) {
        std::cout << "Nearest to m/z 300: " << nearest->mz() << "\n";
    }
}

void exampleTypes() {
    std::cout << "\n========================================\n";
    std::cout << "Type Utilities\n";
    std::cout << "========================================\n";

    // MZTolerance
    auto tol_da = MZTolerance::Da(0.01);
    auto tol_ppm = MZTolerance::PPM(10);

    std::cout << "Tolerance examples:\n";
    std::cout << "  0.01 Da at m/z 500: " << tol_da.absoluteAt(500.0) << " Da\n";
    std::cout << "  10 ppm at m/z 500: " << tol_ppm.absoluteAt(500.0) << " Da\n";
    std::cout << "  10 ppm at m/z 1000: " << tol_ppm.absoluteAt(1000.0) << " Da\n";

    // Matches
    std::cout << "\nMatching:\n";
    std::cout << "  500.001 vs 500.005 with 0.01 Da: "
              << (tol_da.matches(500.001, 500.005) ? "match" : "no match") << "\n";
    std::cout << "  500.001 vs 500.020 with 0.01 Da: "
              << (tol_da.matches(500.001, 500.020) ? "match" : "no match") << "\n";

    // Range
    MZRange range(100.0, 500.0);
    std::cout << "\nRange operations:\n";
    std::cout << "  Range [100, 500] span: " << range.span() << "\n";
    std::cout << "  Range [100, 500] center: " << range.center() << "\n";
    std::cout << "  Contains 300: " << (range.contains(300.0) ? "yes" : "no") << "\n";
    std::cout << "  Contains 600: " << (range.contains(600.0) ? "yes" : "no") << "\n";
}

int main() {
    std::cout << "MassKit C++ Library Examples\n\n";

    try {
        exampleSpectrumOperations();
        exampleMSExperiment();
        examplePeakList();
        exampleTypes();

        std::cout << "\n========================================\n";
        std::cout << "Examples completed successfully!\n";
        std::cout << "========================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
