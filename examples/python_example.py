#!/usr/bin/env python3
"""
Example usage of the MassKit library.

This script demonstrates the main features of the library:
- Loading mzML files
- Accessing spectra and chromatograms
- Peak picking
- Visualization
- Feature detection
"""

import numpy as np

# Import the library
try:
    import masskit
    from masskit import (
        Spectrum,
        Chromatogram,
        MSExperiment,
        load_mzml,
        pick_peaks,
        smooth_spectrum,
        correct_baseline,
    )
except ImportError:
    print("masskit not installed. Run: pip install -e python/")
    exit(1)


def create_synthetic_spectrum():
    """Create a synthetic spectrum for demonstration."""
    # Generate m/z values
    mz = np.arange(100, 1000, 0.1)

    # Create peaks with Gaussian shape
    intensity = np.zeros_like(mz)

    peaks = [
        (150.0, 1000),
        (250.5, 5000),
        (350.2, 3000),
        (500.7, 8000),
        (750.3, 2000),
    ]

    for peak_mz, peak_int in peaks:
        sigma = 0.1
        intensity += peak_int * np.exp(-0.5 * ((mz - peak_mz) / sigma) ** 2)

    # Add noise
    intensity += np.random.normal(0, 50, len(intensity))
    intensity = np.maximum(intensity, 0)

    return Spectrum(mz=mz, intensity=intensity, ms_level=1, rt=60.0)


def example_spectrum_operations():
    """Demonstrate basic spectrum operations."""
    print("=" * 60)
    print("Spectrum Operations")
    print("=" * 60)

    # Create a spectrum
    spec = create_synthetic_spectrum()
    print(f"Created spectrum: {spec}")
    print(f"  Size: {spec.size} points")
    print(f"  m/z range: {spec.mz_range}")
    print(f"  Base peak: m/z {spec.base_peak_mz:.4f}, intensity {spec.base_peak_intensity:.0f}")
    print(f"  TIC: {spec.tic:.0f}")

    # Extract a range
    extracted = spec.extract_range(200, 400)
    print(f"\nExtracted m/z 200-400: {extracted}")

    # Filter by intensity
    filtered = spec.filter_by_intensity(100)
    print(f"After intensity filter: {filtered.size} points")

    # Get top peaks
    top = spec.top_n(10)
    print(f"Top 10 peaks: {top.size} points")


def example_peak_picking():
    """Demonstrate peak picking."""
    print("\n" + "=" * 60)
    print("Peak Picking")
    print("=" * 60)

    # Create spectrum
    spec = create_synthetic_spectrum()

    # Smooth the spectrum first
    smoothed = smooth_spectrum(spec, method="gaussian", window_size=5)
    print(f"Smoothed spectrum")

    # Baseline correction
    corrected = correct_baseline(smoothed, method="snip")
    print(f"Baseline corrected")

    # Pick peaks
    peaks = pick_peaks(corrected, min_snr=5)
    print(f"\nDetected {len(peaks)} peaks:")

    peaks.sort_by_intensity()
    for i, peak in enumerate(peaks[:5]):
        print(f"  {i+1}. m/z {peak.mz:.4f}, intensity {peak.intensity:.0f}, SNR {peak.snr:.1f}")


def example_chromatogram():
    """Demonstrate chromatogram operations."""
    print("\n" + "=" * 60)
    print("Chromatogram Operations")
    print("=" * 60)

    # Create a synthetic chromatogram with a peak
    rt = np.arange(0, 600, 1.0)  # 10 minutes
    intensity = 1000 * np.exp(-0.5 * ((rt - 300) / 30) ** 2)  # Peak at 5 min
    intensity += np.random.normal(0, 10, len(rt))
    intensity = np.maximum(intensity, 0)

    from masskit import ChromatogramType
    chrom = Chromatogram(rt=rt, intensity=intensity, chrom_type=ChromatogramType.TIC)

    print(f"Created chromatogram: {chrom}")
    print(f"  Apex: {chrom.apex_rt:.1f} s")
    print(f"  Max intensity: {chrom.max_intensity:.0f}")
    print(f"  Area: {chrom.compute_area():.0f}")

    # Extract around peak
    peak_region = chrom.extract_range(250, 350)
    print(f"\nPeak region (250-350s):")
    print(f"  Area: {peak_region.compute_area():.0f}")


def example_experiment():
    """Demonstrate MSExperiment container."""
    print("\n" + "=" * 60)
    print("MSExperiment Container")
    print("=" * 60)

    # Create experiment
    exp = MSExperiment()

    # Add spectra at different retention times
    for rt in [60, 120, 180, 240, 300]:
        spec = create_synthetic_spectrum()
        spec.rt = float(rt)
        exp.add_spectrum(spec)

    print(f"Created experiment: {exp}")
    print(f"  Spectrum count: {exp.spectrum_count}")
    print(f"  m/z range: {exp.mz_range}")
    print(f"  RT range: {exp.rt_range}")

    # Generate TIC
    tic = exp.generate_tic()
    print(f"\nGenerated TIC: {tic}")

    # Generate XIC
    xic = exp.generate_xic(target_mz=500.7, tolerance=0.5)
    print(f"Generated XIC @ m/z 500.7: apex at {xic.apex_rt:.1f}s")


def example_file_io():
    """Demonstrate file I/O (if sample file available)."""
    print("\n" + "=" * 60)
    print("File I/O")
    print("=" * 60)

    # This would load a real mzML file
    print("To load an mzML file:")
    print("  exp = load_mzml('sample.mzML')")
    print("  exp = load_mzml('sample.mzML', ms_levels=[1], rt_range=(60, 300))")


def example_visualization():
    """Demonstrate visualization (if matplotlib available)."""
    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from masskit.visualization import plot_spectrum, plot_chromatogram

        print("Matplotlib available - creating plots...")

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot spectrum
        spec = create_synthetic_spectrum()
        plot_spectrum(spec, ax=axes[0], mz_range=(100, 600), title="Synthetic Spectrum")

        # Plot chromatogram
        rt = np.arange(0, 600, 1.0)
        intensity = 1000 * np.exp(-0.5 * ((rt - 300) / 30) ** 2)
        from masskit import ChromatogramType
        chrom = Chromatogram(rt=rt, intensity=intensity, chrom_type=ChromatogramType.TIC)
        plot_chromatogram(chrom, ax=axes[1], show_apex=True, title="Synthetic TIC")

        plt.tight_layout()
        plt.savefig("example_plots.png", dpi=100)
        print("Saved plots to example_plots.png")

    except ImportError:
        print("matplotlib not available - skipping visualization example")
        print("Install with: pip install matplotlib")


def main():
    """Run all examples."""
    print("MassKit Library Examples")
    print(f"Version: {masskit.__version__}")
    print()

    example_spectrum_operations()
    example_peak_picking()
    example_chromatogram()
    example_experiment()
    example_file_io()
    example_visualization()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
