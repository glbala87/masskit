"""
Visualization functions for LC-MS data.
"""

from typing import Optional, List, Tuple, Union, Any
import numpy as np

from .spectrum import Spectrum
from .chromatogram import Chromatogram
from .peak import Peak, PeakList
from .experiment import MSExperiment


def plot_spectrum(
    spectrum: Spectrum,
    ax: Optional[Any] = None,
    mz_range: Optional[Tuple[float, float]] = None,
    color: str = "steelblue",
    linewidth: float = 0.8,
    fill: bool = False,
    fill_alpha: float = 0.3,
    show_base_peak: bool = False,
    title: Optional[str] = None,
    xlabel: str = "m/z",
    ylabel: str = "Intensity",
    **kwargs,
) -> Any:
    """
    Plot a mass spectrum.

    Args:
        spectrum: Spectrum to plot
        ax: Matplotlib axes (None = create new figure)
        mz_range: (min, max) m/z range to display
        color: Line color
        linewidth: Line width
        fill: Fill area under curve
        fill_alpha: Fill transparency
        show_base_peak: Annotate base peak
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        **kwargs: Additional plot parameters

    Returns:
        Matplotlib axes

    Example:
        >>> import matplotlib.pyplot as plt
        >>> plot_spectrum(spectrum, mz_range=(100, 500))
        >>> plt.show()
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    mz = spectrum.mz
    intensity = spectrum.intensity

    # Apply m/z range filter
    if mz_range is not None:
        mask = (mz >= mz_range[0]) & (mz <= mz_range[1])
        mz = mz[mask]
        intensity = intensity[mask]

    if len(mz) == 0:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        return ax

    # Plot spectrum
    if spectrum.spectrum_type.name == "CENTROID":
        # Stem plot for centroided data
        markerline, stemlines, baseline = ax.stem(
            mz, intensity, linefmt=color, markerfmt=" ", basefmt=" ", **kwargs
        )
        plt.setp(stemlines, linewidth=linewidth)
    else:
        # Line plot for profile data
        ax.plot(mz, intensity, color=color, linewidth=linewidth, **kwargs)
        if fill:
            ax.fill_between(mz, intensity, alpha=fill_alpha, color=color)

    # Annotate base peak
    if show_base_peak and len(intensity) > 0:
        max_idx = np.argmax(intensity)
        ax.annotate(
            f"{mz[max_idx]:.4f}",
            xy=(mz[max_idx], intensity[max_idx]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"MS{spectrum.ms_level} @ {spectrum.rt:.2f}s")

    ax.set_xlim(mz_range if mz_range else (mz.min(), mz.max()))
    ax.set_ylim(bottom=0)

    return ax


def plot_chromatogram(
    chromatogram: Chromatogram,
    ax: Optional[Any] = None,
    rt_range: Optional[Tuple[float, float]] = None,
    color: str = "steelblue",
    linewidth: float = 1.0,
    fill: bool = True,
    fill_alpha: float = 0.3,
    show_apex: bool = False,
    title: Optional[str] = None,
    xlabel: str = "Retention Time (s)",
    ylabel: str = "Intensity",
    **kwargs,
) -> Any:
    """
    Plot a chromatogram.

    Args:
        chromatogram: Chromatogram to plot
        ax: Matplotlib axes (None = create new figure)
        rt_range: (min, max) RT range to display
        color: Line color
        linewidth: Line width
        fill: Fill area under curve
        fill_alpha: Fill transparency
        show_apex: Annotate apex
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        **kwargs: Additional plot parameters

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    rt = chromatogram.rt
    intensity = chromatogram.intensity

    # Apply RT range filter
    if rt_range is not None:
        mask = (rt >= rt_range[0]) & (rt <= rt_range[1])
        rt = rt[mask]
        intensity = intensity[mask]

    if len(rt) == 0:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        return ax

    # Plot chromatogram
    ax.plot(rt, intensity, color=color, linewidth=linewidth, **kwargs)
    if fill:
        ax.fill_between(rt, intensity, alpha=fill_alpha, color=color)

    # Annotate apex
    if show_apex and len(intensity) > 0:
        max_idx = np.argmax(intensity)
        ax.annotate(
            f"RT: {rt[max_idx]:.1f}s",
            xy=(rt[max_idx], intensity[max_idx]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
        ax.axvline(rt[max_idx], color="red", linestyle="--", alpha=0.5, linewidth=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)
    else:
        type_name = chromatogram.chrom_type.name
        if chromatogram.target_mz > 0:
            ax.set_title(f"{type_name} @ m/z {chromatogram.target_mz:.4f}")
        else:
            ax.set_title(type_name)

    ax.set_xlim(rt_range if rt_range else (rt.min(), rt.max()))
    ax.set_ylim(bottom=0)

    return ax


def plot_peaks(
    spectrum: Spectrum,
    peaks: PeakList,
    ax: Optional[Any] = None,
    spectrum_color: str = "gray",
    peak_color: str = "red",
    annotate: bool = True,
    max_annotations: int = 10,
    **kwargs,
) -> Any:
    """
    Plot spectrum with detected peaks annotated.

    Args:
        spectrum: Original spectrum
        peaks: Detected peaks
        ax: Matplotlib axes
        spectrum_color: Spectrum color
        peak_color: Peak marker color
        annotate: Add m/z annotations
        max_annotations: Maximum peaks to annotate
        **kwargs: Additional parameters

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Plot spectrum
    plot_spectrum(spectrum, ax=ax, color=spectrum_color, linewidth=0.5)

    if len(peaks) == 0:
        return ax

    # Plot peaks
    mz = peaks.mz_array
    intensity = peaks.intensity_array

    ax.scatter(mz, intensity, color=peak_color, s=20, zorder=5, marker="v")

    # Annotate top peaks
    if annotate:
        sorted_indices = np.argsort(intensity)[::-1][:max_annotations]
        for idx in sorted_indices:
            ax.annotate(
                f"{mz[idx]:.2f}",
                xy=(mz[idx], intensity[idx]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=peak_color,
            )

    ax.set_title(f"Spectrum with {len(peaks)} detected peaks")
    return ax


def plot_heatmap(
    experiment: MSExperiment,
    ax: Optional[Any] = None,
    mz_range: Optional[Tuple[float, float]] = None,
    rt_range: Optional[Tuple[float, float]] = None,
    mz_bins: int = 500,
    rt_bins: int = 200,
    log_scale: bool = True,
    cmap: str = "viridis",
    ms_level: int = 1,
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Plot 2D LC-MS heatmap (RT vs m/z).

    Args:
        experiment: MSExperiment containing spectra
        ax: Matplotlib axes
        mz_range: (min, max) m/z range
        rt_range: (min, max) RT range
        mz_bins: Number of m/z bins
        rt_bins: Number of RT bins
        log_scale: Use log intensity scale
        cmap: Colormap name
        ms_level: MS level to plot (0 for all)
        title: Plot title
        **kwargs: Additional parameters

    Returns:
        Matplotlib axes

    Example:
        >>> plot_heatmap(experiment, mz_range=(400, 800), log_scale=True)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Get spectra
    if ms_level > 0:
        spectra = experiment.get_spectra_by_level(ms_level)
    else:
        spectra = experiment.spectra

    if not spectra:
        ax.set_xlabel("Retention Time (s)")
        ax.set_ylabel("m/z")
        ax.set_title("No spectra to display")
        return ax

    # Determine ranges
    if mz_range is None:
        mz_range = experiment.mz_range
    if rt_range is None:
        rt_range = experiment.rt_range

    # Create 2D histogram
    mz_edges = np.linspace(mz_range[0], mz_range[1], mz_bins + 1)
    rt_edges = np.linspace(rt_range[0], rt_range[1], rt_bins + 1)
    heatmap = np.zeros((rt_bins, mz_bins))

    for spec in spectra:
        if rt_range[0] <= spec.rt <= rt_range[1]:
            rt_idx = int((spec.rt - rt_range[0]) / (rt_range[1] - rt_range[0]) * rt_bins)
            rt_idx = min(rt_idx, rt_bins - 1)

            mask = (spec.mz >= mz_range[0]) & (spec.mz <= mz_range[1])
            mz_vals = spec.mz[mask]
            int_vals = spec.intensity[mask]

            for mz, intensity in zip(mz_vals, int_vals):
                mz_idx = int((mz - mz_range[0]) / (mz_range[1] - mz_range[0]) * mz_bins)
                mz_idx = min(max(mz_idx, 0), mz_bins - 1)
                heatmap[rt_idx, mz_idx] = max(heatmap[rt_idx, mz_idx], intensity)

    # Apply log scale
    if log_scale:
        heatmap = np.log1p(heatmap)

    # Plot
    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[mz_range[0], mz_range[1], rt_range[0], rt_range[1]],
        cmap=cmap,
        **kwargs,
    )

    ax.set_xlabel("m/z")
    ax.set_ylabel("Retention Time (s)")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"LC-MS Heatmap (MS{ms_level})")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log(Intensity + 1)" if log_scale else "Intensity")

    return ax


def plot_tic(
    experiment: MSExperiment,
    ax: Optional[Any] = None,
    ms_level: int = 1,
    **kwargs,
) -> Any:
    """
    Plot TIC from experiment.

    Args:
        experiment: MSExperiment
        ax: Matplotlib axes
        ms_level: MS level (0 for all)
        **kwargs: Additional parameters

    Returns:
        Matplotlib axes
    """
    tic = experiment.generate_tic(level=ms_level)
    return plot_chromatogram(
        tic,
        ax=ax,
        title=f"Total Ion Chromatogram (MS{ms_level})" if ms_level > 0 else "Total Ion Chromatogram",
        **kwargs,
    )


def plot_bpc(
    experiment: MSExperiment,
    ax: Optional[Any] = None,
    ms_level: int = 1,
    **kwargs,
) -> Any:
    """
    Plot BPC from experiment.

    Args:
        experiment: MSExperiment
        ax: Matplotlib axes
        ms_level: MS level (0 for all)
        **kwargs: Additional parameters

    Returns:
        Matplotlib axes
    """
    bpc = experiment.generate_bpc(level=ms_level)
    return plot_chromatogram(
        bpc,
        ax=ax,
        title=f"Base Peak Chromatogram (MS{ms_level})" if ms_level > 0 else "Base Peak Chromatogram",
        **kwargs,
    )


def plot_xic(
    experiment: MSExperiment,
    target_mz: float,
    tolerance: float = 0.01,
    ppm: bool = False,
    ax: Optional[Any] = None,
    ms_level: int = 1,
    **kwargs,
) -> Any:
    """
    Plot XIC from experiment.

    Args:
        experiment: MSExperiment
        target_mz: Target m/z
        tolerance: m/z tolerance (Da or ppm)
        ppm: If True, tolerance is in ppm
        ax: Matplotlib axes
        ms_level: MS level
        **kwargs: Additional parameters

    Returns:
        Matplotlib axes
    """
    xic = experiment.generate_xic(target_mz, tolerance, ppm=ppm, level=ms_level)
    tol_str = f"{tolerance} ppm" if ppm else f"{tolerance} Da"
    return plot_chromatogram(
        xic,
        ax=ax,
        title=f"XIC @ m/z {target_mz:.4f} ± {tol_str}",
        **kwargs,
    )


def plot_mirror(
    spectrum1: Spectrum,
    spectrum2: Spectrum,
    ax: Optional[Any] = None,
    labels: Tuple[str, str] = ("Query", "Reference"),
    mz_range: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Any:
    """
    Plot mirror plot of two spectra.

    Args:
        spectrum1: Top spectrum (positive y)
        spectrum2: Bottom spectrum (negative y)
        ax: Matplotlib axes
        labels: Labels for spectra
        mz_range: (min, max) m/z range
        **kwargs: Additional parameters

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    mz1, int1 = spectrum1.mz, spectrum1.intensity
    mz2, int2 = spectrum2.mz, spectrum2.intensity

    if mz_range:
        mask1 = (mz1 >= mz_range[0]) & (mz1 <= mz_range[1])
        mask2 = (mz2 >= mz_range[0]) & (mz2 <= mz_range[1])
        mz1, int1 = mz1[mask1], int1[mask1]
        mz2, int2 = mz2[mask2], int2[mask2]

    # Normalize
    if len(int1) > 0:
        int1 = int1 / np.max(int1) * 100
    if len(int2) > 0:
        int2 = int2 / np.max(int2) * 100

    # Plot top spectrum
    ax.stem(mz1, int1, linefmt="b-", markerfmt=" ", basefmt=" ", label=labels[0])

    # Plot bottom spectrum (inverted)
    ax.stem(mz2, -int2, linefmt="r-", markerfmt=" ", basefmt=" ", label=labels[1])

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity (%)")
    ax.legend()

    if mz_range:
        ax.set_xlim(mz_range)

    ax.set_title("Mirror Plot")

    return ax
