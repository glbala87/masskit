"""
Signal processing and analysis algorithms for LC-MS data.
"""

from typing import Optional, List, Tuple, Union
import numpy as np
from scipy import ndimage, signal
from scipy.optimize import curve_fit

from .spectrum import Spectrum, SpectrumType
from .chromatogram import Chromatogram
from .peak import Peak, PeakList


def estimate_noise(
    spectrum: Spectrum,
    method: str = "mad",
    percentile: float = 5.0,
) -> float:
    """
    Estimate noise level in a spectrum.

    Args:
        spectrum: Input spectrum
        method: Noise estimation method:
            - "mad": Median absolute deviation (robust)
            - "percentile": Use intensity percentile
            - "std": Standard deviation of low intensities
        percentile: Percentile for "percentile" method

    Returns:
        Estimated noise level

    Example:
        >>> noise = estimate_noise(spectrum)
        >>> peaks = pick_peaks(spectrum, min_snr=3, noise=noise)
    """
    intensity = spectrum.intensity
    if len(intensity) == 0:
        return 0.0

    if method == "mad":
        # Median absolute deviation
        median = np.median(intensity)
        mad = np.median(np.abs(intensity - median))
        # Scale factor for normal distribution
        return mad * 1.4826
    elif method == "percentile":
        # Use low intensity percentile
        threshold = np.percentile(intensity, percentile)
        low_intensity = intensity[intensity <= threshold]
        if len(low_intensity) > 0:
            return np.std(low_intensity)
        return np.std(intensity) * 0.1
    elif method == "std":
        # Standard deviation of bottom quartile
        threshold = np.percentile(intensity, 25)
        low_intensity = intensity[intensity <= threshold]
        if len(low_intensity) > 0:
            return np.std(low_intensity)
        return np.std(intensity) * 0.1
    else:
        raise ValueError(f"Unknown noise estimation method: {method}")


def pick_peaks(
    spectrum: Spectrum,
    min_snr: float = 3.0,
    min_intensity: float = 0.0,
    window_size: int = 3,
    noise: Optional[float] = None,
    fit_peaks: bool = True,
) -> PeakList:
    """
    Pick peaks from a spectrum.

    Detects local maxima and calculates peak properties including
    area, FWHM, and signal-to-noise ratio.

    Args:
        spectrum: Input spectrum
        min_snr: Minimum signal-to-noise ratio
        min_intensity: Minimum absolute intensity
        window_size: Local maximum detection window
        noise: Pre-computed noise level (None = auto-estimate)
        fit_peaks: Fit Gaussian to refine peak parameters

    Returns:
        PeakList with detected peaks

    Example:
        >>> peaks = pick_peaks(spectrum, min_snr=5)
        >>> print(f"Found {len(peaks)} peaks")
        >>> peaks.sort_by_intensity()
        >>> top_peak = peaks[0]
    """
    if len(spectrum) < 3:
        return PeakList()

    mz = spectrum.mz
    intensity = spectrum.intensity

    # Estimate noise if not provided
    if noise is None:
        noise = estimate_noise(spectrum)
    # Ensure noise has a reasonable minimum to prevent overflow in SNR calculation
    min_noise = np.finfo(np.float64).tiny * 1e10  # ~2.2e-298
    if noise <= min_noise:
        noise = np.min(intensity[intensity > 0]) if np.any(intensity > 0) else 1.0
    noise = max(noise, min_noise)

    # Find local maxima
    half_window = window_size // 2
    peaks_list = []

    for i in range(half_window, len(intensity) - half_window):
        # Check if local maximum
        window = intensity[i - half_window : i + half_window + 1]
        if intensity[i] != np.max(window):
            continue
        if intensity[i] < min_intensity:
            continue

        snr = intensity[i] / noise
        if snr < min_snr:
            continue

        # Find peak boundaries (descend to valley)
        left = i
        while left > 0 and intensity[left - 1] < intensity[left]:
            left -= 1

        right = i
        while right < len(intensity) - 1 and intensity[right + 1] < intensity[right]:
            right += 1

        # Calculate peak properties
        peak = Peak()
        peak.mz = mz[i]
        peak.rt = spectrum.rt
        peak.intensity = intensity[i]
        peak.snr = snr
        peak.mz_left = mz[left]
        peak.mz_right = mz[right]
        peak.spectrum_index = spectrum.index

        # Calculate area (trapezoidal)
        if right > left:
            # np.trapezoid for numpy>=2.0, np.trapz for older
            _trap = getattr(np, "trapezoid", np.trapz)
            peak.area = _trap(intensity[left : right + 1], mz[left : right + 1])

        # Calculate FWHM
        half_max = intensity[i] / 2
        fwhm_left = mz[i]
        for j in range(i, left - 1, -1):
            if intensity[j] <= half_max:
                # Interpolate
                if j < i and intensity[j + 1] > half_max:
                    t = (half_max - intensity[j]) / (intensity[j + 1] - intensity[j])
                    fwhm_left = mz[j] + t * (mz[j + 1] - mz[j])
                else:
                    fwhm_left = mz[j]
                break

        fwhm_right = mz[i]
        for j in range(i, right + 1):
            if intensity[j] <= half_max:
                if j > i and intensity[j - 1] > half_max:
                    t = (half_max - intensity[j - 1]) / (intensity[j] - intensity[j - 1])
                    fwhm_right = mz[j - 1] + t * (mz[j] - mz[j - 1])
                else:
                    fwhm_right = mz[j]
                break

        peak.fwhm_mz = fwhm_right - fwhm_left

        # Fit Gaussian for better centroid
        if fit_peaks and right - left >= 3:
            try:
                popt = _fit_gaussian(
                    mz[left : right + 1],
                    intensity[left : right + 1],
                    mz[i],
                    intensity[i],
                )
                if popt is not None:
                    peak.mz = popt[0]  # Center
                    peak.fwhm_mz = 2.355 * popt[1]  # FWHM from sigma
            except Exception:
                pass

        peaks_list.append(peak)

    return PeakList(peaks_list)


def _fit_gaussian(
    x: np.ndarray, y: np.ndarray, center: float, amplitude: float
) -> Optional[Tuple[float, float, float]]:
    """Fit Gaussian to peak data."""
    def gaussian(x, center, sigma, amplitude):
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    try:
        sigma_init = (x[-1] - x[0]) / 4
        popt, _ = curve_fit(
            gaussian,
            x,
            y,
            p0=[center, sigma_init, amplitude],
            bounds=(
                [x[0], 0, 0],
                [x[-1], x[-1] - x[0], amplitude * 2],
            ),
            maxfev=100,
        )
        return tuple(popt)
    except Exception:
        return None


def centroid_spectrum(
    spectrum: Spectrum,
    min_snr: float = 3.0,
    min_intensity: float = 0.0,
) -> Spectrum:
    """
    Convert profile spectrum to centroided.

    Args:
        spectrum: Input profile spectrum
        min_snr: Minimum signal-to-noise ratio
        min_intensity: Minimum absolute intensity

    Returns:
        Centroided spectrum

    Example:
        >>> centroided = centroid_spectrum(profile_spectrum)
    """
    peaks = pick_peaks(spectrum, min_snr=min_snr, min_intensity=min_intensity)

    if len(peaks) == 0:
        result = Spectrum(ms_level=spectrum.ms_level, rt=spectrum.rt)
    else:
        result = Spectrum(
            mz=peaks.mz_array,
            intensity=peaks.intensity_array,
            ms_level=spectrum.ms_level,
            rt=spectrum.rt,
        )

    result.spectrum_type = SpectrumType.CENTROID
    result.polarity = spectrum.polarity
    result.precursors = spectrum.precursors.copy()
    result.native_id = spectrum.native_id
    result.metadata = spectrum.metadata.copy()

    return result


def smooth_spectrum(
    spectrum: Spectrum,
    method: str = "gaussian",
    window_size: int = 5,
    **kwargs,
) -> Spectrum:
    """
    Smooth a spectrum.

    Args:
        spectrum: Input spectrum
        method: Smoothing method:
            - "gaussian": Gaussian kernel smoothing
            - "savgol": Savitzky-Golay filter
            - "moving_average": Simple moving average
        window_size: Size of smoothing window
        **kwargs: Method-specific parameters:
            - gaussian: sigma (default: 1.5)
            - savgol: polyorder (default: 2)

    Returns:
        Smoothed spectrum

    Example:
        >>> smoothed = smooth_spectrum(spectrum, method="savgol", window_size=7)
    """
    if len(spectrum) < window_size:
        return spectrum.copy()

    intensity = spectrum.intensity.copy()

    if method == "gaussian":
        sigma = kwargs.get("sigma", 1.5)
        intensity = ndimage.gaussian_filter1d(intensity, sigma)
    elif method == "savgol":
        polyorder = kwargs.get("polyorder", 2)
        if window_size % 2 == 0:
            window_size += 1
        if polyorder >= window_size:
            polyorder = window_size - 1
        intensity = signal.savgol_filter(intensity, window_size, polyorder)
    elif method == "moving_average":
        kernel = np.ones(window_size) / window_size
        intensity = np.convolve(intensity, kernel, mode="same")
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    # Ensure non-negative
    intensity = np.maximum(intensity, 0)

    result = Spectrum(
        mz=spectrum.mz.copy(),
        intensity=intensity,
        ms_level=spectrum.ms_level,
        rt=spectrum.rt,
        spectrum_type=spectrum.spectrum_type,
        polarity=spectrum.polarity,
    )
    result.precursors = spectrum.precursors.copy()
    result.native_id = spectrum.native_id
    result.metadata = spectrum.metadata.copy()

    return result


def correct_baseline(
    spectrum: Spectrum,
    method: str = "snip",
    **kwargs,
) -> Spectrum:
    """
    Correct baseline drift in a spectrum.

    Args:
        spectrum: Input spectrum
        method: Baseline estimation method:
            - "snip": Statistics-sensitive Non-linear Iterative Peak-clipping
            - "tophat": Morphological top-hat filter
            - "rolling_ball": Rolling ball algorithm
        **kwargs: Method-specific parameters:
            - snip: iterations (default: 40)
            - tophat: half_window (default: 50)
            - rolling_ball: radius (default: 100)

    Returns:
        Baseline-corrected spectrum

    Example:
        >>> corrected = correct_baseline(spectrum, method="snip", iterations=50)
    """
    if len(spectrum) < 3:
        return spectrum.copy()

    intensity = spectrum.intensity.copy()

    if method == "snip":
        iterations = kwargs.get("iterations", 40)
        baseline = _snip_baseline(intensity, iterations)
    elif method == "tophat":
        half_window = kwargs.get("half_window", 50)
        baseline = _tophat_baseline(intensity, half_window)
    elif method == "rolling_ball":
        radius = kwargs.get("radius", 100)
        baseline = _rolling_ball_baseline(intensity, radius)
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    # Subtract baseline
    corrected = intensity - baseline
    corrected = np.maximum(corrected, 0)

    result = Spectrum(
        mz=spectrum.mz.copy(),
        intensity=corrected,
        ms_level=spectrum.ms_level,
        rt=spectrum.rt,
        spectrum_type=spectrum.spectrum_type,
        polarity=spectrum.polarity,
    )
    result.precursors = spectrum.precursors.copy()
    result.native_id = spectrum.native_id
    result.metadata = spectrum.metadata.copy()

    return result


def _snip_baseline(intensity: np.ndarray, iterations: int) -> np.ndarray:
    """SNIP baseline estimation."""
    # Transform to square root space for better peak handling
    y = np.sqrt(np.sqrt(np.maximum(intensity, 0)))

    # Apply SNIP iterations
    for p in range(1, iterations + 1):
        for i in range(p, len(y) - p):
            y[i] = min(y[i], (y[i - p] + y[i + p]) / 2)

    # Transform back
    baseline = y ** 4
    return baseline


def _tophat_baseline(intensity: np.ndarray, half_window: int) -> np.ndarray:
    """Top-hat morphological baseline estimation."""
    # Opening operation = erosion followed by dilation
    struct = np.ones(2 * half_window + 1)
    eroded = ndimage.grey_erosion(intensity, footprint=struct)
    baseline = ndimage.grey_dilation(eroded, footprint=struct)
    return baseline


def _rolling_ball_baseline(intensity: np.ndarray, radius: int) -> np.ndarray:
    """Rolling ball baseline estimation."""
    n = len(intensity)
    baseline = np.zeros(n)

    for i in range(n):
        left = max(0, i - radius)
        right = min(n, i + radius + 1)

        # Find the lowest point in the window
        window = intensity[left:right]
        positions = np.arange(left, right)

        # Distance from center
        distances = np.abs(positions - i)

        # Ball height adjustment
        ball_heights = np.sqrt(np.maximum(0, radius**2 - distances**2))

        # Baseline is minimum of (intensity - ball_height)
        adjusted = window - ball_heights[:len(window)]
        baseline[i] = np.min(adjusted) + np.sqrt(max(0, radius**2 - 0))

    return baseline


def smooth_chromatogram(
    chromatogram: Chromatogram,
    method: str = "gaussian",
    window_size: int = 5,
    **kwargs,
) -> Chromatogram:
    """
    Smooth a chromatogram.

    Args:
        chromatogram: Input chromatogram
        method: Smoothing method ("gaussian", "savgol", "moving_average")
        window_size: Size of smoothing window
        **kwargs: Method-specific parameters

    Returns:
        Smoothed chromatogram
    """
    if len(chromatogram) < window_size:
        return chromatogram.copy()

    intensity = chromatogram.intensity.copy()

    if method == "gaussian":
        sigma = kwargs.get("sigma", 1.5)
        intensity = ndimage.gaussian_filter1d(intensity, sigma)
    elif method == "savgol":
        polyorder = kwargs.get("polyorder", 2)
        if window_size % 2 == 0:
            window_size += 1
        if polyorder >= window_size:
            polyorder = window_size - 1
        intensity = signal.savgol_filter(intensity, window_size, polyorder)
    elif method == "moving_average":
        kernel = np.ones(window_size) / window_size
        intensity = np.convolve(intensity, kernel, mode="same")
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    intensity = np.maximum(intensity, 0)

    result = Chromatogram(
        rt=chromatogram.rt.copy(),
        intensity=intensity,
        chrom_type=chromatogram.chrom_type,
        target_mz=chromatogram.target_mz,
        mz_tolerance=chromatogram.mz_tolerance,
    )
    result.native_id = chromatogram.native_id
    result.metadata = chromatogram.metadata.copy()

    return result


def correct_chromatogram_baseline(
    chromatogram: Chromatogram,
    method: str = "snip",
    **kwargs,
) -> Chromatogram:
    """
    Correct baseline drift in a chromatogram.

    Args:
        chromatogram: Input chromatogram
        method: Baseline estimation method
        **kwargs: Method-specific parameters

    Returns:
        Baseline-corrected chromatogram
    """
    if len(chromatogram) < 3:
        return chromatogram.copy()

    intensity = chromatogram.intensity.copy()

    if method == "snip":
        iterations = kwargs.get("iterations", 40)
        baseline = _snip_baseline(intensity, iterations)
    elif method == "tophat":
        half_window = kwargs.get("half_window", 50)
        baseline = _tophat_baseline(intensity, half_window)
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    corrected = intensity - baseline
    corrected = np.maximum(corrected, 0)

    result = Chromatogram(
        rt=chromatogram.rt.copy(),
        intensity=corrected,
        chrom_type=chromatogram.chrom_type,
        target_mz=chromatogram.target_mz,
        mz_tolerance=chromatogram.mz_tolerance,
    )
    result.native_id = chromatogram.native_id
    result.metadata = chromatogram.metadata.copy()

    return result
