"""
High-level workflows for LC-MS data processing.
"""

from typing import Optional, List, Dict, Any, Callable
import numpy as np

from .spectrum import Spectrum
from .chromatogram import Chromatogram
from .peak import Peak, PeakList
from .feature import Feature, FeatureMap
from .experiment import MSExperiment
from .algorithms import (
    pick_peaks,
    centroid_spectrum,
    smooth_spectrum,
    correct_baseline,
    estimate_noise,
)


class PeakPickingWorkflow:
    """
    Complete peak picking workflow with preprocessing.

    Example:
        >>> workflow = PeakPickingWorkflow(
        ...     smooth=True,
        ...     baseline_correct=True,
        ...     min_snr=5.0,
        ... )
        >>> for spec in experiment.spectra:
        ...     peaks = workflow.process(spec)
    """

    def __init__(
        self,
        smooth: bool = True,
        smooth_method: str = "gaussian",
        smooth_window: int = 5,
        baseline_correct: bool = True,
        baseline_method: str = "snip",
        min_snr: float = 3.0,
        min_intensity: float = 0.0,
        fit_peaks: bool = True,
    ):
        """
        Initialize workflow.

        Args:
            smooth: Apply smoothing
            smooth_method: Smoothing method
            smooth_window: Smoothing window size
            baseline_correct: Apply baseline correction
            baseline_method: Baseline correction method
            min_snr: Minimum signal-to-noise ratio
            min_intensity: Minimum intensity threshold
            fit_peaks: Fit Gaussian to peaks
        """
        self.smooth = smooth
        self.smooth_method = smooth_method
        self.smooth_window = smooth_window
        self.baseline_correct = baseline_correct
        self.baseline_method = baseline_method
        self.min_snr = min_snr
        self.min_intensity = min_intensity
        self.fit_peaks = fit_peaks

    def process(self, spectrum: Spectrum) -> PeakList:
        """
        Process a spectrum and return peaks.

        Args:
            spectrum: Input spectrum

        Returns:
            Detected peaks
        """
        processed = spectrum

        if self.smooth:
            processed = smooth_spectrum(
                processed,
                method=self.smooth_method,
                window_size=self.smooth_window,
            )

        if self.baseline_correct:
            processed = correct_baseline(
                processed,
                method=self.baseline_method,
            )

        peaks = pick_peaks(
            processed,
            min_snr=self.min_snr,
            min_intensity=self.min_intensity,
            fit_peaks=self.fit_peaks,
        )

        return peaks


class FeatureDetectionWorkflow:
    """
    Workflow for detecting 2D features from LC-MS data.

    Combines peak picking with RT-based feature linking.

    Example:
        >>> workflow = FeatureDetectionWorkflow(
        ...     mz_tolerance=0.01,
        ...     rt_tolerance=30.0,
        ... )
        >>> features = workflow.process(experiment)
    """

    def __init__(
        self,
        mz_tolerance: float = 0.01,
        mz_tolerance_ppm: bool = False,
        rt_tolerance: float = 30.0,
        min_scans: int = 3,
        min_snr: float = 3.0,
        min_intensity: float = 0.0,
        smooth: bool = True,
        baseline_correct: bool = True,
    ):
        """
        Initialize workflow.

        Args:
            mz_tolerance: m/z tolerance for linking (Da or ppm)
            mz_tolerance_ppm: If True, mz_tolerance is in ppm
            rt_tolerance: RT tolerance for linking (seconds)
            min_scans: Minimum number of scans for a feature
            min_snr: Minimum SNR for peaks
            min_intensity: Minimum intensity for peaks
            smooth: Apply smoothing
            baseline_correct: Apply baseline correction
        """
        self.mz_tolerance = mz_tolerance
        self.mz_tolerance_ppm = mz_tolerance_ppm
        self.rt_tolerance = rt_tolerance
        self.min_scans = min_scans
        self.min_snr = min_snr
        self.min_intensity = min_intensity

        self.peak_workflow = PeakPickingWorkflow(
            smooth=smooth,
            baseline_correct=baseline_correct,
            min_snr=min_snr,
            min_intensity=min_intensity,
        )

    def process(
        self,
        experiment: MSExperiment,
        ms_level: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> FeatureMap:
        """
        Detect features from experiment.

        Args:
            experiment: Input MSExperiment
            ms_level: MS level to process
            progress_callback: Progress callback(current, total)

        Returns:
            Detected features
        """
        spectra = experiment.get_spectra_by_level(ms_level)
        if not spectra:
            return FeatureMap()

        # Sort by RT
        spectra = sorted(spectra, key=lambda s: s.rt)

        # Pick peaks from all spectra
        all_peaks: List[List[Peak]] = []
        for i, spec in enumerate(spectra):
            peaks = self.peak_workflow.process(spec)
            for peak in peaks:
                peak.rt = spec.rt
                peak.spectrum_index = spec.index
            all_peaks.append(list(peaks))

            if progress_callback:
                progress_callback(i + 1, len(spectra))

        # Link peaks into features
        features = self._link_peaks(all_peaks, spectra)

        fmap = FeatureMap()
        fmap.source_file = experiment.source_file

        for feature in features:
            fmap.add(feature)

        return fmap

    def _link_peaks(
        self,
        all_peaks: List[List[Peak]],
        spectra: List[Spectrum],
    ) -> List[Feature]:
        """Link peaks across spectra into features."""
        features: List[Feature] = []
        active_features: List[Dict[str, Any]] = []

        for scan_idx, peaks in enumerate(all_peaks):
            rt = spectra[scan_idx].rt

            # Match peaks to active features
            matched_features = set()
            matched_peaks = set()

            for peak_idx, peak in enumerate(peaks):
                best_feature = None
                best_distance = float("inf")

                for feat_idx, active in enumerate(active_features):
                    if feat_idx in matched_features:
                        continue

                    # Check RT tolerance
                    if rt - active["last_rt"] > self.rt_tolerance:
                        continue

                    # Check m/z tolerance
                    if self.mz_tolerance_ppm:
                        mz_tol = active["mz"] * self.mz_tolerance * 1e-6
                    else:
                        mz_tol = self.mz_tolerance

                    mz_dist = abs(peak.mz - active["mz"])
                    if mz_dist <= mz_tol:
                        if mz_dist < best_distance:
                            best_distance = mz_dist
                            best_feature = feat_idx

                if best_feature is not None:
                    # Add peak to feature
                    active_features[best_feature]["peaks"].append(peak)
                    active_features[best_feature]["last_rt"] = rt

                    # Update weighted average m/z
                    old_mz = active_features[best_feature]["mz"]
                    old_weight = active_features[best_feature]["intensity_sum"]
                    new_weight = old_weight + peak.intensity
                    active_features[best_feature]["mz"] = (
                        old_mz * old_weight + peak.mz * peak.intensity
                    ) / new_weight
                    active_features[best_feature]["intensity_sum"] = new_weight

                    if peak.intensity > active_features[best_feature]["max_intensity"]:
                        active_features[best_feature]["max_intensity"] = peak.intensity
                        active_features[best_feature]["apex_rt"] = rt

                    matched_features.add(best_feature)
                    matched_peaks.add(peak_idx)

            # Start new features for unmatched peaks
            for peak_idx, peak in enumerate(peaks):
                if peak_idx not in matched_peaks:
                    active_features.append({
                        "mz": peak.mz,
                        "peaks": [peak],
                        "last_rt": rt,
                        "start_rt": rt,
                        "apex_rt": rt,
                        "max_intensity": peak.intensity,
                        "intensity_sum": peak.intensity,
                    })

            # Finalize features that have timed out
            remaining = []
            for active in active_features:
                if rt - active["last_rt"] > self.rt_tolerance:
                    if len(active["peaks"]) >= self.min_scans:
                        feature = self._create_feature(active)
                        features.append(feature)
                else:
                    remaining.append(active)
            active_features = remaining

        # Finalize remaining features
        for active in active_features:
            if len(active["peaks"]) >= self.min_scans:
                feature = self._create_feature(active)
                features.append(feature)

        return features

    def _create_feature(self, active: Dict[str, Any]) -> Feature:
        """Create Feature from tracked data."""
        peaks = active["peaks"]

        feature = Feature()
        feature.mz = active["mz"]
        feature.rt = active["apex_rt"]
        feature.intensity = active["max_intensity"]

        # Calculate volume (sum of areas)
        feature.volume = sum(p.area for p in peaks)

        # Set boundaries
        feature.mz_min = min(p.mz_left for p in peaks) if peaks else active["mz"]
        feature.mz_max = max(p.mz_right for p in peaks) if peaks else active["mz"]
        feature.rt_min = active["start_rt"]
        feature.rt_max = active["last_rt"]

        # Add peaks
        for peak in peaks:
            feature.peaks.add(peak)

        # Quality score based on number of scans and intensity consistency
        n_scans = len(peaks)
        intensities = [p.intensity for p in peaks]
        intensity_cv = np.std(intensities) / (np.mean(intensities) + 1e-10)
        feature.quality = min(1.0, n_scans / 10.0) * max(0.0, 1.0 - intensity_cv)

        return feature


def process_experiment(
    experiment: MSExperiment,
    centroid: bool = True,
    smooth: bool = True,
    baseline_correct: bool = True,
    min_snr: float = 3.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MSExperiment:
    """
    Process all spectra in an experiment.

    Applies smoothing, baseline correction, and optional centroiding.

    Args:
        experiment: Input experiment
        centroid: Convert to centroided
        smooth: Apply smoothing
        baseline_correct: Apply baseline correction
        min_snr: Minimum SNR for centroiding
        progress_callback: Progress callback

    Returns:
        Processed experiment
    """
    result = MSExperiment()
    result.source_file = experiment.source_file
    result.metadata = experiment.metadata.copy()

    for i, spec in enumerate(experiment.spectra):
        processed = spec

        if smooth:
            processed = smooth_spectrum(processed)

        if baseline_correct:
            processed = correct_baseline(processed)

        if centroid:
            processed = centroid_spectrum(processed, min_snr=min_snr)

        result.add_spectrum(processed)

        if progress_callback:
            progress_callback(i + 1, len(experiment.spectra))

    for chrom in experiment.chromatograms:
        result.add_chromatogram(chrom.copy())

    return result
