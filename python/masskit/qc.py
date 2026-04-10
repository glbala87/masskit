"""
Quality control metrics and reporting for LC-MS data.

Provides QC metrics for individual runs and cross-run comparisons
including TIC stability, mass accuracy, peak capacity, and reproducibility.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import numpy as np

from .spectrum import Spectrum
from .chromatogram import Chromatogram
from .experiment import MSExperiment
from .peak import PeakList
from .algorithms import pick_peaks


@dataclass
class QCMetrics:
    """
    Quality control metrics for a single LC-MS run.

    Attributes:
        filename: Source file name
        num_spectra: Total number of spectra
        num_ms1: Number of MS1 spectra
        num_ms2: Number of MS2 spectra
        rt_range: (min, max) retention time range
        mz_range: (min, max) m/z range
        tic_median: Median TIC across MS1 spectra
        tic_cv: Coefficient of variation of TIC (%)
        tic_stability: TIC stability score (0-1, higher is better)
        base_peak_mz_stability: CV of base peak m/z (%)
        ms1_peak_count_median: Median number of peaks per MS1 spectrum
        ms2_trigger_rate: Fraction of MS1 scans that triggered MS2
        peak_capacity: Estimated peak capacity
        injection_time_median: Median ion injection time (if available)
        mass_accuracy_ppm: Mass accuracy in ppm (if reference masses available)
    """
    filename: str = ""
    num_spectra: int = 0
    num_ms1: int = 0
    num_ms2: int = 0
    rt_range: Tuple[float, float] = (0.0, 0.0)
    mz_range: Tuple[float, float] = (0.0, 0.0)
    tic_median: float = 0.0
    tic_cv: float = 0.0
    tic_stability: float = 0.0
    base_peak_mz_stability: float = 0.0
    ms1_peak_count_median: float = 0.0
    ms2_trigger_rate: float = 0.0
    peak_capacity: float = 0.0
    mass_accuracy_ppm: float = 0.0
    scan_rate: float = 0.0

    def overall_score(self) -> float:
        """Calculate overall QC score (0-1)."""
        scores = []
        # TIC stability (CV < 20% is good)
        tic_score = max(0, 1.0 - self.tic_cv / 50.0)
        scores.append(tic_score)

        # Peak count stability
        if self.ms1_peak_count_median > 0:
            scores.append(min(1.0, self.ms1_peak_count_median / 1000.0))

        # MS2 trigger rate
        if self.num_ms1 > 0:
            scores.append(min(1.0, self.ms2_trigger_rate))

        return float(np.mean(scores)) if scores else 0.0

    def status(self) -> str:
        """Return PASS/WARN/FAIL status."""
        score = self.overall_score()
        if score >= 0.7:
            return "PASS"
        elif score >= 0.4:
            return "WARN"
        return "FAIL"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "num_spectra": self.num_spectra,
            "num_ms1": self.num_ms1,
            "num_ms2": self.num_ms2,
            "rt_range_min": self.rt_range[0],
            "rt_range_max": self.rt_range[1],
            "mz_range_min": self.mz_range[0],
            "mz_range_max": self.mz_range[1],
            "tic_median": self.tic_median,
            "tic_cv_percent": self.tic_cv,
            "tic_stability": self.tic_stability,
            "ms1_peak_count_median": self.ms1_peak_count_median,
            "ms2_trigger_rate": self.ms2_trigger_rate,
            "peak_capacity": self.peak_capacity,
            "scan_rate": self.scan_rate,
            "overall_score": self.overall_score(),
            "status": self.status(),
        }

    def __repr__(self) -> str:
        return (
            f"QCMetrics(file='{self.filename}', "
            f"spectra={self.num_spectra}, "
            f"TIC_CV={self.tic_cv:.1f}%, "
            f"status={self.status()})"
        )


def compute_qc_metrics(
    experiment: MSExperiment,
    filename: str = "",
) -> QCMetrics:
    """
    Compute QC metrics for an LC-MS experiment.

    Args:
        experiment: MSExperiment to analyze
        filename: Source filename for reporting

    Returns:
        QCMetrics object

    Example:
        >>> exp = load_mzml("sample.mzML")
        >>> qc = compute_qc_metrics(exp, "sample.mzML")
        >>> print(f"TIC CV: {qc.tic_cv:.1f}%, Status: {qc.status()}")
    """
    metrics = QCMetrics(filename=filename)
    metrics.num_spectra = experiment.num_spectra

    if experiment.num_spectra == 0:
        return metrics

    ms1_spectra = [s for s in experiment.spectra if s.ms_level == 1]
    ms2_spectra = [s for s in experiment.spectra if s.ms_level == 2]

    metrics.num_ms1 = len(ms1_spectra)
    metrics.num_ms2 = len(ms2_spectra)

    # RT range
    all_rts = [s.rt for s in experiment.spectra]
    metrics.rt_range = (min(all_rts), max(all_rts))

    # m/z range from MS1
    if ms1_spectra:
        mz_mins = [np.min(s.mz) for s in ms1_spectra if len(s.mz) > 0]
        mz_maxs = [np.max(s.mz) for s in ms1_spectra if len(s.mz) > 0]
        if mz_mins and mz_maxs:
            metrics.mz_range = (min(mz_mins), max(mz_maxs))

    # TIC metrics
    if ms1_spectra:
        tic_values = np.array([s.tic for s in ms1_spectra])
        metrics.tic_median = float(np.median(tic_values))

        tic_mean = np.mean(tic_values)
        if tic_mean > 0:
            metrics.tic_cv = float(np.std(tic_values) / tic_mean * 100)

        # TIC stability: fraction of scans within 2x of median
        if metrics.tic_median > 0:
            within_range = np.sum(
                (tic_values > metrics.tic_median * 0.5) &
                (tic_values < metrics.tic_median * 2.0)
            )
            metrics.tic_stability = float(within_range / len(tic_values))

    # Base peak m/z stability
    if ms1_spectra:
        bp_mzs = []
        for spec in ms1_spectra:
            if len(spec.intensity) > 0:
                bp_idx = np.argmax(spec.intensity)
                bp_mzs.append(spec.mz[bp_idx])
        if bp_mzs:
            bp_mzs = np.array(bp_mzs)
            bp_mean = np.mean(bp_mzs)
            if bp_mean > 0:
                metrics.base_peak_mz_stability = float(np.std(bp_mzs) / bp_mean * 100)

    # Peak count per MS1
    if ms1_spectra:
        peak_counts = []
        for spec in ms1_spectra[:50]:  # Sample first 50 for speed
            peaks = pick_peaks(spec, min_snr=3.0)
            peak_counts.append(len(peaks))
        if peak_counts:
            metrics.ms1_peak_count_median = float(np.median(peak_counts))

    # MS2 trigger rate
    if ms1_spectra:
        metrics.ms2_trigger_rate = len(ms2_spectra) / len(ms1_spectra)

    # Peak capacity estimate (based on average peak width)
    if ms1_spectra and metrics.rt_range[1] > metrics.rt_range[0]:
        rt_duration = metrics.rt_range[1] - metrics.rt_range[0]
        # Estimate average chromatographic peak width
        # Using a simple heuristic: sample some peaks and measure widths
        avg_peak_width = _estimate_chromatographic_peak_width(experiment)
        if avg_peak_width > 0:
            metrics.peak_capacity = rt_duration / avg_peak_width

    # Scan rate (scans per second)
    if len(all_rts) > 1 and metrics.rt_range[1] > metrics.rt_range[0]:
        duration = metrics.rt_range[1] - metrics.rt_range[0]
        metrics.scan_rate = len(all_rts) / duration if duration > 0 else 0

    return metrics


def _estimate_chromatographic_peak_width(
    experiment: MSExperiment,
    n_samples: int = 5,
) -> float:
    """Estimate average chromatographic peak width from a few intense features."""
    ms1_spectra = [s for s in experiment.spectra if s.ms_level == 1]
    if len(ms1_spectra) < 10:
        return 0.0

    # Find the most intense m/z values
    total_tic = {}
    for spec in ms1_spectra[:100]:
        if len(spec.mz) == 0:
            continue
        bp_idx = np.argmax(spec.intensity)
        mz_key = round(spec.mz[bp_idx], 2)
        total_tic[mz_key] = total_tic.get(mz_key, 0) + spec.intensity[bp_idx]

    if not total_tic:
        return 0.0

    # Get top m/z values
    top_mzs = sorted(total_tic.keys(), key=lambda k: total_tic[k], reverse=True)[:n_samples]

    widths = []
    for target_mz in top_mzs:
        # Build XIC
        rts = []
        intensities = []
        for spec in ms1_spectra:
            rts.append(spec.rt)
            mask = np.abs(spec.mz - target_mz) <= 0.01
            intensities.append(np.sum(spec.intensity[mask]) if np.any(mask) else 0.0)

        intensities = np.array(intensities)
        if len(intensities) == 0 or np.max(intensities) == 0:
            continue

        # Find FWHM
        half_max = np.max(intensities) / 2
        above = intensities >= half_max
        if np.any(above):
            first_idx = np.argmax(above)
            last_idx = len(above) - 1 - np.argmax(above[::-1])
            if last_idx > first_idx:
                rts_arr = np.array(rts)
                width = rts_arr[last_idx] - rts_arr[first_idx]
                if width > 0:
                    widths.append(width)

    return float(np.median(widths)) if widths else 0.0


def compare_qc_metrics(
    metrics_list: List[QCMetrics],
) -> Dict[str, Dict]:
    """
    Compare QC metrics across multiple runs.

    Args:
        metrics_list: List of QCMetrics from multiple runs

    Returns:
        Dict with comparison statistics

    Example:
        >>> metrics = [compute_qc_metrics(exp, name) for exp, name in runs]
        >>> comparison = compare_qc_metrics(metrics)
    """
    if not metrics_list:
        return {}

    comparison = {
        "n_runs": len(metrics_list),
        "filenames": [m.filename for m in metrics_list],
        "statuses": [m.status() for m in metrics_list],
        "overall_scores": [m.overall_score() for m in metrics_list],
    }

    # Numeric metrics to compare
    numeric_fields = [
        "tic_median", "tic_cv", "ms1_peak_count_median",
        "ms2_trigger_rate", "peak_capacity", "scan_rate",
    ]

    for field_name in numeric_fields:
        values = [getattr(m, field_name) for m in metrics_list]
        comparison[field_name] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "cv_percent": float(np.std(values) / np.mean(values) * 100) if np.mean(values) > 0 else 0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return comparison


def generate_qc_report(
    metrics_list: List[QCMetrics],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a text QC report.

    Args:
        metrics_list: List of QCMetrics
        output_path: Optional file path to write report

    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 80)
    lines.append("LC-MS Quality Control Report")
    lines.append("=" * 80)
    lines.append("")

    for m in metrics_list:
        lines.append(f"File: {m.filename}")
        lines.append(f"  Status: {m.status()} (score: {m.overall_score():.2f})")
        lines.append(f"  Spectra: {m.num_spectra} (MS1: {m.num_ms1}, MS2: {m.num_ms2})")
        lines.append(f"  RT range: {m.rt_range[0]:.1f} - {m.rt_range[1]:.1f} s")
        lines.append(f"  m/z range: {m.mz_range[0]:.2f} - {m.mz_range[1]:.2f}")
        lines.append(f"  TIC median: {m.tic_median:.0f}")
        lines.append(f"  TIC CV: {m.tic_cv:.1f}%")
        lines.append(f"  TIC stability: {m.tic_stability:.2f}")
        lines.append(f"  MS1 peak count (median): {m.ms1_peak_count_median:.0f}")
        lines.append(f"  MS2 trigger rate: {m.ms2_trigger_rate:.2f}")
        lines.append(f"  Peak capacity: {m.peak_capacity:.0f}")
        lines.append(f"  Scan rate: {m.scan_rate:.1f} scans/s")
        lines.append("")

    if len(metrics_list) > 1:
        lines.append("-" * 80)
        lines.append("Cross-run comparison:")
        comparison = compare_qc_metrics(metrics_list)
        for field_name in ["tic_median", "tic_cv", "ms1_peak_count_median",
                           "peak_capacity"]:
            if field_name in comparison:
                stats = comparison[field_name]
                lines.append(
                    f"  {field_name}: mean={stats['mean']:.1f}, "
                    f"CV={stats['cv_percent']:.1f}%, "
                    f"range=[{stats['min']:.1f}, {stats['max']:.1f}]"
                )

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report
