"""
Isotope labeling quantification for LC-MS data.

Supports SILAC, TMT/iTRAQ, and dimethyl labeling strategies
for relative and absolute protein/peptide quantification.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .spectrum import Spectrum
from .peak import Peak


class LabelingStrategy(Enum):
    """Supported labeling strategies."""
    SILAC = "silac"
    TMT6 = "tmt6"
    TMT10 = "tmt10"
    TMT11 = "tmt11"
    TMT16 = "tmt16"
    TMT18 = "tmt18"
    ITRAQ4 = "itraq4"
    ITRAQ8 = "itraq8"
    DIMETHYL = "dimethyl"
    LABEL_FREE = "label_free"


# Reporter ion m/z values for TMT and iTRAQ
TMT6_REPORTERS = {
    "126": 126.127726,
    "127": 127.131081,
    "128": 128.134436,
    "129": 129.137790,
    "130": 130.141145,
    "131": 131.138180,
}

TMT10_REPORTERS = {
    "126": 126.127726, "127N": 127.124761, "127C": 127.131081,
    "128N": 128.128116, "128C": 128.134436, "129N": 129.131471,
    "129C": 129.137790, "130N": 130.134825, "130C": 130.141145,
    "131": 131.138180,
}

TMT11_REPORTERS = {
    **TMT10_REPORTERS,
    "131C": 131.144499,
}

TMT16_REPORTERS = {
    **TMT11_REPORTERS,
    "132N": 132.141535, "132C": 132.147854,
    "133N": 133.144890, "133C": 133.151209,
    "134N": 134.148245,
}

TMT18_REPORTERS = {
    **TMT16_REPORTERS,
    "134C": 134.154564,
    "135N": 135.151600,
}

ITRAQ4_REPORTERS = {
    "114": 114.11068,
    "115": 115.10826,
    "116": 116.11162,
    "117": 117.11497,
}

ITRAQ8_REPORTERS = {
    "113": 113.10787, "114": 114.11123, "115": 115.10826,
    "116": 116.11162, "117": 117.11497, "118": 118.11201,
    "119": 119.11537, "121": 121.12201,
}

# SILAC mass shifts (Da)
SILAC_SHIFTS = {
    "light": 0.0,
    "medium": {  # Arg6, Lys4
        "R": 6.020129,
        "K": 4.025107,
    },
    "heavy": {  # Arg10, Lys8
        "R": 10.008269,
        "K": 8.014199,
    },
}

# Dimethyl labeling mass shifts
DIMETHYL_SHIFTS = {
    "light": 28.031300,    # CH2O + NaBH3CN (light)
    "medium": 32.056407,   # CD2O + NaBH3CN (medium)
    "heavy": 36.075670,    # 13CD2O + NaBD3CN (heavy)
}


@dataclass
class ReporterIonQuantification:
    """Quantification result from reporter ion intensities."""
    spectrum_index: int = 0
    precursor_mz: float = 0.0
    channel_intensities: Dict[str, float] = field(default_factory=dict)
    total_intensity: float = 0.0
    signal_to_noise: float = 0.0
    isolation_purity: float = 0.0

    @property
    def ratios(self) -> Dict[str, float]:
        """Compute ratios relative to the first channel."""
        channels = sorted(self.channel_intensities.keys())
        if not channels or self.channel_intensities[channels[0]] == 0:
            return {}
        ref = self.channel_intensities[channels[0]]
        return {ch: self.channel_intensities[ch] / ref for ch in channels}


@dataclass
class SILACPair:
    """A matched SILAC peptide pair/triplet."""
    light_mz: float = 0.0
    heavy_mz: float = 0.0
    medium_mz: Optional[float] = None
    charge: int = 0
    rt: float = 0.0
    light_intensity: float = 0.0
    heavy_intensity: float = 0.0
    medium_intensity: Optional[float] = None
    sequence: str = ""

    @property
    def heavy_light_ratio(self) -> float:
        if self.light_intensity > 0:
            return self.heavy_intensity / self.light_intensity
        return float("inf")

    @property
    def log2_ratio(self) -> float:
        ratio = self.heavy_light_ratio
        if ratio > 0 and ratio != float("inf"):
            return np.log2(ratio)
        return 0.0


@dataclass
class LabelingResult:
    """Complete labeling quantification result."""
    strategy: LabelingStrategy = LabelingStrategy.LABEL_FREE
    n_quantified: int = 0
    reporter_quants: List[ReporterIonQuantification] = field(default_factory=list)
    silac_pairs: List[SILACPair] = field(default_factory=list)
    protein_ratios: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)


def get_reporter_ions(strategy: LabelingStrategy) -> Dict[str, float]:
    """Get reporter ion m/z values for a labeling strategy."""
    mapping = {
        LabelingStrategy.TMT6: TMT6_REPORTERS,
        LabelingStrategy.TMT10: TMT10_REPORTERS,
        LabelingStrategy.TMT11: TMT11_REPORTERS,
        LabelingStrategy.TMT16: TMT16_REPORTERS,
        LabelingStrategy.TMT18: TMT18_REPORTERS,
        LabelingStrategy.ITRAQ4: ITRAQ4_REPORTERS,
        LabelingStrategy.ITRAQ8: ITRAQ8_REPORTERS,
    }
    return mapping.get(strategy, {})


def extract_reporter_ions(
    spectrum: Spectrum,
    strategy: LabelingStrategy,
    tolerance_da: float = 0.003,
    min_intensity: float = 0.0,
) -> ReporterIonQuantification:
    """
    Extract reporter ion intensities from an MS2 spectrum.

    Args:
        spectrum: MS2 spectrum
        strategy: TMT or iTRAQ labeling strategy
        tolerance_da: Mass tolerance in Da for reporter matching
        min_intensity: Minimum intensity threshold

    Returns:
        ReporterIonQuantification with channel intensities
    """
    reporters = get_reporter_ions(strategy)
    if not reporters:
        raise ValueError(f"No reporter ions defined for {strategy}")

    mz_array = spectrum.mz
    int_array = spectrum.intensity
    channel_intensities = {}

    for channel, target_mz in reporters.items():
        mask = np.abs(mz_array - target_mz) <= tolerance_da
        if np.any(mask):
            intensity = float(np.max(int_array[mask]))
            if intensity >= min_intensity:
                channel_intensities[channel] = intensity
            else:
                channel_intensities[channel] = 0.0
        else:
            channel_intensities[channel] = 0.0

    total = sum(channel_intensities.values())

    return ReporterIonQuantification(
        precursor_mz=spectrum.precursor_mz if hasattr(spectrum, "precursor_mz") else 0.0,
        channel_intensities=channel_intensities,
        total_intensity=total,
    )


def batch_extract_reporters(
    spectra: List[Spectrum],
    strategy: LabelingStrategy,
    tolerance_da: float = 0.003,
    min_intensity: float = 0.0,
    min_total_intensity: float = 100.0,
) -> List[ReporterIonQuantification]:
    """
    Extract reporter ions from multiple MS2 spectra.

    Args:
        spectra: List of MS2 spectra
        strategy: TMT or iTRAQ labeling strategy
        tolerance_da: Mass tolerance in Da
        min_intensity: Min intensity per channel
        min_total_intensity: Min total reporter intensity

    Returns:
        List of quantification results (filtered)
    """
    results = []
    for i, spec in enumerate(spectra):
        quant = extract_reporter_ions(spec, strategy, tolerance_da, min_intensity)
        quant.spectrum_index = i
        if quant.total_intensity >= min_total_intensity:
            results.append(quant)
    return results


def find_silac_pairs(
    peaks: List[Peak],
    charge_states: List[int],
    triple: bool = False,
    mz_tolerance: float = 0.01,
    rt_tolerance: float = 30.0,
) -> List[SILACPair]:
    """
    Find SILAC light/heavy (and optionally medium) peptide pairs.

    Args:
        peaks: List of detected peaks/features
        charge_states: Possible charge states to consider
        triple: If True, also search for medium label
        mz_tolerance: m/z tolerance for matching
        rt_tolerance: RT tolerance in seconds

    Returns:
        List of matched SILACPair objects
    """
    pairs = []
    used = set()

    for i, light_peak in enumerate(peaks):
        if i in used:
            continue

        for z in charge_states:
            # Expected heavy shift per charge
            heavy_shift_r = SILAC_SHIFTS["heavy"]["R"] / z
            heavy_shift_k = SILAC_SHIFTS["heavy"]["K"] / z

            for shift in [heavy_shift_r, heavy_shift_k]:
                expected_heavy_mz = light_peak.mz + shift

                for j, heavy_peak in enumerate(peaks):
                    if j in used or j == i:
                        continue

                    if (abs(heavy_peak.mz - expected_heavy_mz) <= mz_tolerance and
                            abs(heavy_peak.rt - light_peak.rt) <= rt_tolerance):

                        pair = SILACPair(
                            light_mz=light_peak.mz,
                            heavy_mz=heavy_peak.mz,
                            charge=z,
                            rt=(light_peak.rt + heavy_peak.rt) / 2,
                            light_intensity=light_peak.intensity,
                            heavy_intensity=heavy_peak.intensity,
                        )

                        if triple:
                            if shift == heavy_shift_r:
                                med_shift = SILAC_SHIFTS["medium"]["R"] / z
                            else:
                                med_shift = SILAC_SHIFTS["medium"]["K"] / z
                            expected_med_mz = light_peak.mz + med_shift

                            for k, med_peak in enumerate(peaks):
                                if k in used or k in (i, j):
                                    continue
                                if (abs(med_peak.mz - expected_med_mz) <= mz_tolerance and
                                        abs(med_peak.rt - light_peak.rt) <= rt_tolerance):
                                    pair.medium_mz = med_peak.mz
                                    pair.medium_intensity = med_peak.intensity
                                    used.add(k)
                                    break

                        pairs.append(pair)
                        used.add(i)
                        used.add(j)
                        break
                if i in used:
                    break
            if i in used:
                break

    return pairs


def compute_dimethyl_shift(
    peptide_sequence: str,
    label: str = "light",
) -> float:
    """
    Compute mass shift from dimethyl labeling.

    Dimethyl labeling modifies N-terminus and lysine residues.

    Args:
        peptide_sequence: Peptide sequence
        label: 'light', 'medium', or 'heavy'

    Returns:
        Total mass shift in Da
    """
    n_sites = 1 + peptide_sequence.upper().count("K")  # N-term + lysines
    shift_per_site = DIMETHYL_SHIFTS.get(label, 0.0)
    return n_sites * shift_per_site


def normalize_reporter_intensities(
    quants: List[ReporterIonQuantification],
    method: str = "median",
    reference_channel: Optional[str] = None,
) -> List[ReporterIonQuantification]:
    """
    Normalize reporter ion intensities across spectra.

    Args:
        quants: List of reporter quantifications
        method: 'median', 'mean', or 'sum'
        reference_channel: Optional reference channel for ratio normalization

    Returns:
        Normalized quantification list
    """
    if not quants:
        return quants

    channels = sorted(quants[0].channel_intensities.keys())
    n_channels = len(channels)

    # Build intensity matrix
    matrix = np.zeros((len(quants), n_channels))
    for i, q in enumerate(quants):
        for j, ch in enumerate(channels):
            matrix[i, j] = q.channel_intensities.get(ch, 0.0)

    # Replace zeros with small value for ratio computation
    matrix[matrix == 0] = 1.0

    if method == "median":
        # Column median normalization
        col_medians = np.median(matrix, axis=0)
        global_median = np.median(col_medians)
        if global_median > 0:
            factors = global_median / np.where(col_medians > 0, col_medians, 1.0)
            matrix *= factors
    elif method == "mean":
        col_means = np.mean(matrix, axis=0)
        global_mean = np.mean(col_means)
        if global_mean > 0:
            factors = global_mean / np.where(col_means > 0, col_means, 1.0)
            matrix *= factors
    elif method == "sum":
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

    # Update quantifications
    normalized = []
    for i, q in enumerate(quants):
        new_q = ReporterIonQuantification(
            spectrum_index=q.spectrum_index,
            precursor_mz=q.precursor_mz,
            channel_intensities={ch: float(matrix[i, j])
                                 for j, ch in enumerate(channels)},
            total_intensity=float(np.sum(matrix[i])),
            signal_to_noise=q.signal_to_noise,
            isolation_purity=q.isolation_purity,
        )
        normalized.append(new_q)

    return normalized


def aggregate_protein_ratios(
    peptide_ratios: Dict[str, List[float]],
    method: str = "median",
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate peptide-level ratios to protein-level.

    Args:
        peptide_ratios: Dict mapping protein -> list of peptide ratios
        method: Aggregation method ('median', 'mean')

    Returns:
        Dict mapping protein -> {'ratio': value, 'std': value, 'n_peptides': count}
    """
    results = {}
    for protein, ratios in peptide_ratios.items():
        arr = np.array(ratios)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue

        if method == "median":
            ratio = float(np.median(arr))
        else:
            ratio = float(np.mean(arr))

        results[protein] = {
            "ratio": ratio,
            "log2_ratio": float(np.log2(ratio)) if ratio > 0 else 0.0,
            "std": float(np.std(arr)),
            "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0.0,
            "n_peptides": len(arr),
        }

    return results


def isotope_correction(
    intensities: Dict[str, float],
    correction_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Apply isotope purity correction to reporter ion intensities.

    Args:
        intensities: Channel -> intensity mapping
        correction_matrix: Isotope purity correction matrix
            (channels x channels). If None, returns uncorrected.

    Returns:
        Corrected intensities
    """
    if correction_matrix is None:
        return intensities

    channels = sorted(intensities.keys())
    raw = np.array([intensities[ch] for ch in channels])

    try:
        corrected = np.linalg.solve(correction_matrix, raw)
        corrected = np.maximum(corrected, 0.0)
    except np.linalg.LinAlgError:
        return intensities

    return {ch: float(corrected[i]) for i, ch in enumerate(channels)}
