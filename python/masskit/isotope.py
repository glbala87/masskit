"""
Isotope pattern detection and charge state deconvolution for LC-MS data.

Provides tools for detecting isotope envelopes, assigning charge states,
and deconvoluting spectra to neutral masses.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import numpy as np
from math import factorial, exp, log

from .spectrum import Spectrum
from .peak import Peak, PeakList


# Averagine model: average amino acid composition
# C4.9384 H7.7583 N1.3577 O1.4773 S0.0417
AVERAGINE_C = 4.9384
AVERAGINE_H = 7.7583
AVERAGINE_N = 1.3577
AVERAGINE_O = 1.4773
AVERAGINE_S = 0.0417
AVERAGINE_MASS = (
    AVERAGINE_C * 12.0000
    + AVERAGINE_H * 1.00794
    + AVERAGINE_N * 14.0031
    + AVERAGINE_O * 15.9949
    + AVERAGINE_S * 31.9721
)

# Neutron mass difference (C13 - C12)
NEUTRON_MASS = 1.003355

# Proton mass
PROTON_MASS = 1.007276


@dataclass
class IsotopePattern:
    """
    Represents a detected isotope envelope.

    Attributes:
        monoisotopic_mz: m/z of the monoisotopic peak
        charge: Assigned charge state
        peaks: List of (mz, intensity) tuples for each isotope peak
        score: Quality of fit to theoretical pattern (0-1)
        rt: Retention time
    """
    monoisotopic_mz: float = 0.0
    charge: int = 1
    peaks: List[Tuple[float, float]] = field(default_factory=list)
    score: float = 0.0
    rt: float = 0.0

    @property
    def num_peaks(self) -> int:
        return len(self.peaks)

    def neutral_mass(self) -> float:
        """Calculate neutral mass from monoisotopic m/z and charge."""
        return (self.monoisotopic_mz - PROTON_MASS) * abs(self.charge)

    def total_intensity(self) -> float:
        """Sum of all isotope peak intensities."""
        return sum(intensity for _, intensity in self.peaks)

    def apex_intensity(self) -> float:
        """Maximum intensity among isotope peaks."""
        if not self.peaks:
            return 0.0
        return max(intensity for _, intensity in self.peaks)

    def mz_range(self) -> Tuple[float, float]:
        """Return (min_mz, max_mz) of the pattern."""
        if not self.peaks:
            return (0.0, 0.0)
        mzs = [mz for mz, _ in self.peaks]
        return (min(mzs), max(mzs))

    def __repr__(self) -> str:
        return (
            f"IsotopePattern(mz={self.monoisotopic_mz:.4f}, "
            f"z={self.charge}, peaks={self.num_peaks}, "
            f"score={self.score:.3f})"
        )


@dataclass
class DeconvolutedMass:
    """
    Represents a deconvoluted neutral mass with associated information.

    Attributes:
        neutral_mass: Calculated neutral (uncharged) mass
        intensity: Summed intensity across charge states
        charge_states: List of observed charge states
        patterns: List of IsotopePattern objects contributing to this mass
        quality_score: Overall quality score (0-1)
    """
    neutral_mass: float = 0.0
    intensity: float = 0.0
    charge_states: List[int] = field(default_factory=list)
    patterns: List[IsotopePattern] = field(default_factory=list)
    quality_score: float = 0.0

    @property
    def num_charge_states(self) -> int:
        return len(self.charge_states)

    def __repr__(self) -> str:
        return (
            f"DeconvolutedMass(mass={self.neutral_mass:.4f}, "
            f"charges={self.charge_states}, "
            f"score={self.quality_score:.3f})"
        )


def averagine_distribution(mass: float, num_peaks: int = 6) -> np.ndarray:
    """
    Generate theoretical isotope distribution using the averagine model.

    Uses a Poisson approximation based on the expected number of heavy
    isotopes for a molecule of the given mass.

    Args:
        mass: Neutral mass of the molecule
        num_peaks: Number of isotope peaks to generate

    Returns:
        Normalized intensity array for the isotope distribution

    Example:
        >>> dist = averagine_distribution(1000.0)
        >>> print(dist)  # [0.58, 0.31, 0.08, ...]
    """
    if mass <= 0:
        result = np.zeros(num_peaks)
        if num_peaks > 0:
            result[0] = 1.0
        return result

    # Number of averagine units
    n_units = mass / AVERAGINE_MASS

    # Expected number of C13 atoms (1.1% natural abundance of C13)
    n_carbon = n_units * AVERAGINE_C
    lambda_c13 = n_carbon * 0.0111

    # Additional contributions from other heavy isotopes
    # N15: 0.366%, O18: 0.205%, S34: 4.25%, H2: 0.012%
    lambda_other = (
        n_units * AVERAGINE_N * 0.00366
        + n_units * AVERAGINE_O * 0.00205
        + n_units * AVERAGINE_S * 0.0425
        + n_units * AVERAGINE_H * 0.00012
    )

    # Combined Poisson parameter
    lam = lambda_c13 + lambda_other

    # Poisson distribution
    distribution = np.zeros(num_peaks)
    for i in range(num_peaks):
        distribution[i] = exp(-lam) * (lam ** i) / factorial(i)

    # Normalize to max = 1
    max_val = np.max(distribution)
    if max_val > 0:
        distribution /= max_val

    return distribution


def assign_charge_state(
    mz_peaks: np.ndarray,
    mz_tolerance: float = 0.01,
    max_charge: int = 6,
) -> Tuple[int, float]:
    """
    Determine charge state from spacing between m/z peaks.

    Args:
        mz_peaks: Array of m/z values (sorted)
        mz_tolerance: Tolerance for matching expected spacing
        max_charge: Maximum charge state to consider

    Returns:
        Tuple of (charge_state, confidence).
        charge_state=0 if undetermined.

    Example:
        >>> charge, conf = assign_charge_state(np.array([500.0, 500.5, 501.0]))
        >>> print(f"Charge: +{charge}, confidence: {conf:.2f}")
    """
    if len(mz_peaks) < 2:
        return (0, 0.0)

    mz_sorted = np.sort(mz_peaks)
    spacings = np.diff(mz_sorted)

    best_charge = 0
    best_confidence = 0.0

    for charge in range(1, max_charge + 1):
        expected_spacing = NEUTRON_MASS / charge
        errors = np.abs(spacings - expected_spacing)
        matches = np.sum(errors <= mz_tolerance)
        if matches > 0:
            confidence = matches / len(spacings)
            avg_error = np.mean(errors[errors <= mz_tolerance])
            # Higher confidence for lower error and more matches
            confidence *= (1.0 - avg_error / mz_tolerance)
            if confidence > best_confidence:
                best_confidence = confidence
                best_charge = charge

    return (best_charge, best_confidence)


def detect_isotope_patterns(
    spectrum: Spectrum,
    charge_range: Tuple[int, int] = (1, 6),
    mz_tolerance: float = 0.01,
    min_peaks: int = 2,
    min_score: float = 0.5,
    min_intensity: float = 0.0,
) -> List[IsotopePattern]:
    """
    Detect isotope envelopes in a spectrum.

    Scans through peaks looking for series of peaks matching the expected
    isotope spacing for various charge states, then scores each pattern
    against the theoretical averagine distribution.

    Args:
        spectrum: Input spectrum (should be centroided or peak-picked)
        charge_range: (min_charge, max_charge) to consider
        mz_tolerance: m/z tolerance for matching isotope peaks (Da)
        min_peaks: Minimum number of isotope peaks required
        min_score: Minimum pattern score (cosine similarity, 0-1)
        min_intensity: Minimum intensity for seed peaks

    Returns:
        List of IsotopePattern objects sorted by score (descending)

    Example:
        >>> patterns = detect_isotope_patterns(spectrum, charge_range=(1, 4))
        >>> for p in patterns:
        ...     print(f"m/z={p.monoisotopic_mz:.4f} z={p.charge} score={p.score:.3f}")
    """
    if len(spectrum) < min_peaks:
        return []

    mz = spectrum.mz
    intensity = spectrum.intensity

    # Sort by m/z
    sort_idx = np.argsort(mz)
    mz = mz[sort_idx]
    intensity = intensity[sort_idx]

    used = np.zeros(len(mz), dtype=bool)
    patterns = []

    min_charge, max_charge = charge_range

    # Iterate through peaks as potential monoisotopic seeds
    # Start from highest intensity peaks for better results
    intensity_order = np.argsort(-intensity)

    for seed_idx in intensity_order:
        if used[seed_idx]:
            continue
        if intensity[seed_idx] < min_intensity:
            continue

        seed_mz = mz[seed_idx]
        seed_int = intensity[seed_idx]

        # Try each charge state
        best_pattern = None
        best_score = -1.0

        for charge in range(min_charge, max_charge + 1):
            expected_spacing = NEUTRON_MASS / charge
            pattern_peaks = [(seed_mz, seed_int)]
            pattern_indices = [seed_idx]

            # Look for subsequent isotope peaks
            for iso_num in range(1, 10):  # Up to 10 isotope peaks
                expected_mz = seed_mz + iso_num * expected_spacing
                # Find closest peak within tolerance
                mz_diffs = np.abs(mz - expected_mz)
                candidates = np.where(
                    (mz_diffs <= mz_tolerance) & (~used)
                )[0]

                if len(candidates) == 0:
                    break

                # Pick the closest
                best_cand = candidates[np.argmin(mz_diffs[candidates])]
                pattern_peaks.append((mz[best_cand], intensity[best_cand]))
                pattern_indices.append(best_cand)

            if len(pattern_peaks) < min_peaks:
                continue

            # Also check for peaks before the seed (seed might not be monoisotopic)
            pre_peaks = []
            pre_indices = []
            for iso_num in range(1, 4):
                expected_mz = seed_mz - iso_num * expected_spacing
                if expected_mz < 0:
                    break
                mz_diffs = np.abs(mz - expected_mz)
                candidates = np.where(
                    (mz_diffs <= mz_tolerance) & (~used)
                )[0]
                if len(candidates) == 0:
                    break
                best_cand = candidates[np.argmin(mz_diffs[candidates])]
                # Only include if intensity is lower (descending left side)
                if intensity[best_cand] < seed_int * 1.5:
                    pre_peaks.insert(0, (mz[best_cand], intensity[best_cand]))
                    pre_indices.insert(0, best_cand)
                else:
                    break

            all_peaks = pre_peaks + pattern_peaks
            all_indices = pre_indices + pattern_indices

            if len(all_peaks) < min_peaks:
                continue

            # Score against theoretical distribution
            mono_mz = all_peaks[0][0]
            neutral_mass = (mono_mz - PROTON_MASS) * charge
            theoretical = averagine_distribution(neutral_mass, len(all_peaks))

            observed = np.array([p[1] for p in all_peaks])
            observed_norm = observed / np.max(observed) if np.max(observed) > 0 else observed

            score = _cosine_similarity(theoretical, observed_norm)

            if score > best_score and score >= min_score:
                best_score = score
                best_pattern = IsotopePattern(
                    monoisotopic_mz=mono_mz,
                    charge=charge,
                    peaks=all_peaks,
                    score=score,
                    rt=spectrum.rt,
                )
                best_indices = all_indices

        if best_pattern is not None:
            patterns.append(best_pattern)
            for idx in best_indices:
                used[idx] = True

    # Sort by score descending
    patterns.sort(key=lambda p: p.score, reverse=True)
    return patterns


def deconvolute_spectrum(
    spectrum: Spectrum,
    charge_range: Tuple[int, int] = (1, 6),
    mz_tolerance: float = 0.01,
    mass_tolerance: float = 0.5,
    min_score: float = 0.3,
) -> List[DeconvolutedMass]:
    """
    Deconvolute a spectrum to neutral masses.

    Detects isotope patterns at various charge states, groups those that
    correspond to the same neutral mass, and produces a list of
    deconvoluted masses.

    Args:
        spectrum: Input spectrum
        charge_range: (min_charge, max_charge) to consider
        mz_tolerance: m/z tolerance for isotope pattern detection
        mass_tolerance: Tolerance for grouping patterns by neutral mass (Da)
        min_score: Minimum isotope pattern score

    Returns:
        List of DeconvolutedMass objects sorted by intensity (descending)

    Example:
        >>> masses = deconvolute_spectrum(spectrum)
        >>> for m in masses:
        ...     print(f"Mass={m.neutral_mass:.2f} Da, charges={m.charge_states}")
    """
    # Detect all isotope patterns
    patterns = detect_isotope_patterns(
        spectrum,
        charge_range=charge_range,
        mz_tolerance=mz_tolerance,
        min_peaks=2,
        min_score=min_score,
    )

    if not patterns:
        return []

    # Group patterns by neutral mass
    masses: List[DeconvolutedMass] = []

    for pattern in patterns:
        neutral_mass = pattern.neutral_mass()
        merged = False

        for dm in masses:
            if abs(dm.neutral_mass - neutral_mass) <= mass_tolerance:
                # Merge into existing group
                dm.intensity += pattern.total_intensity()
                if pattern.charge not in dm.charge_states:
                    dm.charge_states.append(pattern.charge)
                dm.patterns.append(pattern)
                # Update mass as weighted average
                total_int = sum(p.total_intensity() for p in dm.patterns)
                dm.neutral_mass = sum(
                    p.neutral_mass() * p.total_intensity() for p in dm.patterns
                ) / total_int if total_int > 0 else dm.neutral_mass
                merged = True
                break

        if not merged:
            masses.append(DeconvolutedMass(
                neutral_mass=neutral_mass,
                intensity=pattern.total_intensity(),
                charge_states=[pattern.charge],
                patterns=[pattern],
                quality_score=pattern.score,
            ))

    # Update quality scores based on number of charge states
    for dm in masses:
        avg_score = np.mean([p.score for p in dm.patterns])
        charge_bonus = min(0.2, 0.1 * (dm.num_charge_states - 1))
        dm.quality_score = min(1.0, avg_score + charge_bonus)

    # Sort by intensity descending
    masses.sort(key=lambda m: m.intensity, reverse=True)
    return masses


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))
