"""
MS/MS spectral matching and library search for LC-MS data.

Provides spectral similarity scoring, library management, and
preprocessing functions for tandem mass spectrometry data.
"""

from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import re

from .spectrum import Spectrum


@dataclass
class SpectralMatch:
    """
    Result of a spectral library search.

    Attributes:
        query_index: Index of the query spectrum
        library_name: Name of the matched library entry
        score: Similarity score (0-1)
        matched_peaks: Number of matched peak pairs
        precursor_mz: Precursor m/z of library entry
        metadata: Additional metadata from library entry
    """
    query_index: int = 0
    library_name: str = ""
    score: float = 0.0
    matched_peaks: int = 0
    precursor_mz: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SpectralMatch(name='{self.library_name}', "
            f"score={self.score:.4f}, matched={self.matched_peaks})"
        )


@dataclass
class LibrarySpectrum:
    """A spectrum entry in a spectral library."""
    name: str = ""
    precursor_mz: float = 0.0
    mz: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    intensity: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def num_peaks(self) -> int:
        return len(self.mz)


class SpectralLibrary:
    """
    Spectral library for MS/MS matching.

    Supports loading from MGF and MSP formats, adding spectra manually,
    and searching with multiple similarity metrics.

    Example:
        >>> lib = SpectralLibrary()
        >>> lib.load_mgf("library.mgf")
        >>> matches = lib.search(query_spectrum, method='cosine', top_n=5)
    """

    def __init__(self):
        self._entries: List[LibrarySpectrum] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> LibrarySpectrum:
        return self._entries[idx]

    def add_spectrum(
        self,
        name: str,
        precursor_mz: float,
        mz: np.ndarray,
        intensity: np.ndarray,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a spectrum to the library."""
        entry = LibrarySpectrum(
            name=name,
            precursor_mz=precursor_mz,
            mz=np.asarray(mz, dtype=np.float64),
            intensity=np.asarray(intensity, dtype=np.float64),
            metadata=metadata or {},
        )
        self._entries.append(entry)

    def load_mgf(self, filepath: str) -> int:
        """
        Load spectra from an MGF file.

        Args:
            filepath: Path to MGF file

        Returns:
            Number of spectra loaded
        """
        count = 0
        with open(filepath, "r") as f:
            in_ions = False
            current_name = ""
            current_precursor = 0.0
            current_metadata: Dict[str, str] = {}
            mz_list: List[float] = []
            int_list: List[float] = []

            for line in f:
                line = line.strip()

                if line == "BEGIN IONS":
                    in_ions = True
                    current_name = ""
                    current_precursor = 0.0
                    current_metadata = {}
                    mz_list = []
                    int_list = []
                elif line == "END IONS":
                    if mz_list:
                        self.add_spectrum(
                            name=current_name,
                            precursor_mz=current_precursor,
                            mz=np.array(mz_list),
                            intensity=np.array(int_list),
                            metadata=current_metadata,
                        )
                        count += 1
                    in_ions = False
                elif in_ions:
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip().upper()
                        value = value.strip()
                        if key == "TITLE":
                            current_name = value
                        elif key == "PEPMASS":
                            parts = value.split()
                            current_precursor = float(parts[0])
                        elif key == "CHARGE":
                            current_metadata["charge"] = value.rstrip("+-")
                        else:
                            current_metadata[key.lower()] = value
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                mz_list.append(float(parts[0]))
                                int_list.append(float(parts[1]))
                            except ValueError:
                                pass

        return count

    def load_msp(self, filepath: str) -> int:
        """
        Load spectra from an MSP/NIST format file.

        Args:
            filepath: Path to MSP file

        Returns:
            Number of spectra loaded
        """
        count = 0
        with open(filepath, "r") as f:
            current_name = ""
            current_precursor = 0.0
            current_metadata: Dict[str, str] = {}
            mz_list: List[float] = []
            int_list: List[float] = []
            num_peaks_expected = 0
            reading_peaks = False

            for line in f:
                line = line.strip()
                if not line:
                    if mz_list:
                        self.add_spectrum(
                            name=current_name,
                            precursor_mz=current_precursor,
                            mz=np.array(mz_list),
                            intensity=np.array(int_list),
                            metadata=current_metadata,
                        )
                        count += 1
                    current_name = ""
                    current_precursor = 0.0
                    current_metadata = {}
                    mz_list = []
                    int_list = []
                    reading_peaks = False
                    continue

                if ":" in line and not reading_peaks:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    key_upper = key.upper()

                    if key_upper == "NAME":
                        current_name = value
                    elif key_upper in ("PRECURSORMZ", "PRECURSOR_MZ"):
                        current_precursor = float(value)
                    elif key_upper in ("NUM PEAKS", "NUMPEAKS", "NUM_PEAKS"):
                        num_peaks_expected = int(value)
                        reading_peaks = True
                    else:
                        current_metadata[key.lower()] = value
                elif reading_peaks:
                    # Parse peak line: "mz intensity" or "mz\tintensity"
                    parts = re.split(r"[\t ]+", line)
                    if len(parts) >= 2:
                        try:
                            mz_list.append(float(parts[0]))
                            int_list.append(float(parts[1]))
                        except ValueError:
                            pass

            # Handle last entry
            if mz_list:
                self.add_spectrum(
                    name=current_name,
                    precursor_mz=current_precursor,
                    mz=np.array(mz_list),
                    intensity=np.array(int_list),
                    metadata=current_metadata,
                )
                count += 1

        return count

    def save(self, filepath: str, format: str = "mgf") -> None:
        """
        Save library to file.

        Args:
            filepath: Output file path
            format: Output format ('mgf' or 'msp')
        """
        if format == "mgf":
            self._save_mgf(filepath)
        elif format == "msp":
            self._save_msp(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _save_mgf(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            for entry in self._entries:
                f.write("BEGIN IONS\n")
                f.write(f"TITLE={entry.name}\n")
                f.write(f"PEPMASS={entry.precursor_mz}\n")
                for key, value in entry.metadata.items():
                    f.write(f"{key.upper()}={value}\n")
                for mz, intensity in zip(entry.mz, entry.intensity):
                    f.write(f"{mz:.6f} {intensity:.4f}\n")
                f.write("END IONS\n\n")

    def _save_msp(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            for entry in self._entries:
                f.write(f"Name: {entry.name}\n")
                f.write(f"PrecursorMZ: {entry.precursor_mz}\n")
                for key, value in entry.metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"Num Peaks: {entry.num_peaks}\n")
                for mz, intensity in zip(entry.mz, entry.intensity):
                    f.write(f"{mz:.6f}\t{intensity:.4f}\n")
                f.write("\n")

    def search(
        self,
        query_mz: np.ndarray,
        query_intensity: np.ndarray,
        query_precursor_mz: float = 0.0,
        method: str = "cosine",
        tolerance: float = 0.01,
        min_score: float = 0.0,
        top_n: int = 10,
        precursor_tolerance: float = 0.0,
    ) -> List[SpectralMatch]:
        """
        Search library for matching spectra.

        Args:
            query_mz: Query spectrum m/z values
            query_intensity: Query spectrum intensities
            query_precursor_mz: Query precursor m/z (for filtering and modified cosine)
            method: Similarity method ('cosine', 'modified_cosine', 'entropy')
            tolerance: m/z tolerance for peak matching
            min_score: Minimum score threshold
            top_n: Return top N matches
            precursor_tolerance: Filter library by precursor m/z (0 = no filter)

        Returns:
            List of SpectralMatch objects sorted by score (descending)
        """
        results = []

        for i, entry in enumerate(self._entries):
            # Filter by precursor m/z if specified
            if precursor_tolerance > 0 and query_precursor_mz > 0:
                if abs(entry.precursor_mz - query_precursor_mz) > precursor_tolerance:
                    continue

            if method == "cosine":
                score, n_matched = cosine_similarity(
                    query_mz, query_intensity,
                    entry.mz, entry.intensity,
                    tolerance=tolerance,
                )
            elif method == "modified_cosine":
                score, n_matched = modified_cosine_similarity(
                    query_mz, query_intensity, query_precursor_mz,
                    entry.mz, entry.intensity, entry.precursor_mz,
                    tolerance=tolerance,
                )
            elif method == "entropy":
                score, n_matched = spectral_entropy_similarity(
                    query_mz, query_intensity,
                    entry.mz, entry.intensity,
                    tolerance=tolerance,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            if score >= min_score:
                results.append(SpectralMatch(
                    query_index=0,
                    library_name=entry.name,
                    score=score,
                    matched_peaks=n_matched,
                    precursor_mz=entry.precursor_mz,
                    metadata=entry.metadata,
                ))

        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_n]

    def filter_by_precursor(
        self, mz_min: float, mz_max: float
    ) -> "SpectralLibrary":
        """Return a new library filtered by precursor m/z range."""
        lib = SpectralLibrary()
        lib._entries = [
            e for e in self._entries
            if mz_min <= e.precursor_mz <= mz_max
        ]
        return lib


def cosine_similarity(
    mz1: np.ndarray,
    int1: np.ndarray,
    mz2: np.ndarray,
    int2: np.ndarray,
    tolerance: float = 0.01,
) -> Tuple[float, int]:
    """
    Compute cosine similarity between two spectra.

    Args:
        mz1, int1: First spectrum
        mz2, int2: Second spectrum
        tolerance: m/z tolerance for peak matching

    Returns:
        Tuple of (similarity_score, number_of_matched_peaks)
    """
    matched = _match_peaks(mz1, int1, mz2, int2, tolerance)
    if not matched:
        return (0.0, 0)

    vec1 = np.array([m[0] for m in matched])
    vec2 = np.array([m[1] for m in matched])

    dot = np.dot(vec1, vec2)
    norm1 = np.sqrt(np.dot(int1, int1))
    norm2 = np.sqrt(np.dot(int2, int2))

    if norm1 == 0 or norm2 == 0:
        return (0.0, 0)

    score = float(dot / (norm1 * norm2))
    return (min(score, 1.0), len(matched))


def modified_cosine_similarity(
    mz1: np.ndarray,
    int1: np.ndarray,
    precursor1: float,
    mz2: np.ndarray,
    int2: np.ndarray,
    precursor2: float,
    tolerance: float = 0.01,
) -> Tuple[float, int]:
    """
    Compute modified cosine similarity (shift-aware).

    Accounts for the mass difference between precursors when matching
    fragment ions, enabling matching of spectra from related compounds.

    Args:
        mz1, int1: First spectrum
        precursor1: Precursor m/z of first spectrum
        mz2, int2: Second spectrum
        precursor2: Precursor m/z of second spectrum
        tolerance: m/z tolerance for peak matching

    Returns:
        Tuple of (similarity_score, number_of_matched_peaks)
    """
    mass_diff = precursor1 - precursor2

    # Direct matches
    direct_matched = _match_peaks(mz1, int1, mz2, int2, tolerance)

    # Shifted matches: shift mz2 by mass difference
    shifted_mz2 = mz2 + mass_diff
    shifted_matched = _match_peaks(mz1, int1, shifted_mz2, int2, tolerance)

    # Combine matches (avoid duplicate assignments)
    all_matched = direct_matched + shifted_matched
    if not all_matched:
        return (0.0, 0)

    vec1 = np.array([m[0] for m in all_matched])
    vec2 = np.array([m[1] for m in all_matched])

    dot = np.dot(vec1, vec2)
    norm1 = np.sqrt(np.dot(int1, int1))
    norm2 = np.sqrt(np.dot(int2, int2))

    if norm1 == 0 or norm2 == 0:
        return (0.0, 0)

    score = float(dot / (norm1 * norm2))
    return (min(score, 1.0), len(all_matched))


def spectral_entropy_similarity(
    mz1: np.ndarray,
    int1: np.ndarray,
    mz2: np.ndarray,
    int2: np.ndarray,
    tolerance: float = 0.01,
) -> Tuple[float, int]:
    """
    Compute spectral entropy similarity.

    Uses Shannon entropy to weight the similarity calculation,
    giving more weight to peaks that carry more information.

    Args:
        mz1, int1: First spectrum
        mz2, int2: Second spectrum
        tolerance: m/z tolerance for peak matching

    Returns:
        Tuple of (similarity_score, number_of_matched_peaks)
    """
    # Normalize to probability distributions
    sum1 = np.sum(int1)
    sum2 = np.sum(int2)
    if sum1 == 0 or sum2 == 0:
        return (0.0, 0)

    p1 = int1 / sum1
    p2 = int2 / sum2

    matched = _match_peaks(mz1, p1, mz2, p2, tolerance)
    if not matched:
        return (0.0, 0)

    # Entropy of individual spectra
    entropy1 = _shannon_entropy(p1)
    entropy2 = _shannon_entropy(p2)

    # Create merged spectrum
    merged_int = np.zeros(len(matched))
    for i, (a, b) in enumerate(matched):
        merged_int[i] = (a + b) / 2

    # Entropy of merged
    if np.sum(merged_int) > 0:
        merged_int /= np.sum(merged_int)
    entropy_merged = _shannon_entropy(merged_int)

    # Jensen-Shannon divergence based similarity
    avg_entropy = (entropy1 + entropy2) / 2
    if avg_entropy == 0:
        return (0.0, len(matched))

    jsd = entropy_merged - avg_entropy
    # Normalize: max JSD = log(2)
    max_jsd = np.log(2)
    similarity = 1.0 - min(jsd / max_jsd, 1.0) if max_jsd > 0 else 0.0

    return (max(0.0, similarity), len(matched))


def matched_peaks_count(
    mz1: np.ndarray,
    int1: np.ndarray,
    mz2: np.ndarray,
    int2: np.ndarray,
    tolerance: float = 0.01,
) -> int:
    """Count the number of matched peaks between two spectra."""
    matched = _match_peaks(mz1, int1, mz2, int2, tolerance)
    return len(matched)


# =========================================================================
# Preprocessing functions
# =========================================================================

def normalize_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    method: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize spectrum intensities.

    Args:
        mz: m/z values
        intensity: Intensity values
        method: 'max' (scale to max=1), 'sum' (scale to sum=1), 'sqrt' (square root)

    Returns:
        Tuple of (mz, normalized_intensity)
    """
    intensity = intensity.copy()
    if method == "max":
        max_val = np.max(intensity)
        if max_val > 0:
            intensity /= max_val
    elif method == "sum":
        total = np.sum(intensity)
        if total > 0:
            intensity /= total
    elif method == "sqrt":
        intensity = np.sqrt(np.maximum(intensity, 0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return (mz.copy(), intensity)


def filter_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    min_mz: float = 0.0,
    max_mz: Optional[float] = None,
    min_intensity: float = 0.0,
    top_n: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter spectrum peaks.

    Args:
        mz: m/z values
        intensity: Intensity values
        min_mz: Minimum m/z
        max_mz: Maximum m/z (None = no limit)
        min_intensity: Minimum intensity
        top_n: Keep only top N peaks by intensity

    Returns:
        Tuple of (filtered_mz, filtered_intensity)
    """
    mask = np.ones(len(mz), dtype=bool)
    mask &= mz >= min_mz
    if max_mz is not None:
        mask &= mz <= max_mz
    mask &= intensity >= min_intensity

    mz = mz[mask]
    intensity = intensity[mask]

    if top_n is not None and len(mz) > top_n:
        top_idx = np.argsort(-intensity)[:top_n]
        top_idx = np.sort(top_idx)
        mz = mz[top_idx]
        intensity = intensity[top_idx]

    return (mz, intensity)


def sqrt_transform(intensity: np.ndarray) -> np.ndarray:
    """Apply square root transformation to intensities."""
    return np.sqrt(np.maximum(intensity, 0))


def remove_precursor(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    tolerance: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove precursor ion and related peaks.

    Removes the precursor peak and common neutral losses
    (water: -18.011, ammonia: -17.027).

    Args:
        mz: m/z values
        intensity: Intensity values
        precursor_mz: Precursor m/z
        tolerance: m/z tolerance window around precursor

    Returns:
        Tuple of (filtered_mz, filtered_intensity)
    """
    remove_mzs = [
        precursor_mz,
        precursor_mz - 18.011,  # water loss
        precursor_mz - 17.027,  # ammonia loss
        precursor_mz - 35.038,  # water + ammonia loss
    ]

    mask = np.ones(len(mz), dtype=bool)
    for rm_mz in remove_mzs:
        mask &= np.abs(mz - rm_mz) > tolerance

    return (mz[mask], intensity[mask])


# =========================================================================
# Internal helpers
# =========================================================================

def _match_peaks(
    mz1: np.ndarray,
    int1: np.ndarray,
    mz2: np.ndarray,
    int2: np.ndarray,
    tolerance: float,
) -> List[Tuple[float, float]]:
    """Match peaks between two spectra by m/z proximity."""
    if len(mz1) == 0 or len(mz2) == 0:
        return []

    matched = []
    used2 = set()

    # Sort by m/z
    order1 = np.argsort(mz1)
    order2 = np.argsort(mz2)
    sorted_mz2 = mz2[order2]

    for idx1 in order1:
        best_idx2 = -1
        best_diff = tolerance + 1.0

        # Binary search for closest in sorted_mz2
        target = mz1[idx1]
        lo, hi = 0, len(sorted_mz2) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_mz2[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1

        # Check nearby candidates
        for j in range(max(0, lo - 1), min(len(sorted_mz2), lo + 2)):
            orig_j = order2[j]
            if orig_j in used2:
                continue
            diff = abs(mz1[idx1] - mz2[orig_j])
            if diff <= tolerance and diff < best_diff:
                best_diff = diff
                best_idx2 = orig_j

        if best_idx2 >= 0:
            matched.append((int1[idx1], int2[best_idx2]))
            used2.add(best_idx2)

    return matched


def _shannon_entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))
