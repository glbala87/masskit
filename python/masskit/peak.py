"""
Peak and PeakList data structures.
"""

from typing import Optional, List, Dict, Tuple, Iterator
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Peak:
    """
    Represents a detected peak in a spectrum or chromatogram.

    Attributes:
        mz: m/z position
        rt: Retention time
        intensity: Peak apex intensity
        area: Integrated peak area
        fwhm_mz: Full width at half maximum (m/z)
        fwhm_rt: Full width at half maximum (RT)
        charge: Charge state (0 if unknown)
        snr: Signal-to-noise ratio
        quality: Overall quality score (0-1)
    """
    mz: float = 0.0
    rt: float = 0.0
    intensity: float = 0.0
    area: float = 0.0

    # Boundaries
    mz_left: float = 0.0
    mz_right: float = 0.0
    rt_left: float = 0.0
    rt_right: float = 0.0

    # Shape parameters
    fwhm_mz: float = 0.0
    fwhm_rt: float = 0.0
    asymmetry: float = 1.0

    # Charge and isotopes
    charge: int = 0
    monoisotopic_mz: Optional[float] = None
    isotope_index: int = 0

    # Quality metrics
    snr: float = 0.0
    quality: float = 0.0

    # Metadata
    spectrum_index: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def mz_width(self) -> float:
        """Get m/z width."""
        return self.mz_right - self.mz_left

    @property
    def rt_width(self) -> float:
        """Get RT width."""
        return self.rt_right - self.rt_left

    def neutral_mass(self) -> float:
        """Calculate neutral mass from m/z and charge."""
        if self.charge == 0:
            return 0.0
        proton_mass = 1.007276
        return (self.mz - proton_mass) * abs(self.charge)

    def contains_mz(self, mz: float) -> bool:
        """Check if m/z is within peak boundaries."""
        return self.mz_left <= mz <= self.mz_right

    def contains_rt(self, rt: float) -> bool:
        """Check if RT is within peak boundaries."""
        return self.rt_left <= rt <= self.rt_right

    def contains(self, mz: float, rt: float) -> bool:
        """Check if point is within peak boundaries (2D)."""
        return self.contains_mz(mz) and self.contains_rt(rt)


class PeakList:
    """
    Collection of peaks with numpy-backed arrays for efficient access.

    Provides both individual Peak access and numpy array access for
    batch operations.

    Example:
        >>> peaks = PeakList()
        >>> peaks.add(Peak(mz=100.0, intensity=1000))
        >>> peaks.add(Peak(mz=200.0, intensity=5000))
        >>> peaks.mz_array
        array([100., 200.])
        >>> peaks.filter_by_intensity(2000)
        PeakList with 1 peaks
    """

    def __init__(self, peaks: Optional[List[Peak]] = None):
        """
        Create a new PeakList.

        Args:
            peaks: Optional list of peaks
        """
        self._peaks: List[Peak] = peaks if peaks is not None else []

    def __len__(self) -> int:
        return len(self._peaks)

    def __bool__(self) -> bool:
        return len(self._peaks) > 0

    def __getitem__(self, idx: int) -> Peak:
        return self._peaks[idx]

    def __iter__(self) -> Iterator[Peak]:
        return iter(self._peaks)

    def add(self, peak: Peak) -> None:
        """Add a peak to the list."""
        self._peaks.append(peak)

    def extend(self, peaks: List[Peak]) -> None:
        """Add multiple peaks."""
        self._peaks.extend(peaks)

    def clear(self) -> None:
        """Remove all peaks."""
        self._peaks.clear()

    @property
    def size(self) -> int:
        """Get number of peaks."""
        return len(self._peaks)

    @property
    def mz_array(self) -> np.ndarray:
        """Get m/z values as numpy array."""
        return np.array([p.mz for p in self._peaks], dtype=np.float64)

    @property
    def rt_array(self) -> np.ndarray:
        """Get RT values as numpy array."""
        return np.array([p.rt for p in self._peaks], dtype=np.float64)

    @property
    def intensity_array(self) -> np.ndarray:
        """Get intensity values as numpy array."""
        return np.array([p.intensity for p in self._peaks], dtype=np.float64)

    @property
    def area_array(self) -> np.ndarray:
        """Get area values as numpy array."""
        return np.array([p.area for p in self._peaks], dtype=np.float64)

    @property
    def charge_array(self) -> np.ndarray:
        """Get charge states as numpy array."""
        return np.array([p.charge for p in self._peaks], dtype=np.int32)

    @property
    def snr_array(self) -> np.ndarray:
        """Get SNR values as numpy array."""
        return np.array([p.snr for p in self._peaks], dtype=np.float64)

    def sort_by_mz(self) -> "PeakList":
        """Sort by m/z (in-place) and return self."""
        self._peaks.sort(key=lambda p: p.mz)
        return self

    def sort_by_rt(self) -> "PeakList":
        """Sort by RT (in-place) and return self."""
        self._peaks.sort(key=lambda p: p.rt)
        return self

    def sort_by_intensity(self, descending: bool = True) -> "PeakList":
        """Sort by intensity (in-place) and return self."""
        self._peaks.sort(key=lambda p: p.intensity, reverse=descending)
        return self

    def filter_by_mz(self, mz_min: float, mz_max: float) -> "PeakList":
        """
        Filter peaks by m/z range.

        Args:
            mz_min: Minimum m/z
            mz_max: Maximum m/z

        Returns:
            New PeakList with filtered peaks
        """
        return PeakList([p for p in self._peaks if mz_min <= p.mz <= mz_max])

    def filter_by_rt(self, rt_min: float, rt_max: float) -> "PeakList":
        """
        Filter peaks by RT range.

        Args:
            rt_min: Minimum RT
            rt_max: Maximum RT

        Returns:
            New PeakList with filtered peaks
        """
        return PeakList([p for p in self._peaks if rt_min <= p.rt <= rt_max])

    def filter_by_intensity(self, min_intensity: float) -> "PeakList":
        """
        Filter peaks by minimum intensity.

        Args:
            min_intensity: Minimum intensity threshold

        Returns:
            New PeakList with filtered peaks
        """
        return PeakList([p for p in self._peaks if p.intensity >= min_intensity])

    def filter_by_snr(self, min_snr: float) -> "PeakList":
        """
        Filter peaks by minimum SNR.

        Args:
            min_snr: Minimum SNR threshold

        Returns:
            New PeakList with filtered peaks
        """
        return PeakList([p for p in self._peaks if p.snr >= min_snr])

    def filter_by_charge(self, charges: List[int]) -> "PeakList":
        """
        Filter peaks by charge state.

        Args:
            charges: List of allowed charge states

        Returns:
            New PeakList with filtered peaks
        """
        return PeakList([p for p in self._peaks if p.charge in charges])

    def find_nearest_mz(self, target: float) -> Optional[Peak]:
        """
        Find peak closest to target m/z.

        Args:
            target: Target m/z

        Returns:
            Nearest peak or None if list is empty
        """
        if not self._peaks:
            return None
        return min(self._peaks, key=lambda p: abs(p.mz - target))

    def find_nearest_rt(self, target: float) -> Optional[Peak]:
        """
        Find peak closest to target RT.

        Args:
            target: Target RT

        Returns:
            Nearest peak or None if list is empty
        """
        if not self._peaks:
            return None
        return min(self._peaks, key=lambda p: abs(p.rt - target))

    def top_n(self, n: int) -> "PeakList":
        """
        Get top N peaks by intensity.

        Args:
            n: Number of peaks

        Returns:
            New PeakList with top N peaks
        """
        sorted_peaks = sorted(self._peaks, key=lambda p: p.intensity, reverse=True)
        return PeakList(sorted_peaks[:n])

    def to_dataframe(self):
        """
        Convert to pandas DataFrame.

        Returns:
            DataFrame with peak properties
        """
        import pandas as pd
        return pd.DataFrame([
            {
                "mz": p.mz,
                "rt": p.rt,
                "intensity": p.intensity,
                "area": p.area,
                "charge": p.charge,
                "snr": p.snr,
                "quality": p.quality,
                "fwhm_mz": p.fwhm_mz,
                "fwhm_rt": p.fwhm_rt,
            }
            for p in self._peaks
        ])

    @classmethod
    def from_dataframe(cls, df) -> "PeakList":
        """
        Create PeakList from pandas DataFrame.

        Args:
            df: DataFrame with peak properties

        Returns:
            New PeakList
        """
        peaks = []
        for _, row in df.iterrows():
            peak = Peak(
                mz=row.get("mz", 0.0),
                rt=row.get("rt", 0.0),
                intensity=row.get("intensity", 0.0),
                area=row.get("area", 0.0),
                charge=int(row.get("charge", 0)),
                snr=row.get("snr", 0.0),
                quality=row.get("quality", 0.0),
            )
            peaks.append(peak)
        return cls(peaks)

    def copy(self) -> "PeakList":
        """Create a deep copy."""
        return PeakList([Peak(**vars(p)) for p in self._peaks])

    def __repr__(self) -> str:
        return f"PeakList(n={len(self._peaks)})"
