"""
Chromatogram data structure for LC-MS data.
"""

from enum import Enum
from typing import Optional, List, Dict, Tuple, Union
import numpy as np


class ChromatogramType(Enum):
    """Type of chromatogram."""
    UNKNOWN = 0
    TIC = 1  # Total Ion Current
    BPC = 2  # Base Peak Chromatogram
    XIC = 3  # Extracted Ion Chromatogram
    SRM = 4  # Selected Reaction Monitoring
    MRM = 5  # Multiple Reaction Monitoring
    SIM = 6  # Selected Ion Monitoring


class Chromatogram:
    """
    Chromatogram (intensity vs retention time).

    A Chromatogram stores paired arrays of retention times and intensities,
    along with metadata describing the chromatogram type and acquisition
    parameters.

    Attributes:
        rt: Retention time values in seconds
        intensity: Intensity values
        chrom_type: Type of chromatogram (TIC, BPC, XIC, etc.)
        target_mz: Target m/z for XIC/SIM
        mz_tolerance: m/z tolerance for XIC

    Example:
        >>> chrom = Chromatogram(rt=[0, 60, 120], intensity=[100, 500, 200])
        >>> chrom.apex_rt
        60.0
        >>> chrom.compute_area()
        36000.0
    """

    def __init__(
        self,
        rt: Optional[Union[np.ndarray, List[float]]] = None,
        intensity: Optional[Union[np.ndarray, List[float]]] = None,
        chrom_type: ChromatogramType = ChromatogramType.UNKNOWN,
        target_mz: float = 0.0,
        mz_tolerance: float = 0.0,
    ):
        """
        Create a new Chromatogram.

        Args:
            rt: Retention time values in seconds
            intensity: Intensity values
            chrom_type: Type of chromatogram
            target_mz: Target m/z for XIC/SIM
            mz_tolerance: m/z tolerance for XIC
        """
        if rt is not None:
            self._rt = np.asarray(rt, dtype=np.float64)
        else:
            self._rt = np.array([], dtype=np.float64)

        if intensity is not None:
            self._intensity = np.asarray(intensity, dtype=np.float64)
        else:
            self._intensity = np.array([], dtype=np.float64)

        if len(self._rt) != len(self._intensity):
            raise ValueError("RT and intensity arrays must have same length")

        self.chrom_type = chrom_type
        self.target_mz = target_mz
        self.mz_tolerance = mz_tolerance
        self.index: int = 0
        self.native_id: str = ""
        self.precursor_mz: float = 0.0  # For SRM/MRM
        self.product_mz: float = 0.0  # For SRM/MRM
        self.metadata: Dict[str, str] = {}

        self._update_cache()

    def _update_cache(self) -> None:
        """Update cached statistics."""
        if len(self._intensity) > 0:
            max_idx = int(np.argmax(self._intensity))
            self._max_intensity = float(self._intensity[max_idx])
            self._apex_rt = float(self._rt[max_idx])
            self._rt_min = float(np.min(self._rt))
            self._rt_max = float(np.max(self._rt))
        else:
            self._max_intensity = 0.0
            self._apex_rt = 0.0
            self._rt_min = 0.0
            self._rt_max = 0.0

    @property
    def rt(self) -> np.ndarray:
        """Get retention time array."""
        return self._rt

    @rt.setter
    def rt(self, value: Union[np.ndarray, List[float]]) -> None:
        """Set retention time array."""
        self._rt = np.asarray(value, dtype=np.float64)
        self._update_cache()

    @property
    def intensity(self) -> np.ndarray:
        """Get intensity array."""
        return self._intensity

    @intensity.setter
    def intensity(self, value: Union[np.ndarray, List[float]]) -> None:
        """Set intensity array."""
        self._intensity = np.asarray(value, dtype=np.float64)
        self._update_cache()

    def __len__(self) -> int:
        """Return number of data points."""
        return len(self._rt)

    def __bool__(self) -> bool:
        """Check if chromatogram has data."""
        return len(self._rt) > 0

    @property
    def size(self) -> int:
        """Get number of data points."""
        return len(self._rt)

    @property
    def max_intensity(self) -> float:
        """Get maximum intensity."""
        return self._max_intensity

    @property
    def apex_rt(self) -> float:
        """Get retention time at maximum intensity."""
        return self._apex_rt

    @property
    def rt_range(self) -> Tuple[float, float]:
        """Get (min, max) retention time range."""
        return (self._rt_min, self._rt_max)

    def is_sorted(self) -> bool:
        """Check if RT array is sorted."""
        return np.all(self._rt[:-1] <= self._rt[1:]) if len(self._rt) > 1 else True

    def sort_by_rt(self) -> "Chromatogram":
        """Sort by retention time (in-place) and return self."""
        if len(self._rt) > 1:
            indices = np.argsort(self._rt)
            self._rt = self._rt[indices]
            self._intensity = self._intensity[indices]
        return self

    def find_nearest_rt(self, target: float) -> int:
        """
        Find index of point closest to target RT.

        Args:
            target: Target retention time

        Returns:
            Index of nearest point
        """
        if len(self._rt) == 0:
            raise ValueError("Cannot find point in empty chromatogram")
        return int(np.argmin(np.abs(self._rt - target)))

    def extract_range(self, rt_min: float, rt_max: float) -> "Chromatogram":
        """
        Extract points within RT range.

        Args:
            rt_min: Minimum retention time
            rt_max: Maximum retention time

        Returns:
            New Chromatogram with only points in range
        """
        mask = (self._rt >= rt_min) & (self._rt <= rt_max)
        return Chromatogram(
            rt=self._rt[mask].copy(),
            intensity=self._intensity[mask].copy(),
            chrom_type=self.chrom_type,
            target_mz=self.target_mz,
            mz_tolerance=self.mz_tolerance,
        )

    def compute_area(self, rt_min: Optional[float] = None,
                      rt_max: Optional[float] = None) -> float:
        """
        Compute area under the curve using trapezoidal integration.

        Args:
            rt_min: Optional minimum RT for integration
            rt_max: Optional maximum RT for integration

        Returns:
            Integrated area
        """
        if len(self._rt) < 2:
            return 0.0

        if rt_min is not None or rt_max is not None:
            chrom = self.extract_range(
                rt_min if rt_min is not None else self._rt_min,
                rt_max if rt_max is not None else self._rt_max,
            )
            return chrom.compute_area()

        _trap = getattr(np, "trapezoid", np.trapz)
        return float(_trap(self._intensity, self._rt))

    def interpolate_at(self, rt: float) -> float:
        """
        Interpolate intensity at given RT.

        Args:
            rt: Retention time

        Returns:
            Interpolated intensity
        """
        if len(self._rt) == 0:
            return 0.0
        return float(np.interp(rt, self._rt, self._intensity))

    def resample(self, num_points: int) -> "Chromatogram":
        """
        Resample chromatogram to uniform spacing.

        Args:
            num_points: Number of points in output

        Returns:
            Resampled chromatogram
        """
        if len(self._rt) < 2:
            return self.copy()

        new_rt = np.linspace(self._rt_min, self._rt_max, num_points)
        new_intensity = np.interp(new_rt, self._rt, self._intensity)

        return Chromatogram(
            rt=new_rt,
            intensity=new_intensity,
            chrom_type=self.chrom_type,
            target_mz=self.target_mz,
            mz_tolerance=self.mz_tolerance,
        )

    def smooth(self, window_size: int = 5) -> "Chromatogram":
        """
        Apply moving average smoothing.

        Args:
            window_size: Size of smoothing window

        Returns:
            Smoothed chromatogram
        """
        if len(self._intensity) < window_size:
            return self.copy()

        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(self._intensity, kernel, mode='same')

        return Chromatogram(
            rt=self._rt.copy(),
            intensity=smoothed,
            chrom_type=self.chrom_type,
            target_mz=self.target_mz,
            mz_tolerance=self.mz_tolerance,
        )

    def normalize(self, method: str = "max") -> "Chromatogram":
        """
        Normalize intensities.

        Args:
            method: Normalization method ("max", "sum", "area")

        Returns:
            Normalized chromatogram
        """
        intensity = self._intensity.copy()
        if len(intensity) == 0:
            return self.copy()

        if method == "max":
            factor = np.max(intensity)
        elif method == "sum":
            factor = np.sum(intensity)
        elif method == "area":
            factor = self.compute_area()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if factor > 0:
            intensity /= factor

        return Chromatogram(
            rt=self._rt.copy(),
            intensity=intensity,
            chrom_type=self.chrom_type,
            target_mz=self.target_mz,
            mz_tolerance=self.mz_tolerance,
        )

    def copy(self) -> "Chromatogram":
        """Create a deep copy."""
        chrom = Chromatogram(
            rt=self._rt.copy(),
            intensity=self._intensity.copy(),
            chrom_type=self.chrom_type,
            target_mz=self.target_mz,
            mz_tolerance=self.mz_tolerance,
        )
        chrom.index = self.index
        chrom.native_id = self.native_id
        chrom.precursor_mz = self.precursor_mz
        chrom.product_mz = self.product_mz
        chrom.metadata = self.metadata.copy()
        return chrom

    def to_dataframe(self):
        """
        Convert to pandas DataFrame.

        Returns:
            DataFrame with 'rt' and 'intensity' columns
        """
        import pandas as pd
        return pd.DataFrame({
            "rt": self._rt,
            "intensity": self._intensity,
        })

    @classmethod
    def from_dataframe(cls, df, chrom_type: ChromatogramType = ChromatogramType.UNKNOWN) -> "Chromatogram":
        """
        Create Chromatogram from pandas DataFrame.

        Args:
            df: DataFrame with 'rt' and 'intensity' columns
            chrom_type: Chromatogram type

        Returns:
            New Chromatogram
        """
        return cls(
            rt=df["rt"].values,
            intensity=df["intensity"].values,
            chrom_type=chrom_type,
        )

    def __repr__(self) -> str:
        return (
            f"Chromatogram(size={self.size}, type={self.chrom_type.name}, "
            f"rt_range=({self._rt_min:.1f}, {self._rt_max:.1f})s)"
        )
