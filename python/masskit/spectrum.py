"""
Spectrum data structure for mass spectrometry data.
"""

from enum import Enum
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
from dataclasses import dataclass, field


class SpectrumType(Enum):
    """Type of spectrum data."""
    UNKNOWN = 0
    PROFILE = 1
    CENTROID = 2


class Polarity(Enum):
    """Ion polarity mode."""
    UNKNOWN = 0
    POSITIVE = 1
    NEGATIVE = -1


@dataclass
class Precursor:
    """Precursor ion information for MS/MS spectra."""
    mz: float = 0.0
    intensity: float = 0.0
    charge: int = 0
    isolation_window_lower: float = 0.0
    isolation_window_upper: float = 0.0
    activation_method: str = ""
    collision_energy: float = 0.0

    @property
    def isolation_window_width(self) -> float:
        """Get isolation window width."""
        return self.isolation_window_upper - self.isolation_window_lower

    def neutral_mass(self) -> float:
        """Calculate neutral mass from m/z and charge."""
        if self.charge == 0:
            return 0.0
        proton_mass = 1.007276
        return (self.mz - proton_mass) * abs(self.charge)


class Spectrum:
    """
    Mass spectrum with m/z and intensity arrays.

    A Spectrum stores paired arrays of m/z values and corresponding intensities,
    along with metadata about the spectrum acquisition.

    Attributes:
        mz: m/z values as numpy array
        intensity: Intensity values as numpy array
        ms_level: MS level (1 for MS1, 2 for MS/MS, etc.)
        rt: Retention time in seconds
        spectrum_type: Profile or centroid
        polarity: Positive or negative ion mode
        precursors: List of precursor ions (for MS/MS)

    Example:
        >>> spec = Spectrum(mz=[100.0, 200.0, 300.0], intensity=[1000, 5000, 2000])
        >>> spec.base_peak_mz
        200.0
        >>> spec.tic
        8000.0
    """

    def __init__(
        self,
        mz: Optional[Union[np.ndarray, List[float]]] = None,
        intensity: Optional[Union[np.ndarray, List[float]]] = None,
        ms_level: int = 1,
        rt: float = 0.0,
        spectrum_type: SpectrumType = SpectrumType.UNKNOWN,
        polarity: Polarity = Polarity.UNKNOWN,
    ):
        """
        Create a new Spectrum.

        Args:
            mz: m/z values
            intensity: Intensity values
            ms_level: MS level (1 = MS1, 2 = MS/MS, etc.)
            rt: Retention time in seconds
            spectrum_type: Profile or centroid
            polarity: Ion polarity mode
        """
        if mz is not None:
            self._mz = np.asarray(mz, dtype=np.float64)
        else:
            self._mz = np.array([], dtype=np.float64)

        if intensity is not None:
            self._intensity = np.asarray(intensity, dtype=np.float64)
        else:
            self._intensity = np.array([], dtype=np.float64)

        # Use the validation helper if it can be imported (avoids circular at import time)
        try:
            from .validation import validate_mz_intensity
            validate_mz_intensity(self._mz, self._intensity)
        except ImportError:
            if len(self._mz) != len(self._intensity):
                raise ValueError("m/z and intensity arrays must have same length")

        self.ms_level = ms_level
        self.rt = rt
        self.spectrum_type = spectrum_type
        self.polarity = polarity
        self.index: int = 0
        self.native_id: str = ""
        self.precursors: List[Precursor] = []
        self.metadata: Dict[str, str] = {}

        self._update_cache()

    def _update_cache(self) -> None:
        """Update cached statistics."""
        # Defensive: arrays may temporarily mismatch when setters are called
        # independently. Fall back to safe defaults rather than crashing.
        if len(self._intensity) > 0 and len(self._intensity) == len(self._mz):
            self._tic = float(np.sum(self._intensity))
            max_idx = int(np.argmax(self._intensity))
            self._base_peak_intensity = float(self._intensity[max_idx])
            self._base_peak_mz = float(self._mz[max_idx])
            self._mz_min = float(np.min(self._mz))
            self._mz_max = float(np.max(self._mz))
        elif len(self._mz) > 0:
            # m/z set but intensity not yet matching — provide partial info
            self._tic = float(np.sum(self._intensity)) if len(self._intensity) > 0 else 0.0
            self._base_peak_intensity = 0.0
            self._base_peak_mz = 0.0
            self._mz_min = float(np.min(self._mz))
            self._mz_max = float(np.max(self._mz))
        else:
            self._tic = 0.0
            self._base_peak_intensity = 0.0
            self._base_peak_mz = 0.0
            self._mz_min = 0.0
            self._mz_max = 0.0

    @property
    def mz(self) -> np.ndarray:
        """Get m/z array."""
        return self._mz

    @mz.setter
    def mz(self, value: Union[np.ndarray, List[float]]) -> None:
        """Set m/z array."""
        self._mz = np.asarray(value, dtype=np.float64)
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
        return len(self._mz)

    def __bool__(self) -> bool:
        """Check if spectrum has data."""
        return len(self._mz) > 0

    @property
    def size(self) -> int:
        """Get number of data points."""
        return len(self._mz)

    @property
    def tic(self) -> float:
        """Get total ion current."""
        return self._tic

    @property
    def base_peak_intensity(self) -> float:
        """Get base peak intensity."""
        return self._base_peak_intensity

    @property
    def base_peak_mz(self) -> float:
        """Get base peak m/z."""
        return self._base_peak_mz

    @property
    def mz_range(self) -> Tuple[float, float]:
        """Get (min, max) m/z range."""
        return (self._mz_min, self._mz_max)

    def is_sorted(self) -> bool:
        """Check if m/z array is sorted."""
        return np.all(self._mz[:-1] <= self._mz[1:]) if len(self._mz) > 1 else True

    def sort_by_mz(self) -> "Spectrum":
        """Sort by m/z (in-place) and return self."""
        if len(self._mz) > 1:
            indices = np.argsort(self._mz)
            self._mz = self._mz[indices]
            self._intensity = self._intensity[indices]
        return self

    def find_nearest_mz(self, target: float) -> int:
        """
        Find index of peak closest to target m/z.

        Args:
            target: Target m/z value

        Returns:
            Index of nearest peak
        """
        if len(self._mz) == 0:
            raise ValueError("Cannot find peak in empty spectrum")
        return int(np.argmin(np.abs(self._mz - target)))

    def extract_range(self, mz_min: float, mz_max: float) -> "Spectrum":
        """
        Extract peaks within m/z range.

        Args:
            mz_min: Minimum m/z
            mz_max: Maximum m/z

        Returns:
            New Spectrum with only peaks in range
        """
        mask = (self._mz >= mz_min) & (self._mz <= mz_max)
        return Spectrum(
            mz=self._mz[mask].copy(),
            intensity=self._intensity[mask].copy(),
            ms_level=self.ms_level,
            rt=self.rt,
            spectrum_type=self.spectrum_type,
            polarity=self.polarity,
        )

    def filter_by_intensity(self, min_intensity: float) -> "Spectrum":
        """
        Filter peaks by minimum intensity.

        Args:
            min_intensity: Minimum intensity threshold

        Returns:
            New Spectrum with only peaks above threshold
        """
        mask = self._intensity >= min_intensity
        return Spectrum(
            mz=self._mz[mask].copy(),
            intensity=self._intensity[mask].copy(),
            ms_level=self.ms_level,
            rt=self.rt,
            spectrum_type=self.spectrum_type,
            polarity=self.polarity,
        )

    def top_n(self, n: int) -> "Spectrum":
        """
        Get top N peaks by intensity.

        Args:
            n: Number of peaks to keep

        Returns:
            New Spectrum with top N peaks
        """
        if n >= len(self._mz):
            return self.copy()
        indices = np.argsort(self._intensity)[-n:]
        indices = np.sort(indices)  # Keep m/z order
        return Spectrum(
            mz=self._mz[indices].copy(),
            intensity=self._intensity[indices].copy(),
            ms_level=self.ms_level,
            rt=self.rt,
            spectrum_type=self.spectrum_type,
            polarity=self.polarity,
        )

    def normalize(self, method: str = "max") -> "Spectrum":
        """
        Normalize intensities.

        Args:
            method: Normalization method ("max", "sum", "rms")

        Returns:
            New Spectrum with normalized intensities
        """
        intensity = self._intensity.copy()
        if len(intensity) == 0:
            return self.copy()

        if method == "max":
            factor = np.max(intensity)
        elif method == "sum":
            factor = np.sum(intensity)
        elif method == "rms":
            factor = np.sqrt(np.mean(intensity ** 2))
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if factor > 0:
            intensity /= factor

        spec = Spectrum(
            mz=self._mz.copy(),
            intensity=intensity,
            ms_level=self.ms_level,
            rt=self.rt,
            spectrum_type=self.spectrum_type,
            polarity=self.polarity,
        )
        spec.precursors = self.precursors.copy()
        spec.metadata = self.metadata.copy()
        return spec

    def copy(self) -> "Spectrum":
        """Create a deep copy."""
        spec = Spectrum(
            mz=self._mz.copy(),
            intensity=self._intensity.copy(),
            ms_level=self.ms_level,
            rt=self.rt,
            spectrum_type=self.spectrum_type,
            polarity=self.polarity,
        )
        spec.index = self.index
        spec.native_id = self.native_id
        spec.precursors = [Precursor(**vars(p)) for p in self.precursors]
        spec.metadata = self.metadata.copy()
        return spec

    def to_dataframe(self):
        """
        Convert to pandas DataFrame.

        Returns:
            DataFrame with 'mz' and 'intensity' columns
        """
        import pandas as pd
        return pd.DataFrame({
            "mz": self._mz,
            "intensity": self._intensity,
        })

    @classmethod
    def from_dataframe(cls, df, ms_level: int = 1, rt: float = 0.0) -> "Spectrum":
        """
        Create Spectrum from pandas DataFrame.

        Args:
            df: DataFrame with 'mz' and 'intensity' columns
            ms_level: MS level
            rt: Retention time

        Returns:
            New Spectrum
        """
        return cls(
            mz=df["mz"].values,
            intensity=df["intensity"].values,
            ms_level=ms_level,
            rt=rt,
        )

    def __repr__(self) -> str:
        return (
            f"Spectrum(size={self.size}, ms_level={self.ms_level}, "
            f"rt={self.rt:.2f}s, mz_range=({self._mz_min:.2f}, {self._mz_max:.2f}))"
        )
