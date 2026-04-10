"""
Feature and FeatureMap data structures for LC-MS features.
"""

from typing import Optional, List, Dict, Tuple, Iterator
import numpy as np
from dataclasses import dataclass, field
from .peak import Peak, PeakList


@dataclass
class Feature:
    """
    Represents a 2D feature in LC-MS data.

    A Feature represents a molecular entity detected across multiple spectra,
    with a defined m/z and RT extent, intensity profile, and associated peaks.

    Attributes:
        mz: Centroid m/z position
        rt: Apex retention time
        intensity: Apex intensity
        volume: Integrated 2D volume
        charge: Charge state (0 if unknown)
        quality: Quality score (0-1)
    """
    id: int = 0
    mz: float = 0.0
    rt: float = 0.0
    intensity: float = 0.0
    volume: float = 0.0

    # Boundaries
    mz_min: float = 0.0
    mz_max: float = 0.0
    rt_min: float = 0.0
    rt_max: float = 0.0

    # Charge and isotopes
    charge: int = 0
    monoisotopic_mz: float = 0.0
    isotope_count: int = 0

    # Quality metrics
    quality: float = 0.0
    snr: float = 0.0
    isotope_score: float = 0.0

    # Constituent peaks
    peaks: PeakList = field(default_factory=PeakList)

    # Metadata
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def mz_width(self) -> float:
        """Get m/z width."""
        return self.mz_max - self.mz_min

    @property
    def rt_width(self) -> float:
        """Get RT width in seconds."""
        return self.rt_max - self.rt_min

    def neutral_mass(self) -> float:
        """Calculate neutral mass from m/z and charge."""
        if self.charge == 0:
            return 0.0
        proton_mass = 1.007276
        mz = self.monoisotopic_mz if self.monoisotopic_mz > 0 else self.mz
        return (mz - proton_mass) * abs(self.charge)

    def contains(self, mz: float, rt: float) -> bool:
        """Check if point is within feature boundaries."""
        return (self.mz_min <= mz <= self.mz_max and
                self.rt_min <= rt <= self.rt_max)

    def overlaps(self, other: "Feature") -> bool:
        """Check if this feature overlaps with another."""
        mz_overlap = (self.mz_min <= other.mz_max and
                      self.mz_max >= other.mz_min)
        rt_overlap = (self.rt_min <= other.rt_max and
                      self.rt_max >= other.rt_min)
        return mz_overlap and rt_overlap


class FeatureMap:
    """
    Collection of features with spatial indexing.

    Example:
        >>> fmap = FeatureMap()
        >>> fmap.add(Feature(mz=500.0, rt=120.0, intensity=10000))
        >>> fmap.find_in_mz_range(400, 600)
        [Feature(mz=500.0, ...)]
    """

    def __init__(self, features: Optional[List[Feature]] = None):
        """
        Create a new FeatureMap.

        Args:
            features: Optional list of features
        """
        self._features: List[Feature] = []
        self._next_id = 0
        self.source_file: str = ""
        self.metadata: Dict[str, str] = {}

        if features:
            for f in features:
                self.add(f)

    def __len__(self) -> int:
        return len(self._features)

    def __bool__(self) -> bool:
        return len(self._features) > 0

    def __getitem__(self, idx: int) -> Feature:
        return self._features[idx]

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._features)

    def add(self, feature: Feature) -> None:
        """Add a feature to the map."""
        feature.id = self._next_id
        self._next_id += 1
        self._features.append(feature)

    def extend(self, features: List[Feature]) -> None:
        """Add multiple features."""
        for f in features:
            self.add(f)

    def clear(self) -> None:
        """Remove all features."""
        self._features.clear()
        self._next_id = 0

    @property
    def size(self) -> int:
        """Get number of features."""
        return len(self._features)

    @property
    def mz_array(self) -> np.ndarray:
        """Get m/z values as numpy array."""
        return np.array([f.mz for f in self._features], dtype=np.float64)

    @property
    def rt_array(self) -> np.ndarray:
        """Get RT values as numpy array."""
        return np.array([f.rt for f in self._features], dtype=np.float64)

    @property
    def intensity_array(self) -> np.ndarray:
        """Get intensity values as numpy array."""
        return np.array([f.intensity for f in self._features], dtype=np.float64)

    @property
    def volume_array(self) -> np.ndarray:
        """Get volume values as numpy array."""
        return np.array([f.volume for f in self._features], dtype=np.float64)

    @property
    def charge_array(self) -> np.ndarray:
        """Get charge states as numpy array."""
        return np.array([f.charge for f in self._features], dtype=np.int32)

    @property
    def mz_range(self) -> Tuple[float, float]:
        """Get overall m/z range."""
        if not self._features:
            return (0.0, 0.0)
        mz_vals = self.mz_array
        return (float(np.min(mz_vals)), float(np.max(mz_vals)))

    @property
    def rt_range(self) -> Tuple[float, float]:
        """Get overall RT range."""
        if not self._features:
            return (0.0, 0.0)
        rt_vals = self.rt_array
        return (float(np.min(rt_vals)), float(np.max(rt_vals)))

    def sort_by_mz(self) -> "FeatureMap":
        """Sort by m/z (in-place) and return self."""
        self._features.sort(key=lambda f: f.mz)
        return self

    def sort_by_rt(self) -> "FeatureMap":
        """Sort by RT (in-place) and return self."""
        self._features.sort(key=lambda f: f.rt)
        return self

    def sort_by_intensity(self, descending: bool = True) -> "FeatureMap":
        """Sort by intensity (in-place) and return self."""
        self._features.sort(key=lambda f: f.intensity, reverse=descending)
        return self

    def sort_by_quality(self, descending: bool = True) -> "FeatureMap":
        """Sort by quality (in-place) and return self."""
        self._features.sort(key=lambda f: f.quality, reverse=descending)
        return self

    def find_in_mz_range(self, mz_min: float, mz_max: float) -> List[Feature]:
        """Find features within m/z range."""
        return [f for f in self._features if mz_min <= f.mz <= mz_max]

    def find_in_rt_range(self, rt_min: float, rt_max: float) -> List[Feature]:
        """Find features within RT range."""
        return [f for f in self._features if rt_min <= f.rt <= rt_max]

    def find_in_range(
        self,
        mz_min: float,
        mz_max: float,
        rt_min: float,
        rt_max: float,
    ) -> List[Feature]:
        """Find features within both m/z and RT range."""
        return [
            f for f in self._features
            if mz_min <= f.mz <= mz_max and rt_min <= f.rt <= rt_max
        ]

    def find_nearest(self, mz: float, rt: float) -> Optional[Feature]:
        """
        Find feature closest to given m/z and RT.

        Uses normalized Euclidean distance.
        """
        if not self._features:
            return None

        mz_range = self.mz_range
        rt_range = self.rt_range
        mz_span = mz_range[1] - mz_range[0] + 1e-10
        rt_span = rt_range[1] - rt_range[0] + 1e-10

        def distance(f: Feature) -> float:
            mz_dist = (f.mz - mz) / mz_span
            rt_dist = (f.rt - rt) / rt_span
            return np.sqrt(mz_dist**2 + rt_dist**2)

        return min(self._features, key=distance)

    def filter_by_charge(self, charges: List[int]) -> "FeatureMap":
        """Filter features by charge state."""
        return FeatureMap([f for f in self._features if f.charge in charges])

    def filter_by_quality(self, min_quality: float) -> "FeatureMap":
        """Filter features by minimum quality."""
        return FeatureMap([f for f in self._features if f.quality >= min_quality])

    def filter_by_intensity(self, min_intensity: float) -> "FeatureMap":
        """Filter features by minimum intensity."""
        return FeatureMap([f for f in self._features if f.intensity >= min_intensity])

    def top_n(self, n: int, by: str = "intensity") -> "FeatureMap":
        """
        Get top N features.

        Args:
            n: Number of features
            by: Sort key ("intensity", "volume", "quality")
        """
        if by == "intensity":
            key = lambda f: f.intensity
        elif by == "volume":
            key = lambda f: f.volume
        elif by == "quality":
            key = lambda f: f.quality
        else:
            raise ValueError(f"Unknown sort key: {by}")

        sorted_features = sorted(self._features, key=key, reverse=True)
        return FeatureMap(sorted_features[:n])

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([
            {
                "id": f.id,
                "mz": f.mz,
                "rt": f.rt,
                "intensity": f.intensity,
                "volume": f.volume,
                "charge": f.charge,
                "quality": f.quality,
                "snr": f.snr,
                "mz_width": f.mz_width,
                "rt_width": f.rt_width,
                "isotope_count": f.isotope_count,
            }
            for f in self._features
        ])

    @classmethod
    def from_dataframe(cls, df) -> "FeatureMap":
        """Create FeatureMap from pandas DataFrame."""
        fmap = cls()
        for _, row in df.iterrows():
            feature = Feature(
                mz=row.get("mz", 0.0),
                rt=row.get("rt", 0.0),
                intensity=row.get("intensity", 0.0),
                volume=row.get("volume", 0.0),
                charge=int(row.get("charge", 0)),
                quality=row.get("quality", 0.0),
            )
            fmap.add(feature)
        return fmap

    def copy(self) -> "FeatureMap":
        """Create a deep copy."""
        fmap = FeatureMap()
        for f in self._features:
            new_f = Feature(
                mz=f.mz,
                rt=f.rt,
                intensity=f.intensity,
                volume=f.volume,
                mz_min=f.mz_min,
                mz_max=f.mz_max,
                rt_min=f.rt_min,
                rt_max=f.rt_max,
                charge=f.charge,
                quality=f.quality,
                snr=f.snr,
            )
            fmap.add(new_f)
        fmap.source_file = self.source_file
        fmap.metadata = self.metadata.copy()
        return fmap

    def __repr__(self) -> str:
        mz_range = self.mz_range
        rt_range = self.rt_range
        return (
            f"FeatureMap(n={len(self._features)}, "
            f"mz_range=({mz_range[0]:.2f}, {mz_range[1]:.2f}), "
            f"rt_range=({rt_range[0]:.1f}, {rt_range[1]:.1f})s)"
        )
