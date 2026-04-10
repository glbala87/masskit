"""
MSExperiment container for LC-MS data.
"""

from typing import Optional, List, Dict, Tuple, Iterator, Union
import numpy as np
from .spectrum import Spectrum, Polarity
from .chromatogram import Chromatogram, ChromatogramType


class MSExperiment:
    """
    Container for a complete LC-MS experiment.

    Holds all spectra and chromatograms from an LC-MS run, along with
    instrument metadata and run information.

    Example:
        >>> exp = MSExperiment()
        >>> exp.add_spectrum(Spectrum(mz=[100, 200], intensity=[1000, 2000], rt=60.0))
        >>> exp.spectrum_count
        1
        >>> tic = exp.generate_tic()
    """

    def __init__(self):
        """Create a new MSExperiment."""
        self._spectra: List[Spectrum] = []
        self._chromatograms: List[Chromatogram] = []

        # Metadata
        self.source_file: str = ""
        self.date_time: str = ""
        self.instrument_model: str = ""
        self.instrument_serial: str = ""
        self.software: str = ""
        self.metadata: Dict[str, str] = {}

    # =========================================================================
    # Spectra Access
    # =========================================================================

    @property
    def spectrum_count(self) -> int:
        """Get number of spectra."""
        return len(self._spectra)

    @property
    def num_spectra(self) -> int:
        """Alias for spectrum_count (for backwards compatibility)."""
        return len(self._spectra)

    @property
    def has_spectra(self) -> bool:
        """Check if experiment has spectra."""
        return len(self._spectra) > 0

    def spectrum(self, index: int) -> Spectrum:
        """Get spectrum by index."""
        return self._spectra[index]

    def __getitem__(self, index: int) -> Spectrum:
        """Get spectrum by index."""
        return self._spectra[index]

    @property
    def spectra(self) -> List[Spectrum]:
        """Get all spectra."""
        return self._spectra

    def add_spectrum(self, spec: Spectrum) -> None:
        """Add a spectrum."""
        spec.index = len(self._spectra)
        self._spectra.append(spec)

    def reserve_spectra(self, n: int) -> None:
        """Pre-allocate space for spectra (no-op in Python, for API compat)."""
        pass

    def iter_spectra(self) -> Iterator[Spectrum]:
        """Iterate over spectra."""
        return iter(self._spectra)

    # =========================================================================
    # Chromatograms Access
    # =========================================================================

    @property
    def chromatogram_count(self) -> int:
        """Get number of chromatograms."""
        return len(self._chromatograms)

    @property
    def has_chromatograms(self) -> bool:
        """Check if experiment has chromatograms."""
        return len(self._chromatograms) > 0

    def chromatogram(self, index: int) -> Chromatogram:
        """Get chromatogram by index."""
        return self._chromatograms[index]

    @property
    def chromatograms(self) -> List[Chromatogram]:
        """Get all chromatograms."""
        return self._chromatograms

    def add_chromatogram(self, chrom: Chromatogram) -> None:
        """Add a chromatogram."""
        chrom.index = len(self._chromatograms)
        self._chromatograms.append(chrom)

    def iter_chromatograms(self) -> Iterator[Chromatogram]:
        """Iterate over chromatograms."""
        return iter(self._chromatograms)

    # =========================================================================
    # Filtering and Selection
    # =========================================================================

    def get_spectra_by_level(self, level: int) -> List[Spectrum]:
        """Get spectra at given MS level."""
        return [s for s in self._spectra if s.ms_level == level]

    def count_spectra_by_level(self, level: int) -> int:
        """Count spectra at given MS level."""
        return sum(1 for s in self._spectra if s.ms_level == level)

    def get_spectra_in_rt_range(
        self, rt_min: float, rt_max: float
    ) -> List[Spectrum]:
        """Get spectra within RT range."""
        return [s for s in self._spectra if rt_min <= s.rt <= rt_max]

    def find_spectrum_by_native_id(self, native_id: str) -> Optional[Spectrum]:
        """Find spectrum by native ID."""
        for s in self._spectra:
            if s.native_id == native_id:
                return s
        return None

    def find_spectrum_by_rt(
        self, rt: float, level: int = 0
    ) -> Optional[Spectrum]:
        """Find spectrum closest to given RT."""
        candidates = self._spectra if level == 0 else self.get_spectra_by_level(level)
        if not candidates:
            return None
        return min(candidates, key=lambda s: abs(s.rt - rt))

    def get_tic_chromatogram(self) -> Optional[Chromatogram]:
        """Get TIC chromatogram if present."""
        for c in self._chromatograms:
            if c.chrom_type == ChromatogramType.TIC:
                return c
        return None

    # =========================================================================
    # Generate Chromatograms
    # =========================================================================

    def generate_tic(self, level: int = 1) -> Chromatogram:
        """
        Generate TIC from spectra.

        Args:
            level: MS level (0 for all levels)

        Returns:
            TIC chromatogram
        """
        spectra = (self._spectra if level == 0
                   else self.get_spectra_by_level(level))

        rt = np.array([s.rt for s in spectra], dtype=np.float64)
        intensity = np.array([s.tic for s in spectra], dtype=np.float64)

        # Sort by RT
        indices = np.argsort(rt)
        rt = rt[indices]
        intensity = intensity[indices]

        chrom = Chromatogram(rt=rt, intensity=intensity, chrom_type=ChromatogramType.TIC)
        chrom.native_id = "TIC"
        return chrom

    def generate_bpc(self, level: int = 1) -> Chromatogram:
        """
        Generate base peak chromatogram from spectra.

        Args:
            level: MS level (0 for all levels)

        Returns:
            BPC chromatogram
        """
        spectra = (self._spectra if level == 0
                   else self.get_spectra_by_level(level))

        rt = np.array([s.rt for s in spectra], dtype=np.float64)
        intensity = np.array([s.base_peak_intensity for s in spectra], dtype=np.float64)

        indices = np.argsort(rt)
        rt = rt[indices]
        intensity = intensity[indices]

        chrom = Chromatogram(rt=rt, intensity=intensity, chrom_type=ChromatogramType.BPC)
        chrom.native_id = "BPC"
        return chrom

    def generate_xic(
        self,
        target_mz: float,
        tolerance: float,
        ppm: bool = False,
        level: int = 1,
    ) -> Chromatogram:
        """
        Generate extracted ion chromatogram (XIC).

        Args:
            target_mz: Target m/z
            tolerance: m/z tolerance (Da or ppm)
            ppm: If True, tolerance is in ppm
            level: MS level

        Returns:
            XIC chromatogram
        """
        spectra = (self._spectra if level == 0
                   else self.get_spectra_by_level(level))

        rt_list = []
        intensity_list = []

        for spec in spectra:
            if ppm:
                tol = target_mz * tolerance * 1e-6
            else:
                tol = tolerance

            mz_min = target_mz - tol
            mz_max = target_mz + tol

            # Sum intensities in range
            mask = (spec.mz >= mz_min) & (spec.mz <= mz_max)
            total_intensity = float(np.sum(spec.intensity[mask]))

            rt_list.append(spec.rt)
            intensity_list.append(total_intensity)

        rt = np.array(rt_list, dtype=np.float64)
        intensity = np.array(intensity_list, dtype=np.float64)

        indices = np.argsort(rt)
        rt = rt[indices]
        intensity = intensity[indices]

        chrom = Chromatogram(
            rt=rt,
            intensity=intensity,
            chrom_type=ChromatogramType.XIC,
            target_mz=target_mz,
            mz_tolerance=tolerance if not ppm else target_mz * tolerance * 1e-6,
        )
        return chrom

    # =========================================================================
    # Ranges
    # =========================================================================

    @property
    def mz_range(self) -> Tuple[float, float]:
        """Get overall m/z range."""
        if not self._spectra:
            return (0.0, 0.0)
        mz_min = min(s.mz_range[0] for s in self._spectra if s.size > 0)
        mz_max = max(s.mz_range[1] for s in self._spectra if s.size > 0)
        return (mz_min, mz_max)

    @property
    def rt_range(self) -> Tuple[float, float]:
        """Get overall RT range."""
        if not self._spectra:
            return (0.0, 0.0)
        rt_vals = [s.rt for s in self._spectra]
        return (min(rt_vals), max(rt_vals))

    # =========================================================================
    # Sorting
    # =========================================================================

    def sort_spectra_by_rt(self) -> None:
        """Sort spectra by retention time."""
        self._spectra.sort(key=lambda s: s.rt)
        for i, s in enumerate(self._spectra):
            s.index = i

    def sort_spectra_by_level_and_rt(self) -> None:
        """Sort spectra by MS level, then RT."""
        self._spectra.sort(key=lambda s: (s.ms_level, s.rt))
        for i, s in enumerate(self._spectra):
            s.index = i

    # =========================================================================
    # Clear/Reset
    # =========================================================================

    def clear(self) -> None:
        """Clear all data."""
        self._spectra.clear()
        self._chromatograms.clear()
        self.metadata.clear()

    def clear_spectra(self) -> None:
        """Clear only spectra."""
        self._spectra.clear()

    def clear_chromatograms(self) -> None:
        """Clear only chromatograms."""
        self._chromatograms.clear()

    # =========================================================================
    # Statistics
    # =========================================================================

    @property
    def total_data_points(self) -> int:
        """Get total number of data points across all spectra."""
        return sum(s.size for s in self._spectra)

    @property
    def average_spectrum_size(self) -> float:
        """Get average spectrum size."""
        if not self._spectra:
            return 0.0
        return self.total_data_points / len(self._spectra)

    def summary(self) -> Dict[str, Union[int, float, str]]:
        """Get summary statistics."""
        ms_levels = {}
        for s in self._spectra:
            ms_levels[s.ms_level] = ms_levels.get(s.ms_level, 0) + 1

        return {
            "spectrum_count": len(self._spectra),
            "chromatogram_count": len(self._chromatograms),
            "total_data_points": self.total_data_points,
            "average_spectrum_size": self.average_spectrum_size,
            "mz_range": self.mz_range,
            "rt_range": self.rt_range,
            "ms_levels": ms_levels,
            "source_file": self.source_file,
        }

    def __repr__(self) -> str:
        return (
            f"MSExperiment(spectra={len(self._spectra)}, "
            f"chromatograms={len(self._chromatograms)}, "
            f"source={self.source_file!r})"
        )
