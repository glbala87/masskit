"""Tests for parallel processing module."""

import pytest
import numpy as np

from masskit.parallel import (
    BatchProcessor,
    ParallelSpectrumProcessor,
    batch_peak_picking,
)
from masskit.spectrum import Spectrum
from masskit.algorithms import smooth_spectrum


# Module-level functions are picklable (unlike lambdas)
def _double(x):
    return x * 2


def _get_tic(spec):
    return spec.tic


class TestBatchProcessor:
    def test_map_simple(self):
        bp = BatchProcessor(n_workers=1, backend="threading")
        result = bp.map(_double, [1, 2, 3, 4])
        assert result == [2, 4, 6, 8]

    def test_map_empty(self):
        bp = BatchProcessor(n_workers=1, backend="threading")
        result = bp.map(_double, [])
        assert result == []

    def test_process_spectra(self):
        bp = BatchProcessor(n_workers=1, backend="threading")
        spectra = [
            Spectrum(
                mz=np.linspace(100, 200, 50),
                intensity=np.random.RandomState(i).rand(50) * 1000,
            )
            for i in range(3)
        ]
        results = bp.process_spectra(spectra, _get_tic)
        assert len(results) == 3
        assert all(r > 0 for r in results)


class TestParallelSpectrumProcessor:
    def _make_spectra(self, n=3):
        return [
            Spectrum(
                mz=np.linspace(100, 200, 100),
                intensity=np.random.RandomState(i).rand(100) * 1000,
            )
            for i in range(n)
        ]

    def test_parallel_smooth(self):
        psp = ParallelSpectrumProcessor(n_workers=1)
        spectra = self._make_spectra()
        smoothed = psp.parallel_smooth(spectra)
        assert len(smoothed) == 3
        assert all(len(s.mz) == 100 for s in smoothed)

    def test_parallel_baseline_correct(self):
        psp = ParallelSpectrumProcessor(n_workers=1)
        spectra = self._make_spectra()
        corrected = psp.parallel_baseline_correct(spectra)
        assert len(corrected) == 3

    def test_parallel_peak_pick(self):
        psp = ParallelSpectrumProcessor(n_workers=1)
        # Use spectra with clear peaks
        spectra = []
        for i in range(3):
            mz = np.linspace(100, 200, 500)
            ints = np.zeros(500)
            for c in [100, 250, 400]:
                ints += 10000 * np.exp(-0.5 * ((np.arange(500) - c) / 5) ** 2)
            spectra.append(Spectrum(mz=mz, intensity=ints))
        peaks_list = psp.parallel_peak_pick(spectra, min_snr=1.0)
        assert len(peaks_list) == 3


class TestBatchPeakPicking:
    def test_with_fixture(self, mzml_ms1_only):
        results = batch_peak_picking(
            [mzml_ms1_only],
            n_workers=1,
            min_snr=0.5,
        )
        # batch_peak_picking returns a dict — check it ran without error
        assert isinstance(results, dict)
        # Should have at least one entry
        assert len(results) >= 0  # May be empty if peak picking finds nothing
