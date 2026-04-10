"""Tests for Chromatogram data structure."""

import pytest
import numpy as np

from masskit.chromatogram import Chromatogram, ChromatogramType


class TestChromatogramCreation:
    def test_empty(self):
        c = Chromatogram()
        assert len(c) == 0
        assert not bool(c)

    def test_with_data(self):
        c = Chromatogram(
            rt=[10.0, 20.0, 30.0, 40.0],
            intensity=[100.0, 500.0, 300.0, 50.0],
        )
        assert len(c) == 4
        assert bool(c)
        assert c.size == 4

    def test_type(self):
        c = Chromatogram(chrom_type=ChromatogramType.TIC)
        assert c.chrom_type == ChromatogramType.TIC

    def test_enum_values(self):
        assert ChromatogramType.TIC is not None
        assert ChromatogramType.BPC is not None
        assert ChromatogramType.XIC is not None
        assert ChromatogramType.SRM is not None
        assert ChromatogramType.UNKNOWN is not None


class TestChromatogramProperties:
    def _make_chrom(self):
        return Chromatogram(
            rt=[10.0, 20.0, 30.0, 40.0, 50.0],
            intensity=[100.0, 500.0, 800.0, 300.0, 50.0],
        )

    def test_max_intensity(self):
        c = self._make_chrom()
        assert c.max_intensity == 800.0

    def test_apex_rt(self):
        c = self._make_chrom()
        assert c.apex_rt == 30.0

    def test_rt_range(self):
        c = self._make_chrom()
        assert c.rt_range == (10.0, 50.0)

    def test_setters(self):
        c = Chromatogram()
        c.rt = [1.0, 2.0, 3.0]
        c.intensity = [10.0, 20.0, 30.0]
        assert len(c) == 3

    def test_metadata(self):
        c = Chromatogram()
        c.native_id = "TIC"
        c.index = 5
        assert c.native_id == "TIC"
        assert c.index == 5


class TestChromatogramMethods:
    def _make_chrom(self):
        return Chromatogram(
            rt=[10.0, 20.0, 30.0, 40.0, 50.0],
            intensity=[100.0, 500.0, 800.0, 300.0, 50.0],
        )

    def test_is_sorted(self):
        c = self._make_chrom()
        assert c.is_sorted()

    def test_sort_unsorted(self):
        c = Chromatogram(rt=[30.0, 10.0, 20.0], intensity=[3.0, 1.0, 2.0])
        c.sort_by_rt()
        assert c.is_sorted()

    def test_find_nearest_rt(self):
        c = self._make_chrom()
        idx = c.find_nearest_rt(25.0)
        assert c.rt[idx] == 20.0 or c.rt[idx] == 30.0

    def test_extract_range(self):
        c = self._make_chrom()
        sub = c.extract_range(15.0, 35.0)
        assert all(15.0 <= rt <= 35.0 for rt in sub.rt)

    def test_compute_area(self):
        c = Chromatogram(
            rt=[0.0, 1.0, 2.0],
            intensity=[0.0, 1.0, 0.0],
        )
        area = c.compute_area()
        assert abs(area - 1.0) < 0.01  # Triangle area = 1

    def test_interpolate_at(self):
        c = Chromatogram(rt=[0.0, 10.0], intensity=[0.0, 100.0])
        val = c.interpolate_at(5.0)
        assert abs(val - 50.0) < 1.0

    def test_resample(self):
        c = self._make_chrom()
        resampled = c.resample(10)
        assert len(resampled) == 10

    def test_smooth(self):
        c = self._make_chrom()
        smoothed = c.smooth(window_size=3)
        assert len(smoothed) == len(c)

    def test_normalize_max(self):
        c = self._make_chrom()
        n = c.normalize("max")
        assert abs(n.max_intensity - 1.0) < 1e-9

    def test_normalize_sum(self):
        c = self._make_chrom()
        n = c.normalize("sum")
        assert abs(sum(n.intensity) - 1.0) < 1e-6

    def test_copy(self):
        c = self._make_chrom()
        cp = c.copy()
        assert len(cp) == len(c)
        # Modify the copy and verify the original is unchanged
        new_rt = list(cp.rt)
        new_rt[0] = 999.0
        cp.rt = new_rt
        assert c.rt[0] != 999.0  # Original unchanged

    def test_srm_fields(self):
        c = Chromatogram(chrom_type=ChromatogramType.SRM)
        c.precursor_mz = 500.0
        c.product_mz = 300.0
        c.target_mz = 500.0
        assert c.precursor_mz == 500.0
        assert c.product_mz == 300.0
