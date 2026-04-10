"""Tests for Feature and FeatureMap data structures."""

import pytest
import numpy as np

from masskit.feature import Feature, FeatureMap
from masskit.peak import Peak, PeakList


class TestFeature:
    def test_defaults(self):
        f = Feature()
        assert f.mz == 0.0
        assert f.rt == 0.0
        assert f.charge == 0

    def test_mz_width(self):
        f = Feature()
        f.mz_min = 499.99
        f.mz_max = 500.01
        assert abs(f.mz_width - 0.02) < 1e-9

    def test_rt_width(self):
        f = Feature()
        f.rt_min = 50.0
        f.rt_max = 80.0
        assert abs(f.rt_width - 30.0) < 1e-9

    def test_neutral_mass_with_charge(self):
        f = Feature()
        f.mz = 500.0
        f.charge = 2
        nm = f.neutral_mass()
        # neutral_mass = (mz - proton) * charge
        assert nm > 990

    def test_neutral_mass_zero_charge(self):
        f = Feature()
        f.mz = 500.0
        f.charge = 0
        assert f.neutral_mass() == 0.0

    def test_contains(self):
        f = Feature()
        f.mz_min = 499.0
        f.mz_max = 501.0
        f.rt_min = 50.0
        f.rt_max = 80.0
        assert f.contains(500.0, 60.0)
        assert not f.contains(600.0, 60.0)
        assert not f.contains(500.0, 90.0)

    def test_overlaps(self):
        f1 = Feature()
        f1.mz_min, f1.mz_max = 499.0, 501.0
        f1.rt_min, f1.rt_max = 50.0, 80.0

        f2 = Feature()
        f2.mz_min, f2.mz_max = 500.0, 502.0
        f2.rt_min, f2.rt_max = 70.0, 100.0
        assert f1.overlaps(f2)

    def test_no_overlap(self):
        f1 = Feature()
        f1.mz_min, f1.mz_max = 499.0, 501.0
        f1.rt_min, f1.rt_max = 50.0, 60.0

        f2 = Feature()
        f2.mz_min, f2.mz_max = 600.0, 602.0
        f2.rt_min, f2.rt_max = 70.0, 100.0
        assert not f1.overlaps(f2)


class TestFeatureMap:
    def _make_fmap(self, n=5):
        fmap = FeatureMap()
        for i in range(n):
            f = Feature()
            f.mz = 100.0 + i * 50
            f.rt = 60.0 + i * 10
            f.intensity = 1000.0 + i * 100
            f.volume = 5000.0 + i * 200
            f.quality = 0.5 + i * 0.1
            f.charge = 2 if i % 2 == 0 else 1
            f.mz_min = f.mz - 0.01
            f.mz_max = f.mz + 0.01
            f.rt_min = f.rt - 5
            f.rt_max = f.rt + 5
            fmap.add(f)
        return fmap

    def test_empty(self):
        fmap = FeatureMap()
        assert len(fmap) == 0
        assert not bool(fmap)

    def test_add_and_len(self):
        fmap = self._make_fmap()
        assert len(fmap) == 5
        assert bool(fmap)

    def test_getitem(self):
        fmap = self._make_fmap()
        assert fmap[0].mz == 100.0

    def test_iter(self):
        fmap = self._make_fmap()
        features = list(fmap)
        assert len(features) == 5

    def test_extend(self):
        fmap = self._make_fmap(3)
        fmap2 = self._make_fmap(2)
        fmap.extend(list(fmap2))
        assert len(fmap) == 5

    def test_clear(self):
        fmap = self._make_fmap()
        fmap.clear()
        assert len(fmap) == 0

    def test_mz_array(self):
        fmap = self._make_fmap()
        arr = fmap.mz_array
        assert len(arr) == 5
        assert arr[0] == 100.0

    def test_rt_array(self):
        fmap = self._make_fmap()
        assert len(fmap.rt_array) == 5

    def test_intensity_array(self):
        fmap = self._make_fmap()
        assert len(fmap.intensity_array) == 5

    def test_mz_range(self):
        fmap = self._make_fmap()
        mn, mx = fmap.mz_range
        assert mn == 100.0
        assert mx == 300.0

    def test_rt_range(self):
        fmap = self._make_fmap()
        mn, mx = fmap.rt_range
        assert mn == 60.0
        assert mx == 100.0

    def test_sort_by_mz(self):
        fmap = self._make_fmap()
        fmap.sort_by_mz()
        mzs = [f.mz for f in fmap]
        assert mzs == sorted(mzs)

    def test_sort_by_intensity(self):
        fmap = self._make_fmap()
        fmap.sort_by_intensity(descending=True)
        ints = [f.intensity for f in fmap]
        assert ints == sorted(ints, reverse=True)

    def test_find_in_mz_range(self):
        fmap = self._make_fmap()
        found = fmap.find_in_mz_range(140.0, 210.0)
        assert all(140.0 <= f.mz <= 210.0 for f in found)

    def test_find_in_rt_range(self):
        fmap = self._make_fmap()
        found = fmap.find_in_rt_range(65.0, 85.0)
        assert all(65.0 <= f.rt <= 85.0 for f in found)

    def test_find_nearest(self):
        fmap = self._make_fmap()
        nearest = fmap.find_nearest(152.0, 72.0)
        assert nearest is not None
        assert nearest.mz == 150.0

    def test_filter_by_charge(self):
        fmap = self._make_fmap()
        filtered = fmap.filter_by_charge([2])
        assert all(f.charge == 2 for f in filtered)

    def test_filter_by_quality(self):
        fmap = self._make_fmap()
        filtered = fmap.filter_by_quality(0.7)
        assert all(f.quality >= 0.7 for f in filtered)

    def test_filter_by_intensity(self):
        fmap = self._make_fmap()
        filtered = fmap.filter_by_intensity(1200.0)
        assert all(f.intensity >= 1200.0 for f in filtered)

    def test_top_n(self):
        fmap = self._make_fmap()
        top2 = fmap.top_n(2, by="intensity")
        assert len(top2) == 2

    def test_copy(self):
        fmap = self._make_fmap()
        cp = fmap.copy()
        assert len(cp) == len(fmap)
        cp.clear()
        assert len(fmap) == 5  # Original unchanged
