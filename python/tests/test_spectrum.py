"""Tests for Spectrum and related data structures."""

import pytest
import numpy as np

from masskit.spectrum import Spectrum, SpectrumType, Polarity
from masskit.peak import Peak, PeakList


class TestSpectrum:
    def test_create_empty(self):
        spec = Spectrum()
        assert len(spec.mz) == 0
        assert len(spec.intensity) == 0

    def test_create_with_data(self):
        mz = np.array([100.0, 200.0, 300.0])
        ints = np.array([1000.0, 2000.0, 500.0])
        spec = Spectrum(mz=mz, intensity=ints)
        assert len(spec.mz) == 3
        np.testing.assert_array_equal(spec.mz, mz)

    def test_mismatched_lengths(self):
        # SpectrumError or ValueError both acceptable (validation hierarchy)
        with pytest.raises((ValueError, Exception), match="same length"):
            Spectrum(mz=[1.0, 2.0], intensity=[1.0])

    def test_spectrum_type(self):
        assert SpectrumType.PROFILE.value == 1
        assert SpectrumType.CENTROID.value == 2
        assert SpectrumType.UNKNOWN.value == 0

    def test_polarity(self):
        assert Polarity.POSITIVE is not None
        assert Polarity.NEGATIVE is not None

    def test_tic(self):
        spec = Spectrum(mz=[100.0, 200.0], intensity=[1000.0, 2000.0])
        assert spec.tic == 3000.0

    def test_base_peak(self):
        spec = Spectrum(mz=[100.0, 200.0, 300.0], intensity=[500.0, 2000.0, 100.0])
        assert spec.base_peak_mz == 200.0
        assert spec.base_peak_intensity == 2000.0

    def test_mz_range(self):
        spec = Spectrum(mz=[100.0, 200.0, 300.0], intensity=[1.0, 1.0, 1.0])
        assert spec.mz_range == (100.0, 300.0)

    def test_extract_range(self):
        spec = Spectrum(mz=[100.0, 200.0, 300.0], intensity=[1.0, 2.0, 3.0])
        sub = spec.extract_range(150.0, 250.0)
        assert len(sub.mz) == 1
        assert sub.mz[0] == 200.0

    def test_filter_by_intensity(self):
        spec = Spectrum(mz=[100.0, 200.0, 300.0], intensity=[100.0, 2000.0, 50.0])
        filtered = spec.filter_by_intensity(200.0)
        assert len(filtered.mz) == 1

    def test_top_n(self):
        spec = Spectrum(mz=[1.0, 2.0, 3.0, 4.0], intensity=[10.0, 40.0, 20.0, 30.0])
        top2 = spec.top_n(2)
        assert len(top2.mz) == 2

    def test_normalize(self):
        spec = Spectrum(mz=[100.0, 200.0], intensity=[500.0, 1000.0])
        normed = spec.normalize("max")
        assert abs(normed.intensity[1] - 1.0) < 1e-6

    def test_copy(self):
        spec = Spectrum(mz=[100.0], intensity=[1000.0], ms_level=2, rt=60.0)
        cp = spec.copy()
        assert len(cp.mz) == 1
        assert cp.ms_level == 2
        assert cp.rt == 60.0

    def test_sort_by_mz(self):
        spec = Spectrum(mz=[300.0, 100.0, 200.0], intensity=[3.0, 1.0, 2.0])
        spec.sort_by_mz()
        assert spec.is_sorted()
        assert spec.mz[0] == 100.0

    def test_find_nearest_mz(self):
        spec = Spectrum(mz=[100.0, 200.0, 300.0], intensity=[1.0, 1.0, 1.0])
        idx = spec.find_nearest_mz(195.0)
        assert idx == 1

    def test_find_nearest_empty_raises(self):
        spec = Spectrum()
        with pytest.raises(ValueError):
            spec.find_nearest_mz(100.0)

    def test_repr(self):
        spec = Spectrum(mz=[100.0], intensity=[1.0], rt=30.0)
        r = repr(spec)
        assert "Spectrum" in r
        assert "30.00" in r


class TestPeak:
    def test_create_peak(self):
        peak = Peak(mz=500.5, intensity=1234.0, rt=120.0)
        assert peak.mz == 500.5
        assert peak.intensity == 1234.0
        assert peak.rt == 120.0

    def test_peak_list(self):
        peaks = PeakList()
        peaks.add(Peak(mz=100.0, intensity=500.0))
        peaks.add(Peak(mz=200.0, intensity=1000.0))
        assert len(peaks) == 2


class TestAlgorithms:
    def test_pick_peaks(self):
        from masskit.algorithms import pick_peaks

        mz = np.linspace(100, 200, 1000)
        ints = np.zeros(1000)
        # Add a Gaussian peak
        center = 500
        for i in range(1000):
            ints[i] = 1000 * np.exp(-0.5 * ((i - center) / 10) ** 2)

        spec = Spectrum(mz=mz, intensity=ints)
        peaks = pick_peaks(spec)
        assert len(peaks) >= 1

    def test_smooth_spectrum(self):
        from masskit.algorithms import smooth_spectrum

        mz = np.linspace(100, 200, 100)
        ints = np.random.rand(100) * 1000
        spec = Spectrum(mz=mz, intensity=ints)
        smoothed = smooth_spectrum(spec)
        assert len(smoothed.mz) == len(mz)

    def test_estimate_noise(self):
        from masskit.algorithms import estimate_noise

        mz = np.linspace(100, 200, 1000)
        noise = np.random.normal(0, 10, 1000)
        spec = Spectrum(mz=mz, intensity=np.abs(noise))
        noise_level = estimate_noise(spec)
        assert noise_level > 0
