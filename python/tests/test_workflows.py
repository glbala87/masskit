"""Tests for high-level workflow modules."""

import pytest
import numpy as np

from masskit.workflows import (
    PeakPickingWorkflow,
    FeatureDetectionWorkflow,
    process_experiment,
)
from masskit.spectrum import Spectrum
from masskit.experiment import MSExperiment
from masskit.io import load_mzml


class TestPeakPickingWorkflow:
    def test_default_init(self):
        wf = PeakPickingWorkflow()
        assert wf.smooth is True
        assert wf.baseline_correct is True
        assert wf.min_snr == 3.0

    def test_process_spectrum(self):
        # Spectrum with clear peaks
        mz = np.linspace(100, 200, 500)
        intensity = np.zeros(500)
        # Add 3 Gaussian peaks
        for center in [100, 250, 400]:
            for i in range(500):
                intensity[i] += 10000 * np.exp(-0.5 * ((i - center) / 5) ** 2)
        spec = Spectrum(mz=mz, intensity=intensity)

        wf = PeakPickingWorkflow(min_snr=1.0, smooth=False, baseline_correct=False)
        peaks = wf.process(spec)
        assert peaks is not None

    def test_with_smoothing(self):
        mz = np.linspace(100, 200, 500)
        intensity = np.random.RandomState(42).rand(500) * 100 + 1000
        spec = Spectrum(mz=mz, intensity=intensity)

        wf = PeakPickingWorkflow(smooth=True, baseline_correct=True, min_snr=1.0)
        peaks = wf.process(spec)
        assert peaks is not None

    def test_no_fit(self):
        wf = PeakPickingWorkflow(fit_peaks=False)
        assert wf.fit_peaks is False


class TestFeatureDetectionWorkflow:
    def test_init(self):
        wf = FeatureDetectionWorkflow(mz_tolerance=0.02, rt_tolerance=15.0)
        assert wf.mz_tolerance == 0.02
        assert wf.rt_tolerance == 15.0

    def test_empty_experiment(self):
        wf = FeatureDetectionWorkflow()
        exp = MSExperiment()
        result = wf.process(exp)
        assert result is not None
        assert len(result) == 0

    def test_process_real_data(self, mzml_ms1_only):
        exp = load_mzml(mzml_ms1_only)
        wf = FeatureDetectionWorkflow(
            mz_tolerance=10.0,
            rt_tolerance=60.0,
            min_scans=1,
            min_snr=0.5,
            smooth=False,
            baseline_correct=False,
        )
        features = wf.process(exp)
        assert features is not None

    def test_progress_callback(self, mzml_ms1_only):
        exp = load_mzml(mzml_ms1_only)
        progress_log = []

        def cb(current, total):
            progress_log.append((current, total))

        wf = FeatureDetectionWorkflow(
            min_scans=1, min_snr=0.5,
            smooth=False, baseline_correct=False,
        )
        wf.process(exp, progress_callback=cb)
        assert len(progress_log) > 0

    def test_ppm_tolerance(self):
        wf = FeatureDetectionWorkflow(mz_tolerance_ppm=True)
        assert wf.mz_tolerance_ppm is True


class TestProcessExperiment:
    def test_basic(self, mzml_ms1_only):
        exp = load_mzml(mzml_ms1_only)
        result = process_experiment(exp, centroid=False)
        assert result.spectrum_count == exp.spectrum_count

    def test_with_centroid(self, mzml_ms1_only):
        exp = load_mzml(mzml_ms1_only)
        result = process_experiment(exp, centroid=True, smooth=True)
        assert result is not None

    def test_no_processing(self, mzml_ms1_only):
        exp = load_mzml(mzml_ms1_only)
        result = process_experiment(
            exp, centroid=False, smooth=False, baseline_correct=False,
        )
        assert result.spectrum_count == exp.spectrum_count

    def test_progress_callback(self, mzml_ms1_only):
        exp = load_mzml(mzml_ms1_only)
        log = []
        process_experiment(
            exp, centroid=False, smooth=False, baseline_correct=False,
            progress_callback=lambda c, t: log.append((c, t)),
        )
        assert len(log) == exp.spectrum_count
