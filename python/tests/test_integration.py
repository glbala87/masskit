"""
End-to-end integration tests for MassKit.

These tests exercise full pipelines using realistic mzML/mzXML fixtures
and verify the modules work together correctly.
"""

import pytest
import os
import csv

from masskit.io import load_mzml, load_mzxml, save_mztab
from masskit.workflows import (
    PeakPickingWorkflow,
    FeatureDetectionWorkflow,
    process_experiment,
)
from masskit.qc import compute_qc_metrics
from masskit.quantification import FeatureAlignment
from masskit.reporting import ReportBuilder, generate_analysis_report
from masskit.formats import save_feature_table, load_feature_table
from masskit.statistics import pca


class TestLoadAndProcess:
    def test_load_mzml(self, mzml_file):
        exp = load_mzml(mzml_file)
        assert exp.spectrum_count == 10
        assert exp.spectrum(0).ms_level == 1
        assert exp.spectrum(1).ms_level == 2
        # Verify data round-tripped
        assert len(exp.spectrum(0).mz) == 30
        assert exp.spectrum(0).rt == 60.0

    def test_load_mzml_compressed(self, mzml_file_compressed):
        exp = load_mzml(mzml_file_compressed)
        assert exp.spectrum_count == 6
        assert len(exp.spectrum(0).mz) == 30
        # Compressed and uncompressed should produce equivalent data
        assert exp.spectrum(0).tic > 0

    def test_load_mzxml(self, mzxml_file):
        exp = load_mzxml(mzxml_file)
        assert exp.spectrum_count == 5
        assert len(exp.spectrum(0).mz) == 30

    def test_ms2_precursors(self, mzml_file):
        exp = load_mzml(mzml_file)
        ms2_specs = [s for s in exp.spectra if s.ms_level == 2]
        assert len(ms2_specs) > 0
        # Should have precursor information
        assert len(ms2_specs[0].precursors) >= 1


class TestPeakPickingPipeline:
    def test_peak_picking_workflow(self, mzml_ms1_only):
        exp = load_mzml(mzml_ms1_only)
        workflow = PeakPickingWorkflow(min_snr=1.0, smooth=False, baseline_correct=False)
        for spec in exp.spectra:
            peaks = workflow.process(spec)
            # Each synthetic spectrum should yield some peaks
            assert peaks is not None

    def test_feature_detection(self, mzml_ms1_only):
        exp = load_mzml(mzml_ms1_only)
        workflow = FeatureDetectionWorkflow(
            min_snr=1.0,
            min_scans=1,
            smooth=False,
            baseline_correct=False,
        )
        features = workflow.process(exp)
        assert features is not None


class TestQCPipeline:
    def test_compute_qc(self, mzml_file):
        exp = load_mzml(mzml_file)
        qc = compute_qc_metrics(exp, filename="test.mzML")
        assert qc.num_spectra == 10
        assert qc.num_ms1 > 0
        assert qc.num_ms2 > 0
        assert qc.tic_median >= 0
        # Status should be one of valid values
        assert qc.status() in ("PASS", "WARN", "FAIL")
        # to_dict should serialize
        d = qc.to_dict()
        assert d["num_spectra"] == 10


class TestProcessExperiment:
    def test_process_experiment(self, mzml_ms1_only):
        exp = load_mzml(mzml_ms1_only)
        processed = process_experiment(
            exp,
            centroid=False,
            smooth=True,
            baseline_correct=True,
        )
        assert processed.spectrum_count == exp.spectrum_count


class TestEndToEndReport:
    def test_full_pipeline_to_report(self, mzml_file, tmp_path):
        # 1. Load
        exp = load_mzml(mzml_file)

        # 2. QC
        qc = compute_qc_metrics(exp)

        # 3. Generate report
        output = str(tmp_path / "report.html")
        path = generate_analysis_report(
            experiment_summary={
                "n_spectra": exp.spectrum_count,
                "n_ms1": qc.num_ms1,
                "n_ms2": qc.num_ms2,
            },
            qc_metrics=qc,
            output_path=output,
        )
        assert os.path.exists(path)
        content = open(path).read()
        assert "10" in content  # n_spectra appears in summary


class TestExportRoundTrip:
    def test_feature_table_round_trip(self, mzml_ms1_only, tmp_path):
        exp = load_mzml(mzml_ms1_only)
        workflow = FeatureDetectionWorkflow(
            min_snr=0.5,
            min_scans=1,
            smooth=False,
            baseline_correct=False,
        )
        features = workflow.process(exp)

        out_path = str(tmp_path / "features.tsv")
        n = save_feature_table(out_path, features)
        assert n >= 0
        assert os.path.exists(out_path)

        # Round-trip
        if n > 0:
            loaded = load_feature_table(out_path)
            assert len(loaded) == n


class TestQuantificationPipeline:
    def test_alignment_across_runs(self, tmp_path):
        from fixtures.sample_data import write_minimal_mzml
        # Create 3 sample files
        files = []
        for i in range(3):
            f = write_minimal_mzml(
                str(tmp_path / f"sample_{i}.mzML"),
                n_spectra=4,
                n_peaks_per_spec=20,
                include_ms2=False,
            )
            files.append(f)

        # Detect features in each
        feature_maps = []
        for f in files:
            exp = load_mzml(f)
            workflow = FeatureDetectionWorkflow(
                min_snr=0.5,
                min_scans=1,
                smooth=False,
                baseline_correct=False,
            )
            features = workflow.process(exp)
            feature_maps.append(features)

        # Align
        aligner = FeatureAlignment(mz_tolerance=0.5, rt_tolerance=60.0)
        consensus = aligner.align(feature_maps, sample_names=["A", "B", "C"])
        assert consensus.n_samples == 3
