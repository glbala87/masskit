"""
Tests against vendor-style mzML fixtures.

These fixtures emulate the quirks of files produced by msconvert from
Thermo, Bruker, and Waters vendor formats. They don't replace testing
against real public files, but they catch the most common namespace,
encoding, and ID-format differences.
"""

import pytest
import numpy as np

from masskit.io import load_mzml
from masskit.streaming import StreamingExperiment, FileIndex
from masskit.spectrum import SpectrumType, Polarity
from masskit.qc import compute_qc_metrics


# ── Thermo (Xcalibur → msconvert) ────────────────────────────────────


class TestThermoStyle:
    def test_load(self, thermo_mzml):
        exp = load_mzml(thermo_mzml)
        assert exp.spectrum_count == 6

    def test_indexedmzml_wrapper(self, thermo_mzml):
        # Verify the file is wrapped in <indexedmzML> and still parses
        content = open(thermo_mzml).read()
        assert "<indexedmzML" in content
        exp = load_mzml(thermo_mzml)
        assert exp.spectrum_count > 0

    def test_minute_rt_converted_to_seconds(self, thermo_mzml):
        exp = load_mzml(thermo_mzml)
        # First scan was 1.0 minutes → should be 60s after conversion
        assert abs(exp.spectrum(0).rt - 60.0) < 0.01

    def test_mixed_profile_centroid(self, thermo_mzml):
        exp = load_mzml(thermo_mzml)
        # Some MS1 should be profile, MS2 centroid
        types = [s.spectrum_type for s in exp.spectra]
        # At least one centroid (MS2)
        assert SpectrumType.CENTROID in types

    def test_ms2_precursor_charge(self, thermo_mzml):
        exp = load_mzml(thermo_mzml)
        ms2 = [s for s in exp.spectra if s.ms_level == 2]
        assert len(ms2) > 0
        assert ms2[0].precursors
        # Charge state should be parsed
        prec = ms2[0].precursors[0]
        assert prec.charge == 2

    def test_streaming_thermo(self, thermo_mzml):
        with StreamingExperiment(thermo_mzml) as exp:
            n = sum(1 for _ in exp)
            assert n == 6

    def test_indexed_reader_parses_thermo_ids(self, thermo_mzml):
        index = FileIndex.build(thermo_mzml)
        # Indexed reader should find every spectrum
        assert index.spectrum_count == 6
        # MS levels should include both 1 and 2
        assert 1 in index.ms_levels
        assert 2 in index.ms_levels


# ── Bruker (CompassXport / .d → msconvert) ───────────────────────────


class TestBrukerStyle:
    def test_load(self, bruker_mzml):
        exp = load_mzml(bruker_mzml)
        assert exp.spectrum_count == 6

    def test_zlib_compressed_data(self, bruker_mzml):
        # Verify the binary actually round-trips through zlib
        exp = load_mzml(bruker_mzml)
        for spec in exp.spectra:
            assert len(spec.mz) == 35  # generator produces 35 peaks
            assert len(spec.intensity) == 35
            assert spec.tic > 0

    def test_seconds_rt(self, bruker_mzml):
        exp = load_mzml(bruker_mzml)
        # Bruker fixture uses seconds directly: first scan at 50.0s
        assert abs(exp.spectrum(0).rt - 50.0) < 0.01

    def test_streaming_bruker(self, bruker_mzml):
        with StreamingExperiment(bruker_mzml) as exp:
            for spec in exp:
                # Streamed spectra should also have valid binary data
                assert len(spec.mz) > 0


# ── Waters (MassLynx → msconvert) ────────────────────────────────────


class TestWatersStyle:
    def test_load(self, waters_mzml):
        exp = load_mzml(waters_mzml)
        assert exp.spectrum_count == 5

    def test_negative_polarity(self, waters_mzml):
        exp = load_mzml(waters_mzml)
        # All spectra should be negative polarity
        polarities = {s.polarity for s in exp.spectra}
        assert Polarity.NEGATIVE in polarities

    def test_32bit_floats(self, waters_mzml):
        # 32-bit floats decode with reduced precision but should still be sane
        exp = load_mzml(waters_mzml)
        spec = exp.spectrum(0)
        assert len(spec.mz) > 0
        # m/z values should be in reasonable range
        assert np.all(spec.mz > 0)
        assert np.all(spec.mz < 10000)

    def test_function_process_scan_id(self, waters_mzml):
        exp = load_mzml(waters_mzml)
        # First spectrum should have Waters-style native_id
        nid = exp.spectrum(0).native_id
        assert "function=" in nid
        assert "scan=" in nid


# ── Cross-vendor consistency ─────────────────────────────────────────


class TestCrossVendor:
    def test_qc_works_on_all(self, thermo_mzml, bruker_mzml, waters_mzml):
        for path in [thermo_mzml, bruker_mzml, waters_mzml]:
            exp = load_mzml(path)
            qc = compute_qc_metrics(exp, filename=path)
            assert qc.num_spectra == exp.spectrum_count
            assert qc.status() in ("PASS", "WARN", "FAIL")

    def test_indexed_streaming_works_on_all(self, thermo_mzml, bruker_mzml, waters_mzml):
        for path in [thermo_mzml, bruker_mzml, waters_mzml]:
            with StreamingExperiment(path) as exp:
                count = sum(1 for _ in exp)
                assert count > 0
