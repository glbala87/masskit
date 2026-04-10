"""Extended tests for the I/O module using real fixtures."""

import pytest
import os

from masskit.io import (
    load_mzml,
    load_mzxml,
    save_mztab,
    decode_binary,
)
from masskit.experiment import MSExperiment
from masskit.feature import FeatureMap, Feature
from masskit.exceptions import FileFormatError


class TestLoadMzMLOptions:
    def test_max_spectra(self, mzml_file):
        exp = load_mzml(mzml_file, max_spectra=3)
        assert exp.spectrum_count == 3

    def test_ms_level_filter(self, mzml_file):
        exp = load_mzml(mzml_file, ms_levels=[1])
        assert all(s.ms_level == 1 for s in exp.spectra)
        assert exp.spectrum_count > 0

    def test_ms2_filter(self, mzml_file):
        exp = load_mzml(mzml_file, ms_levels=[2])
        assert all(s.ms_level == 2 for s in exp.spectra)

    def test_rt_range_filter(self, mzml_file):
        exp = load_mzml(mzml_file, rt_range=(60.0, 70.0))
        for s in exp.spectra:
            assert 60.0 <= s.rt <= 70.0

    def test_skip_chromatograms(self, mzml_file):
        exp = load_mzml(mzml_file, skip_chromatograms=True)
        assert exp.spectrum_count > 0

    def test_progress_callback(self, mzml_file):
        progress_log = []

        def callback(current, total):
            progress_log.append((current, total))
            return True

        exp = load_mzml(mzml_file, progress_callback=callback)
        assert len(progress_log) > 0

    def test_progress_callback_abort(self, mzml_file):
        def callback(current, total):
            return False if current >= 2 else True

        exp = load_mzml(mzml_file, progress_callback=callback)
        # Should stop early
        assert exp.spectrum_count <= 2


class TestLoadMzXMLOptions:
    def test_max_spectra(self, mzxml_file):
        exp = load_mzxml(mzxml_file, max_spectra=2)
        assert exp.spectrum_count == 2

    def test_ms_level_filter(self, mzxml_file):
        exp = load_mzxml(mzxml_file, ms_levels=[1])
        assert all(s.ms_level == 1 for s in exp.spectra)


class TestSaveMzTab:
    def test_save_features(self, tmp_path):
        fmap = FeatureMap()
        for i in range(3):
            f = Feature()
            f.mz = 100.0 + i * 50
            f.rt = 60.0 + i * 30
            f.intensity = 1000.0 + i * 200
            f.charge = 1
            f.volume = 5000.0 + i * 100
            fmap.add(f)

        out = str(tmp_path / "out.mztab")
        save_mztab(fmap, out)
        assert os.path.exists(out)

        content = open(out).read()
        assert "MTD" in content
        assert "SMH" in content
        assert "feature_1" in content


class TestDecodeBinaryEdgeCases:
    def test_decode_big_endian(self):
        import base64
        import struct
        values = [1.5, 2.5, 3.5]
        binary = struct.pack(">3d", *values)
        encoded = base64.b64encode(binary).decode("ascii")
        result = decode_binary(encoded, is_64bit=True, is_little_endian=False)
        assert len(result) == 3
        assert abs(result[0] - 1.5) < 1e-6


class TestMzMLEdgeCases:
    def test_load_mzml_with_chromatograms(self, mzml_file):
        # Should at least not error when chromatograms are absent
        exp = load_mzml(mzml_file)
        assert isinstance(exp.chromatograms, list)
