"""Extended CLI tests using fixture files."""

import pytest
import os
import csv

from masskit.cli import main


class TestCmdInfo:
    def test_info_basic(self, mzml_file, capsys):
        rc = main(["info", mzml_file])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Spectra:" in out
        assert "MS1" in out

    def test_info_verbose(self, mzml_file, capsys):
        rc = main(["info", mzml_file, "-v"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "First 10 spectra" in out


class TestCmdPeaks:
    def test_peaks_to_stdout(self, mzml_ms1_only, capsys):
        rc = main(["peaks", mzml_ms1_only, "--snr", "0.5"])
        assert rc == 0 or rc is None

    def test_peaks_to_file(self, mzml_ms1_only, tmp_path):
        out = str(tmp_path / "peaks.csv")
        rc = main(["peaks", mzml_ms1_only, "-o", out, "--snr", "0.5"])
        assert rc == 0 or rc is None
        assert os.path.exists(out)

    def test_peaks_tsv(self, mzml_ms1_only, tmp_path):
        out = str(tmp_path / "peaks.tsv")
        rc = main(["peaks", mzml_ms1_only, "-o", out, "--tsv", "--snr", "0.5"])
        assert rc == 0 or rc is None
        content = open(out).read()
        assert "\t" in content

    def test_peaks_no_ms_level_data(self, mzml_ms1_only, capsys):
        rc = main(["peaks", mzml_ms1_only, "--ms-level", "5"])
        # Should fail with no MS5 spectra
        assert rc == 1


class TestCmdXIC:
    def test_xic_to_stdout(self, mzml_file, capsys):
        rc = main(["xic", mzml_file, "--mz", "500.0", "--tolerance", "100.0"])
        assert rc == 0 or rc is None

    def test_xic_to_file(self, mzml_file, tmp_path):
        out = str(tmp_path / "xic.tsv")
        rc = main(["xic", mzml_file, "--mz", "500.0", "--tolerance", "100.0", "-o", out])
        assert rc == 0 or rc is None
        assert os.path.exists(out)


class TestCmdQC:
    def test_qc_basic(self, mzml_file, capsys):
        rc = main(["qc", mzml_file])
        assert rc == 0 or rc is None
        out = capsys.readouterr().out
        assert "TIC" in out or "Spectra" in out

    def test_qc_multiple_files(self, mzml_file, mzml_ms1_only):
        rc = main(["qc", mzml_file, mzml_ms1_only])
        assert rc == 0 or rc is None


class TestCmdConvert:
    def test_convert_to_mztab(self, mzml_ms1_only, tmp_path):
        out = str(tmp_path / "out.mztab")
        rc = main(["convert", mzml_ms1_only, "-o", out, "-f", "mztab"])
        assert rc == 0 or rc is None
        assert os.path.exists(out)

    def test_convert_default_output(self, mzml_ms1_only, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        rc = main(["convert", mzml_ms1_only, "-f", "mztab"])
        assert rc == 0 or rc is None


class TestCmdQuantify:
    def test_quantify_multiple_files(self, tmp_path):
        from fixtures.sample_data import write_minimal_mzml
        files = []
        for i in range(2):
            f = write_minimal_mzml(
                str(tmp_path / f"sample_{i}.mzML"),
                n_spectra=4,
                include_ms2=False,
            )
            files.append(f)

        out = str(tmp_path / "quant.tsv")
        rc = main([
            "quantify", *files, "-o", out,
            "--mz-tolerance", "1.0",
            "--rt-tolerance", "60.0",
            "--min-presence", "0.0",
        ])
        # Quantification on tiny synthetic data may produce no features but shouldn't crash
        assert rc is None or rc == 0


class TestCmdSearch:
    def test_search_with_mgf_library(self, mzml_file, mgf_file, tmp_path):
        out = str(tmp_path / "search.tsv")
        rc = main([
            "search", mzml_file,
            "-l", mgf_file,
            "-o", out,
            "--min-score", "0.0",
            "--method", "cosine",
        ])
        # Tiny synthetic data may not match anything
        assert rc is None or rc == 0
