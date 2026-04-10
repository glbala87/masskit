"""Tests for the CLI module."""

import pytest
from masskit.cli import create_parser, main, _load_experiment


class TestParser:
    def test_create_parser(self):
        parser = create_parser()
        assert parser is not None

    def test_no_command(self):
        result = main([])
        assert result == 0

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_info_missing_file(self):
        with pytest.raises((FileNotFoundError, SystemExit)):
            main(["info", "nonexistent.mzML"])

    def test_peaks_parser(self):
        parser = create_parser()
        args = parser.parse_args(["peaks", "test.mzML", "--snr", "5.0", "--tsv"])
        assert args.command == "peaks"
        assert args.snr == 5.0
        assert args.tsv is True

    def test_xic_parser(self):
        parser = create_parser()
        args = parser.parse_args(["xic", "test.mzML", "--mz", "500.0", "--tolerance", "0.05"])
        assert args.command == "xic"
        assert args.mz == 500.0
        assert args.tolerance == 0.05

    def test_quantify_parser(self):
        parser = create_parser()
        args = parser.parse_args(["quantify", "a.mzML", "b.mzML", "--normalize", "median"])
        assert args.command == "quantify"
        assert len(args.files) == 2
        assert args.normalize == "median"

    def test_search_parser(self):
        parser = create_parser()
        args = parser.parse_args(["search", "q.mzML", "-l", "lib.msp", "--method", "entropy"])
        assert args.command == "search"
        assert args.method == "entropy"

    def test_qc_parser(self):
        parser = create_parser()
        args = parser.parse_args(["qc", "a.mzML", "b.mzML", "--plot"])
        assert args.command == "qc"
        assert args.plot is True


class TestLoadExperiment:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_experiment("/nonexistent/file.mzML")

    def test_unknown_format_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_experiment("/nonexistent/file.raw")
