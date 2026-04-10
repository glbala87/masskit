"""Tests for CLI config wiring."""

import pytest
import json
import os

from masskit.cli import main, _load_cli_config, _apply_config_defaults, create_parser
from masskit.config import LCMSConfig


class TestConfigWiring:
    def test_config_flag_loads_file(self, tmp_path):
        config_path = tmp_path / "cfg.json"
        config = LCMSConfig()
        config.peak_picking.min_snr = 7.5
        config.save(str(config_path))

        parser = create_parser()
        args = parser.parse_args(["--config", str(config_path), "info", "dummy.mzML"])
        loaded = _load_cli_config(args)
        assert loaded.peak_picking.min_snr == 7.5

    def test_no_config_uses_discover(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        parser = create_parser()
        args = parser.parse_args(["info", "dummy.mzML"])
        loaded = _load_cli_config(args)
        assert loaded.peak_picking.min_snr == 3.0  # default

    def test_apply_config_defaults_to_args(self):
        parser = create_parser()
        args = parser.parse_args(["peaks", "x.mzML"])
        config = LCMSConfig()
        config.peak_picking.min_snr = 9.9
        result = _apply_config_defaults(args, config)
        assert result.snr == 9.9

    def test_cli_override_wins(self):
        parser = create_parser()
        # Explicit --snr 5.0 should NOT be overridden by config
        args = parser.parse_args(["peaks", "x.mzML", "--snr", "5.0"])
        config = LCMSConfig()
        config.peak_picking.min_snr = 9.9
        result = _apply_config_defaults(args, config)
        assert result.snr == 5.0

    def test_log_level_flag(self, mzml_file):
        rc = main(["--log-level", "ERROR", "info", mzml_file])
        assert rc == 0

    def test_invalid_config_file(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        rc = main(["--config", str(bad), "info", "dummy.mzML"])
        assert rc == 1

    def test_config_env_vars(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MASSKIT_PEAK_PICKING__MIN_SNR", "12.0")
        parser = create_parser()
        args = parser.parse_args(["info", "x.mzML"])
        loaded = _load_cli_config(args)
        assert loaded.peak_picking.min_snr == 12.0
