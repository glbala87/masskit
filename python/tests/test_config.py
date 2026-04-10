"""Tests for the configuration module."""

import pytest
import json
import tempfile
import os

from masskit.config import (
    LCMSConfig,
    PeakPickingConfig,
    QuantificationConfig,
    SearchConfig,
    IOConfig,
)
from masskit.exceptions import ConfigurationError


class TestDefaults:
    def test_default_config(self):
        config = LCMSConfig()
        assert config.peak_picking.min_snr == 3.0
        assert config.quantification.normalization == "none"
        assert config.search.fdr_threshold == 0.01

    def test_subconfig_defaults(self):
        pp = PeakPickingConfig()
        assert pp.smooth is True
        assert pp.smooth_method == "gaussian"


class TestToDict:
    def test_round_trip(self):
        config = LCMSConfig()
        d = config.to_dict()
        assert "peak_picking" in d
        assert d["peak_picking"]["min_snr"] == 3.0

    def test_from_dict(self):
        data = {
            "peak_picking": {"min_snr": 5.0},
            "search": {"fdr_threshold": 0.05},
        }
        config = LCMSConfig.from_dict(data)
        assert config.peak_picking.min_snr == 5.0
        assert config.search.fdr_threshold == 0.05
        # Unchanged defaults
        assert config.quantification.mz_tolerance == 0.01


class TestFileConfig:
    def test_save_and_load_json(self):
        config = LCMSConfig()
        config.peak_picking.min_snr = 7.5
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            try:
                config.save(f.name)
                loaded = LCMSConfig.from_file(f.name)
                assert loaded.peak_picking.min_snr == 7.5
            finally:
                os.unlink(f.name)

    def test_missing_file(self):
        with pytest.raises(ConfigurationError):
            LCMSConfig.from_file("/nonexistent/config.json")

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("{invalid json")
            f.flush()
            try:
                with pytest.raises(ConfigurationError):
                    LCMSConfig.from_file(f.name)
            finally:
                os.unlink(f.name)

    def test_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write("[section]\nkey = 'value'\n")
            f.flush()
            try:
                with pytest.raises(ConfigurationError, match="Unsupported"):
                    LCMSConfig.from_file(f.name)
            finally:
                os.unlink(f.name)


class TestEnvConfig:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MASSKIT_PEAK_PICKING__MIN_SNR", "10.0")
        monkeypatch.setenv("MASSKIT_QUANTIFICATION__NORMALIZATION", "median")
        monkeypatch.setenv("MASSKIT_LOG_LEVEL", "DEBUG")
        config = LCMSConfig.from_env()
        assert config.peak_picking.min_snr == 10.0
        assert config.quantification.normalization == "median"
        assert config.log_level == "DEBUG"

    def test_bool_env(self, monkeypatch):
        monkeypatch.setenv("MASSKIT_PEAK_PICKING__SMOOTH", "false")
        config = LCMSConfig.from_env()
        assert config.peak_picking.smooth is False

    def test_invalid_env_value(self, monkeypatch):
        monkeypatch.setenv("MASSKIT_PEAK_PICKING__MIN_SNR", "not_a_number")
        # Should not raise, just log warning
        config = LCMSConfig.from_env()
        assert config.peak_picking.min_snr == 3.0  # Default unchanged


class TestValidation:
    def test_valid_config(self):
        config = LCMSConfig()
        config.validate()  # Should not raise

    def test_invalid_snr(self):
        config = LCMSConfig()
        config.peak_picking.min_snr = -1.0
        with pytest.raises(ConfigurationError, match="min_snr"):
            config.validate()

    def test_invalid_fdr(self):
        config = LCMSConfig()
        config.search.fdr_threshold = 2.0
        with pytest.raises(ConfigurationError, match="fdr_threshold"):
            config.validate()

    def test_invalid_log_level(self):
        config = LCMSConfig()
        config.log_level = "VERBOSE"
        with pytest.raises(ConfigurationError, match="log_level"):
            config.validate()

    def test_multiple_errors(self):
        config = LCMSConfig()
        config.peak_picking.min_snr = -1.0
        config.search.fdr_threshold = 2.0
        with pytest.raises(ConfigurationError):
            config.validate()


class TestDiscover:
    def test_discover_defaults(self):
        # In test env with no config files, should return defaults
        config = LCMSConfig.discover()
        assert config.peak_picking.min_snr == 3.0
