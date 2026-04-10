"""
Centralized configuration management for MassKit.

Supports configuration via:
1. Programmatic defaults
2. Config files (YAML/JSON)
3. Environment variable overrides (MASSKIT_ prefix)

Example:
    >>> from masskit.config import LCMSConfig
    >>> config = LCMSConfig()
    >>> config.peak_picking.min_snr
    3.0
    >>> config = LCMSConfig.from_file("config.json")
    >>> config = LCMSConfig.from_env()
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

ENV_PREFIX = "MASSKIT_"


@dataclass
class PeakPickingConfig:
    """Configuration for peak picking."""
    min_snr: float = 3.0
    min_intensity: float = 0.0
    smooth: bool = True
    smooth_method: str = "gaussian"
    smooth_window: int = 5
    baseline_correct: bool = True
    baseline_method: str = "snip"
    fit_peaks: bool = True


@dataclass
class QuantificationConfig:
    """Configuration for quantification."""
    mz_tolerance: float = 0.01
    rt_tolerance: float = 30.0
    min_presence: float = 0.5
    normalization: str = "none"
    fill_missing: str = "half_min"


@dataclass
class SearchConfig:
    """Configuration for spectral/database search."""
    precursor_tolerance_ppm: float = 10.0
    fragment_tolerance_da: float = 0.02
    min_matched_peaks: int = 4
    min_score: float = 0.7
    fdr_threshold: float = 0.01
    enzyme: str = "trypsin"
    missed_cleavages: int = 2


@dataclass
class IOConfig:
    """Configuration for I/O operations."""
    max_spectra: int = 0
    skip_chromatograms: bool = False
    max_file_size_mb: int = 0


@dataclass
class LCMSConfig:
    """
    Central configuration for all MassKit operations.

    Example:
        >>> config = LCMSConfig()
        >>> config.peak_picking.min_snr = 5.0
        >>> config.quantification.normalization = "median"
    """
    peak_picking: PeakPickingConfig = field(default_factory=PeakPickingConfig)
    quantification: QuantificationConfig = field(default_factory=QuantificationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    io: IOConfig = field(default_factory=IOConfig)
    log_level: str = "WARNING"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        data = self.to_dict()
        Path(filepath).write_text(json.dumps(data, indent=2))
        logger.info("Configuration saved to %s", filepath)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LCMSConfig":
        """Create configuration from a dictionary."""
        config = cls()
        if "peak_picking" in data:
            for k, v in data["peak_picking"].items():
                if hasattr(config.peak_picking, k):
                    setattr(config.peak_picking, k, v)
        if "quantification" in data:
            for k, v in data["quantification"].items():
                if hasattr(config.quantification, k):
                    setattr(config.quantification, k, v)
        if "search" in data:
            for k, v in data["search"].items():
                if hasattr(config.search, k):
                    setattr(config.search, k, v)
        if "io" in data:
            for k, v in data["io"].items():
                if hasattr(config.io, k):
                    setattr(config.io, k, v)
        if "log_level" in data:
            config.log_level = data["log_level"]
        return config

    @classmethod
    def from_file(cls, filepath: str) -> "LCMSConfig":
        """
        Load configuration from a JSON or YAML file.

        Args:
            filepath: Path to config file (.json or .yaml/.yml)

        Returns:
            LCMSConfig instance
        """
        path = Path(filepath)
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {filepath}")

        text = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError:
                raise ConfigurationError(
                    "PyYAML required for YAML config files: pip install pyyaml"
                )
        elif path.suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Invalid JSON in {filepath}: {e}")
        else:
            raise ConfigurationError(
                f"Unsupported config format: {path.suffix} (use .json or .yaml)"
            )

        logger.info("Configuration loaded from %s", filepath)
        return cls.from_dict(data)

    @classmethod
    def from_env(cls) -> "LCMSConfig":
        """
        Create configuration from environment variables.

        Environment variables use MASSKIT_ prefix with double underscore
        for nesting:
            MASSKIT_PEAK_PICKING__MIN_SNR=5.0
            MASSKIT_QUANTIFICATION__NORMALIZATION=median
            MASSKIT_LOG_LEVEL=DEBUG
        """
        config = cls()

        for key, value in os.environ.items():
            if not key.startswith(ENV_PREFIX):
                continue
            parts = key[len(ENV_PREFIX):].lower().split("__")

            if len(parts) == 1:
                if hasattr(config, parts[0]):
                    setattr(config, parts[0], value)
            elif len(parts) == 2:
                section_name, param_name = parts
                section = getattr(config, section_name, None)
                if section is not None and hasattr(section, param_name):
                    current = getattr(section, param_name)
                    # Type coercion
                    try:
                        if isinstance(current, bool):
                            setattr(section, param_name, value.lower() in ("true", "1", "yes"))
                        elif isinstance(current, int):
                            setattr(section, param_name, int(value))
                        elif isinstance(current, float):
                            setattr(section, param_name, float(value))
                        else:
                            setattr(section, param_name, value)
                    except (ValueError, TypeError) as e:
                        logger.warning("Invalid env var %s=%s: %s", key, value, e)

        return config

    @classmethod
    def discover(cls) -> "LCMSConfig":
        """
        Auto-discover configuration from standard locations.

        Checks (in order):
        1. ./masskit.json or ./masskit.yaml
        2. ~/.masskit/config.json or ~/.masskit/config.yaml
        3. Environment variables (MASSKIT_ prefix)
        4. Defaults
        """
        # Check current directory
        for name in ("masskit.json", "masskit.yaml", "masskit.yml"):
            if Path(name).exists():
                logger.info("Found config: %s", name)
                return cls.from_file(name)

        # Check home directory
        home_dir = Path.home() / ".masskit"
        for name in ("config.json", "config.yaml", "config.yml"):
            path = home_dir / name
            if path.exists():
                logger.info("Found config: %s", path)
                return cls.from_file(str(path))

        # Check env vars
        has_env = any(k.startswith(ENV_PREFIX) for k in os.environ)
        if has_env:
            logger.info("Loading config from environment variables")
            return cls.from_env()

        return cls()

    def validate(self) -> None:
        """Validate configuration values."""
        errors = []
        if self.peak_picking.min_snr < 0:
            errors.append("peak_picking.min_snr must be >= 0")
        if self.quantification.mz_tolerance <= 0:
            errors.append("quantification.mz_tolerance must be > 0")
        if self.quantification.rt_tolerance <= 0:
            errors.append("quantification.rt_tolerance must be > 0")
        if not 0 <= self.quantification.min_presence <= 1:
            errors.append("quantification.min_presence must be between 0 and 1")
        if self.search.fdr_threshold <= 0 or self.search.fdr_threshold > 1:
            errors.append("search.fdr_threshold must be between 0 and 1")
        if self.search.missed_cleavages < 0:
            errors.append("search.missed_cleavages must be >= 0")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(f"Invalid log_level: {self.log_level}")

        if errors:
            raise ConfigurationError(
                "Configuration validation failed:\n  " + "\n  ".join(errors)
            )
