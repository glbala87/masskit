"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow benchmark tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (skipped unless --runslow is used)",
    )

from masskit.spectrum import Spectrum
from masskit.experiment import MSExperiment

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))

from fixtures.sample_data import (
    write_minimal_mzml,
    write_minimal_mzxml,
    write_minimal_fasta,
    write_minimal_mgf,
    write_thermo_style_mzml,
    write_bruker_style_mzml,
    write_waters_style_mzml,
)


@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducible tests."""
    np.random.seed(42)
    yield
    np.random.seed(None)


@pytest.fixture
def sample_spectrum():
    """Create a sample spectrum for testing."""
    mz = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    intensity = np.array([500.0, 1000.0, 2000.0, 800.0, 300.0])
    return Spectrum(mz=mz, intensity=intensity, ms_level=1, rt=60.0)


@pytest.fixture
def sample_ms2_spectrum():
    """Create a sample MS2 spectrum for testing."""
    mz = np.array([120.0, 180.0, 250.0, 350.0, 450.0])
    intensity = np.array([800.0, 1500.0, 3000.0, 1200.0, 500.0])
    return Spectrum(mz=mz, intensity=intensity, ms_level=2, rt=65.0)


@pytest.fixture
def sample_experiment():
    """Create a sample MSExperiment with a few spectra."""
    exp = MSExperiment()
    rng = np.random.default_rng(42)
    for i in range(5):
        mz = np.linspace(100, 500, 50)
        intensity = rng.lognormal(5, 1, 50)
        spec = Spectrum(mz=mz, intensity=intensity, ms_level=1, rt=float(i * 30))
        exp.add_spectrum(spec)
    return exp


# ── File-based fixtures ──────────────────────────────────────────────


@pytest.fixture
def mzml_file(tmp_path):
    """Create a small valid mzML file with mixed MS1/MS2 spectra."""
    return write_minimal_mzml(
        str(tmp_path / "sample.mzML"),
        n_spectra=10,
        n_peaks_per_spec=30,
        include_ms2=True,
    )


@pytest.fixture
def mzml_file_compressed(tmp_path):
    """Create a small mzML file with zlib-compressed binary."""
    return write_minimal_mzml(
        str(tmp_path / "sample_compressed.mzML"),
        n_spectra=6,
        compress=True,
    )


@pytest.fixture
def mzml_ms1_only(tmp_path):
    """Create an mzML file with only MS1 spectra."""
    return write_minimal_mzml(
        str(tmp_path / "ms1_only.mzML"),
        n_spectra=5,
        include_ms2=False,
    )


@pytest.fixture
def mzxml_file(tmp_path):
    """Create a small valid mzXML file."""
    return write_minimal_mzxml(
        str(tmp_path / "sample.mzXML"),
        n_spectra=5,
    )


@pytest.fixture
def fasta_file(tmp_path):
    """Create a small synthetic FASTA file."""
    return write_minimal_fasta(str(tmp_path / "proteins.fasta"))


@pytest.fixture
def mgf_file(tmp_path):
    """Create a small synthetic MGF file."""
    return write_minimal_mgf(str(tmp_path / "library.mgf"))


# ── Vendor-style mzML fixtures ───────────────────────────────────────


@pytest.fixture
def thermo_mzml(tmp_path):
    """mzML mimicking Thermo (msconvert from Xcalibur RAW): indexedmzML, scan IDs, mixed profile/centroid."""
    return write_thermo_style_mzml(str(tmp_path / "thermo.mzML"), n_spectra=6)


@pytest.fixture
def bruker_mzml(tmp_path):
    """mzML mimicking Bruker (CompassXport): zlib-compressed, simple namespace."""
    return write_bruker_style_mzml(str(tmp_path / "bruker.mzML"), n_spectra=6)


@pytest.fixture
def waters_mzml(tmp_path):
    """mzML mimicking Waters: 32-bit floats, function/process/scan IDs, negative polarity."""
    return write_waters_style_mzml(str(tmp_path / "waters.mzML"), n_spectra=5)
