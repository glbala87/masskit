"""
Optional integration tests against real public mzML files.

These are skipped by default because they download large files.
Enable with the MASSKIT_REAL_DATA env var:

    MASSKIT_REAL_DATA=1 pytest tests/test_real_public_data.py

The test downloads small public mzML files from PRIDE/MetaboLights mirrors,
caches them in ~/.cache/masskit_test_data/, and exercises the full pipeline.
"""

import os
import urllib.request
from pathlib import Path

import pytest

from masskit.io import load_mzml
from masskit.qc import compute_qc_metrics
from masskit.streaming import StreamingExperiment

# Each entry: (cache_filename, list of mirror URLs to try, expected min spectra)
# These URLs point to small public test files maintained by the proteomics community.
# The test is gated and skipped if no MASSKIT_REAL_DATA env var is set, so we never
# download in normal test runs.
_HUPO_BASE = "https://raw.githubusercontent.com/HUPO-PSI/mzML/master/examples"

PUBLIC_FILES = [
    # ProteoWizard reference - small Thermo-derived test file (mixed MS1/MS2, mzML 1.1)
    (
        "tiny.pwiz.1.1.mzML",
        [f"{_HUPO_BASE}/tiny.pwiz.1.1.mzML"],
        1,
    ),
    # LTQ-FT (Thermo), mzML 0.99 — tests mislabeled binary precision (says 32-bit but is 64-bit)
    (
        "tiny4_LTQ-FT.mzML0.99.1.mzML",
        [f"{_HUPO_BASE}/tiny4_LTQ-FT.mzML0.99.1.mzML"],
        1,
    ),
    # 1 minute of data, mzML 0.99 — tests big-endian 32-bit binary
    (
        "1min.mzML",
        [f"{_HUPO_BASE}/1min.mzML"],
        10,
    ),
]

CACHE_DIR = Path.home() / ".cache" / "masskit_test_data"


def _download_if_needed(filename: str, urls: list) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / filename
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    last_err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "masskit-test"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                cached.write_bytes(resp.read())
            return cached
        except Exception as e:  # noqa: BLE001 — best-effort multi-mirror fetch
            last_err = e
            continue

    pytest.skip(f"could not download {filename}: {last_err}")


@pytest.fixture(scope="module")
def real_data_enabled():
    if not os.environ.get("MASSKIT_REAL_DATA"):
        pytest.skip("set MASSKIT_REAL_DATA=1 to run real-data tests")


@pytest.mark.parametrize("filename,urls,min_spectra", PUBLIC_FILES)
def test_load_real_mzml(real_data_enabled, filename, urls, min_spectra):
    """Verify a real public mzML file loads cleanly."""
    path = _download_if_needed(filename, urls)
    exp = load_mzml(str(path))
    assert exp.spectrum_count >= min_spectra
    # Every spectrum should have valid m/z and intensity arrays
    for spec in exp.spectra[:5]:  # check at most first 5 to keep test fast
        assert len(spec.mz) == len(spec.intensity)
        assert spec.tic >= 0


@pytest.mark.parametrize("filename,urls,min_spectra", PUBLIC_FILES)
def test_qc_real_mzml(real_data_enabled, filename, urls, min_spectra):
    path = _download_if_needed(filename, urls)
    exp = load_mzml(str(path))
    qc = compute_qc_metrics(exp, filename=filename)
    assert qc.num_spectra == exp.spectrum_count
    assert qc.status() in ("PASS", "WARN", "FAIL")


@pytest.mark.parametrize("filename,urls,min_spectra", PUBLIC_FILES)
def test_streaming_real_mzml(real_data_enabled, filename, urls, min_spectra):
    path = _download_if_needed(filename, urls)
    with StreamingExperiment(str(path)) as exp:
        n = sum(1 for _ in exp)
        assert n >= min_spectra
