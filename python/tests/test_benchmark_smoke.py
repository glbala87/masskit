"""
Pytest wrapper for the smoke benchmark.

By default this is skipped (it generates a 100MB+ file). Run explicitly:

    pytest tests/test_benchmark_smoke.py -m slow --runslow
"""

import pytest
import sys
from pathlib import Path

# Make benchmarks/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def runslow(request):
    return request.config.getoption("--runslow", default=False)


@pytest.mark.slow
def test_smoke_benchmark_25mb(runslow):
    """Smaller smoke test always runnable; 25MB takes a few seconds."""
    if not runslow:
        pytest.skip("requires --runslow")
    from benchmarks.bench_large_mzml import run_benchmark
    summary = run_benchmark(target_mb=25.0)
    # Sanity assertions
    assert summary["n_spectra"] > 100
    assert summary["actual_mb"] > 10
    assert summary["load_time_s"] < summary["target_mb"] * 2  # < 2 s/MB
    assert summary["rss_overhead_ratio"] < 50  # not > 50x file size


@pytest.mark.slow
def test_smoke_benchmark_100mb(runslow):
    """The full 100MB+ smoke benchmark."""
    if not runslow:
        pytest.skip("requires --runslow")
    from benchmarks.bench_large_mzml import run_benchmark
    summary = run_benchmark(target_mb=100.0)
    assert summary["actual_mb"] >= 80  # at least 80 MB
    assert summary["load_mb_per_s"] > 5  # at least 5 MB/s
    assert summary["rss_overhead_ratio"] < 50


@pytest.mark.slow
def test_gb_streaming_benchmark(runslow):
    """
    GB-scale streaming benchmark.

    Verifies that StreamingExperiment keeps memory bounded when
    processing a file larger than RAM would comfortably hold.
    Uses 500MB by default (reduced from 1GB for CI speed).
    """
    if not runslow:
        pytest.skip("requires --runslow")
    from benchmarks.bench_gb_streaming import run_gb_benchmark
    summary = run_gb_benchmark(target_mb=500.0)
    assert summary["actual_mb"] >= 400
    assert summary["memory_bounded"], (
        f"RSS {summary['rss_max_during_mb']:.0f}MB was not bounded below "
        f"50% of file size {summary['actual_mb']:.0f}MB"
    )
    assert summary["stream_mb_per_s"] > 3  # at least 3 MB/s
