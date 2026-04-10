"""
Smoke benchmark: load a 100MB+ synthetic mzML and report wall time + peak RSS.

Run manually:
    python -m benchmarks.bench_large_mzml
    python -m benchmarks.bench_large_mzml --target-mb 200 --json results.json

Or via pytest with the slow marker:
    pytest -m slow benchmarks/

This generates a synthetic mzML on disk, loads it via load_mzml, runs an
indexed pass via StreamingExperiment, and prints a summary table.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Make sibling tests/ importable for fixture generators
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "tests"))

from fixtures.sample_data import write_minimal_mzml  # noqa: E402

from masskit.io import load_mzml  # noqa: E402
from masskit.streaming import StreamingExperiment, IndexedMzMLReader  # noqa: E402
from masskit.workflows import process_experiment  # noqa: E402
from masskit.qc import compute_qc_metrics  # noqa: E402


# RSS is reported in KB on Linux, bytes on macOS
_RSS_DIVISOR = 1024 * 1024 if sys.platform != "darwin" else 1024 * 1024


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / _RSS_DIVISOR


def _generate_large_mzml(path: Path, target_mb: float) -> Path:
    """Generate a synthetic mzML file of approximately the requested size."""
    # Empirically: 500 peaks/spec → ~12 KB/spec on disk including XML overhead.
    # So target_mb * 1024 / 12 spectra.
    n_spectra = max(50, int((target_mb * 1024) / 12))
    print(f"  generating {n_spectra} spectra (target ~{target_mb:.0f} MB)...", flush=True)
    write_minimal_mzml(
        str(path),
        n_spectra=n_spectra,
        n_peaks_per_spec=500,
        include_ms2=True,
        compress=False,
    )
    return path


def _format_row(label: str, value_str: str) -> str:
    return f"  {label:<32} {value_str}"


def run_benchmark(target_mb: float = 100.0, output_json: str | None = None) -> Dict[str, Any]:
    print("=" * 60)
    print("MassKit smoke benchmark")
    print("=" * 60)

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "masskit_bench"
    tmp.mkdir(parents=True, exist_ok=True)
    mzml_path = tmp / f"bench_{int(target_mb)}MB.mzML"

    print(f"\n[1/4] Generating fixture at {mzml_path}")
    t0 = time.perf_counter()
    _generate_large_mzml(mzml_path, target_mb)
    gen_time = time.perf_counter() - t0
    actual_mb = mzml_path.stat().st_size / (1024 * 1024)
    print(_format_row("file size:", f"{actual_mb:.1f} MB"))
    print(_format_row("generation time:", f"{gen_time:.2f} s"))

    # ── 2. Full load via load_mzml ───────────────────────────────────
    print("\n[2/4] Loading via load_mzml() (full in-memory)")
    gc.collect()
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    exp = load_mzml(str(mzml_path))
    load_time = time.perf_counter() - t0
    rss_after = _peak_rss_mb()
    n_spectra = exp.spectrum_count
    print(_format_row("spectra loaded:", f"{n_spectra}"))
    print(_format_row("load time:", f"{load_time:.2f} s"))
    print(_format_row("MB/s throughput:", f"{actual_mb / load_time:.1f}"))
    print(_format_row("RSS before/after (peak):", f"{rss_before:.0f} / {rss_after:.0f} MB"))

    # ── 3. Streaming pass ───────────────────────────────────────────
    print("\n[3/4] Streaming pass via StreamingExperiment")
    del exp
    gc.collect()
    rss_before_stream = _peak_rss_mb()
    t0 = time.perf_counter()
    with StreamingExperiment(str(mzml_path)) as sexp:
        n_streamed = sum(1 for _ in sexp)
    stream_time = time.perf_counter() - t0
    rss_after_stream = _peak_rss_mb()
    print(_format_row("spectra streamed:", f"{n_streamed}"))
    print(_format_row("stream time:", f"{stream_time:.2f} s"))
    print(_format_row("RSS during stream:", f"{rss_after_stream:.0f} MB"))

    # ── 4. QC pipeline ──────────────────────────────────────────────
    print("\n[4/4] Computing QC metrics on full experiment")
    gc.collect()
    t0 = time.perf_counter()
    exp = load_mzml(str(mzml_path))
    qc = compute_qc_metrics(exp, filename=str(mzml_path))
    qc_time = time.perf_counter() - t0
    print(_format_row("QC time (incl. reload):", f"{qc_time:.2f} s"))
    print(_format_row("QC status:", qc.status()))
    print(_format_row("MS1 / MS2 spectra:", f"{qc.num_ms1} / {qc.num_ms2}"))

    # ── Summary ─────────────────────────────────────────────────────
    summary = {
        "target_mb": target_mb,
        "actual_mb": round(actual_mb, 2),
        "n_spectra": n_spectra,
        "generation_time_s": round(gen_time, 3),
        "load_time_s": round(load_time, 3),
        "load_mb_per_s": round(actual_mb / load_time, 2),
        "stream_time_s": round(stream_time, 3),
        "qc_time_s": round(qc_time, 3),
        "rss_after_load_mb": round(rss_after, 1),
        "rss_after_stream_mb": round(rss_after_stream, 1),
        "rss_overhead_ratio": round(rss_after / actual_mb, 2),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for k, v in summary.items():
        print(_format_row(k + ":", str(v)))

    # Sanity checks (smoke gate — flag anything obviously wrong)
    print("\nSanity gates:")
    failures = []
    if actual_mb < target_mb * 0.5:
        failures.append(f"file size {actual_mb:.0f}MB << target {target_mb:.0f}MB")
    if load_time > target_mb * 2:  # > 2 seconds per MB is alarming
        failures.append(f"load_time {load_time:.1f}s exceeds 2s/MB threshold")
    if rss_after > actual_mb * 50:  # > 50x file size is alarming
        failures.append(f"RSS {rss_after:.0f}MB > 50x file size")

    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
    else:
        print("  [PASS] all gates")

    if output_json:
        Path(output_json).write_text(json.dumps(summary, indent=2))
        print(f"\nResults written to {output_json}")

    # Cleanup
    try:
        mzml_path.unlink()
    except OSError:
        pass

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-mb", type=float, default=100.0,
                        help="Target file size in MB (default: 100)")
    parser.add_argument("--json", help="Write summary JSON to this path")
    args = parser.parse_args()
    run_benchmark(target_mb=args.target_mb, output_json=args.json)


if __name__ == "__main__":
    main()
