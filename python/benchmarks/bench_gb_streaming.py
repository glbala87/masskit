"""
GB-scale streaming benchmark: tests memory-bounded processing of 1GB+ mzML.

Unlike bench_large_mzml.py which tests full in-memory load, this benchmark
demonstrates that StreamingExperiment keeps peak RSS bounded regardless
of file size by streaming spectra one-at-a-time.

Run manually:
    python -m benchmarks.bench_gb_streaming
    python -m benchmarks.bench_gb_streaming --target-mb 1000

Or via pytest:
    pytest tests/test_benchmark_smoke.py --runslow -k gb

NOTE: Generation of the mzML fixture itself requires enough disk space.
      The file is deleted after the benchmark.
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

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "tests"))

from fixtures.sample_data import write_minimal_mzml  # noqa: E402
from masskit.streaming import StreamingExperiment  # noqa: E402

_RSS_DIVISOR = 1024 * 1024 if sys.platform != "darwin" else 1024 * 1024


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIVISOR


def _format_row(label: str, value: str) -> str:
    return f"  {label:<36} {value}"


def run_gb_benchmark(target_mb: float = 1000.0, output_json: str | None = None) -> Dict[str, Any]:
    print("=" * 64)
    print("MassKit GB-scale streaming benchmark")
    print("=" * 64)

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "masskit_bench"
    tmp.mkdir(parents=True, exist_ok=True)
    mzml_path = tmp / f"bench_{int(target_mb)}MB.mzML"

    # ── 1. Generate fixture ──────────────────────────────────────────
    n_spectra = max(50, int((target_mb * 1024) / 12))
    print(f"\n[1/3] Generating {n_spectra} spectra (~{target_mb:.0f} MB)...")
    t0 = time.perf_counter()
    write_minimal_mzml(
        str(mzml_path),
        n_spectra=n_spectra,
        n_peaks_per_spec=500,
        include_ms2=True,
        compress=False,
    )
    gen_time = time.perf_counter() - t0
    actual_mb = mzml_path.stat().st_size / (1024 * 1024)
    print(_format_row("file size:", f"{actual_mb:.1f} MB"))
    print(_format_row("generation time:", f"{gen_time:.1f} s"))

    # ── 2. Streaming pass (memory-bounded) ───────────────────────────
    print("\n[2/3] Streaming pass (memory should stay bounded)")
    gc.collect()
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()

    n_streamed = 0
    ms1_count = 0
    ms2_count = 0
    tic_sum = 0.0
    rss_max_during = rss_before

    with StreamingExperiment(str(mzml_path)) as sexp:
        for spec in sexp:
            n_streamed += 1
            tic_sum += spec.tic
            if spec.ms_level == 1:
                ms1_count += 1
            else:
                ms2_count += 1
            # Check RSS every 1000 spectra to avoid overhead
            if n_streamed % 1000 == 0:
                cur_rss = _peak_rss_mb()
                rss_max_during = max(rss_max_during, cur_rss)

    stream_time = time.perf_counter() - t0
    rss_after = _peak_rss_mb()
    rss_max_during = max(rss_max_during, rss_after)

    print(_format_row("spectra streamed:", f"{n_streamed}"))
    print(_format_row("MS1/MS2:", f"{ms1_count}/{ms2_count}"))
    print(_format_row("total TIC:", f"{tic_sum:.2e}"))
    print(_format_row("stream time:", f"{stream_time:.2f} s"))
    print(_format_row("throughput:", f"{actual_mb / stream_time:.1f} MB/s"))
    print(_format_row("RSS before:", f"{rss_before:.0f} MB"))
    print(_format_row("RSS max during streaming:", f"{rss_max_during:.0f} MB"))
    print(_format_row("RSS after:", f"{rss_after:.0f} MB"))

    # ── 3. XIC extraction (streaming) ────────────────────────────────
    print("\n[3/3] XIC extraction via streaming")
    gc.collect()
    t0 = time.perf_counter()
    with StreamingExperiment(str(mzml_path)) as sexp:
        rts, intensities = sexp.get_xic(mz=500.0, tolerance=10.0)
    xic_time = time.perf_counter() - t0
    print(_format_row("XIC points extracted:", f"{len(rts)}"))
    print(_format_row("XIC time:", f"{xic_time:.2f} s"))

    # ── Summary ─────────────────────────────────────────────────────
    summary = {
        "target_mb": target_mb,
        "actual_mb": round(actual_mb, 2),
        "n_spectra": n_streamed,
        "generation_time_s": round(gen_time, 3),
        "stream_time_s": round(stream_time, 3),
        "stream_mb_per_s": round(actual_mb / stream_time, 2),
        "xic_time_s": round(xic_time, 3),
        "rss_before_mb": round(rss_before, 1),
        "rss_max_during_mb": round(rss_max_during, 1),
        "rss_after_mb": round(rss_after, 1),
        "memory_bounded": rss_max_during < actual_mb * 0.5,
    }

    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    for k, v in summary.items():
        print(_format_row(k + ":", str(v)))

    # Sanity gates
    print("\nSanity gates:")
    failures = []
    if not summary["memory_bounded"]:
        failures.append(
            f"RSS {rss_max_during:.0f}MB exceeded 50% of file size {actual_mb:.0f}MB "
            f"(streaming should keep memory sub-linear)"
        )
    if stream_time > target_mb * 2:
        failures.append(f"stream_time {stream_time:.1f}s exceeds 2s/MB threshold")

    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
    else:
        print("  [PASS] all gates")
        print(f"  Memory bounded: RSS peaked at {rss_max_during:.0f}MB for {actual_mb:.0f}MB file")

    if output_json:
        Path(output_json).write_text(json.dumps(summary, indent=2))

    # Cleanup
    try:
        mzml_path.unlink()
    except OSError:
        pass

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-mb", type=float, default=1000.0,
                        help="Target file size in MB (default: 1000 = 1GB)")
    parser.add_argument("--json", help="Write summary JSON to this path")
    args = parser.parse_args()
    run_gb_benchmark(target_mb=args.target_mb, output_json=args.json)


if __name__ == "__main__":
    main()
