#!/usr/bin/env python3
"""
MassKit - Analyze Real Data

Complete analysis pipeline for mzML/mzXML files.
Produces peaks, features, chromatograms, QC metrics, and an HTML report.

Usage:
    python analyze.py sample.mzML
    python analyze.py sample.mzML --output-dir results/
    python analyze.py *.mzML --snr 5 --report
    python analyze.py sample.mzML --annotate PEPTIDER
    python analyze.py --help

Supported formats: .mzML, .mzXML, .mgf
"""

import argparse
import sys
import os
import csv
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))


def parse_args():
    p = argparse.ArgumentParser(
        description="MassKit - Analyze LC-MS data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py sample.mzML
  python analyze.py sample.mzML --output-dir results/
  python analyze.py sample.mzML --snr 5 --ms-level 1
  python analyze.py s1.mzML s2.mzML s3.mzML --quantify --report
  python analyze.py sample.mzML --annotate PEPTIDER --charge 2
  python analyze.py sample.mzML --search library.mgf --min-score 0.7
        """,
    )

    p.add_argument("files", nargs="+", help="Input mzML/mzXML file(s)")
    p.add_argument("-o", "--output-dir", default="masskit_results",
                   help="Output directory (default: masskit_results/)")

    # Processing options
    proc = p.add_argument_group("Processing")
    proc.add_argument("--snr", type=float, default=3.0,
                      help="Signal-to-noise ratio threshold (default: 3.0)")
    proc.add_argument("--ms-level", type=int, default=None,
                      help="Only process this MS level (default: all)")
    proc.add_argument("--rt-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                      help="RT range filter in seconds")
    proc.add_argument("--mz-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                      help="m/z range filter")
    proc.add_argument("--no-smooth", action="store_true",
                      help="Skip spectrum smoothing")
    proc.add_argument("--no-baseline", action="store_true",
                      help="Skip baseline correction")

    # Feature detection
    feat = p.add_argument_group("Feature Detection")
    feat.add_argument("--features", action="store_true",
                      help="Run feature detection (link peaks across scans)")
    feat.add_argument("--mz-tolerance", type=float, default=0.01,
                      help="m/z tolerance for feature linking (default: 0.01 Da)")
    feat.add_argument("--rt-tolerance", type=float, default=30.0,
                      help="RT tolerance for feature linking (default: 30s)")

    # Multi-file quantification
    quant = p.add_argument_group("Quantification (multi-file)")
    quant.add_argument("--quantify", action="store_true",
                       help="Run label-free quantification across files")
    quant.add_argument("--normalize", choices=["median", "quantile", "tic"],
                       default="median", help="Normalization method (default: median)")

    # Spectral search
    search = p.add_argument_group("Spectral Search")
    search.add_argument("--search", metavar="LIBRARY",
                        help="Search against spectral library (MGF file)")
    search.add_argument("--min-score", type=float, default=0.7,
                        help="Minimum match score (default: 0.7)")

    # Annotation
    ann = p.add_argument_group("Annotation")
    ann.add_argument("--annotate", metavar="SEQUENCE",
                     help="Annotate MS2 spectra with peptide sequence")
    ann.add_argument("--charge", type=int, default=2,
                     help="Precursor charge for annotation (default: 2)")

    # XIC
    xic = p.add_argument_group("XIC Extraction")
    xic.add_argument("--xic", type=float, nargs="+", metavar="MZ",
                     help="Extract ion chromatograms at these m/z values")
    xic.add_argument("--xic-tolerance", type=float, default=0.5,
                     help="XIC extraction tolerance in Da (default: 0.5)")

    # Output
    out = p.add_argument_group("Output")
    out.add_argument("--report", action="store_true",
                     help="Generate HTML analysis report")
    out.add_argument("--format", choices=["csv", "tsv"], default="csv",
                     help="Table output format (default: csv)")
    out.add_argument("--quiet", action="store_true",
                     help="Minimal console output")

    return p.parse_args()


def log(msg, quiet=False):
    if not quiet:
        print(msg)


def load_file(filepath, ms_level=None, rt_range=None):
    """Load an mzML or mzXML file."""
    from masskit.io import load_mzml, load_mzxml

    ext = filepath.lower()
    ms_levels = [ms_level] if ms_level else None
    rt_r = tuple(rt_range) if rt_range else None

    if ext.endswith(".mzxml"):
        return load_mzxml(filepath, ms_levels=ms_levels, rt_range=rt_r)
    else:
        return load_mzml(filepath, ms_levels=ms_levels, rt_range=rt_r)


def save_table(rows, headers, filepath, fmt="csv"):
    """Save rows to CSV or TSV."""
    delimiter = "\t" if fmt == "tsv" else ","
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(rows)


def analyze_single_file(filepath, args, output_dir):
    """Full analysis pipeline for one file."""
    from masskit import pick_peaks, smooth_spectrum, correct_baseline, estimate_noise
    from masskit.workflows import PeakPickingWorkflow, FeatureDetectionWorkflow

    basename = os.path.splitext(os.path.basename(filepath))[0]
    log(f"\n{'='*60}", args.quiet)
    log(f"  Analyzing: {filepath}", args.quiet)
    log(f"{'='*60}", args.quiet)

    # Load
    t0 = time.time()
    exp = load_file(filepath, args.ms_level, args.rt_range)
    t_load = time.time() - t0

    n_spectra = exp.spectrum_count
    ms1 = [s for s in exp.spectra if s.ms_level == 1]
    ms2 = [s for s in exp.spectra if s.ms_level == 2]
    log(f"  Loaded: {n_spectra} spectra ({len(ms1)} MS1, {len(ms2)} MS2) in {t_load:.1f}s", args.quiet)

    if n_spectra == 0:
        log("  WARNING: No spectra found. Check file format and filters.", args.quiet)
        return None

    log(f"  RT range: {exp.rt_range[0]:.1f} - {exp.rt_range[1]:.1f} s", args.quiet)
    log(f"  m/z range: {exp.mz_range[0]:.1f} - {exp.mz_range[1]:.1f}", args.quiet)

    results = {
        "file": filepath,
        "n_spectra": n_spectra,
        "n_ms1": len(ms1),
        "n_ms2": len(ms2),
        "rt_range": exp.rt_range,
        "mz_range": exp.mz_range,
    }

    # --- TIC & BPC ---
    tic = exp.generate_tic()
    tic_path = os.path.join(output_dir, f"{basename}_tic.{args.format}")
    save_table(
        [(f"{rt:.2f}", f"{i:.0f}") for rt, i in zip(tic.rt, tic.intensity)],
        ["rt_seconds", "intensity"],
        tic_path, args.format,
    )
    log(f"  TIC saved: {tic_path}", args.quiet)

    # --- XIC extraction ---
    if args.xic:
        for target_mz in args.xic:
            xic = exp.generate_xic(target_mz=target_mz, tolerance=args.xic_tolerance)
            xic_path = os.path.join(output_dir, f"{basename}_xic_{target_mz:.2f}.{args.format}")
            save_table(
                [(f"{rt:.2f}", f"{i:.0f}") for rt, i in zip(xic.rt, xic.intensity)],
                ["rt_seconds", "intensity"],
                xic_path, args.format,
            )
            log(f"  XIC @ m/z {target_mz}: saved to {xic_path}", args.quiet)

    # --- Peak picking ---
    log(f"  Peak picking (SNR>={args.snr})...", args.quiet)
    workflow = PeakPickingWorkflow(
        smooth=not args.no_smooth,
        baseline_correct=not args.no_baseline,
        min_snr=args.snr,
    )

    all_peaks = []
    for spec in exp.spectra:
        peaks = workflow.process(spec)
        for peak in peaks:
            peak.rt = spec.rt
        all_peaks.extend(list(peaks))

    # Save peaks
    peak_rows = []
    for p in sorted(all_peaks, key=lambda x: x.intensity, reverse=True):
        peak_rows.append([
            f"{p.mz:.6f}", f"{p.intensity:.0f}", f"{p.rt:.2f}",
            f"{p.snr:.1f}", f"{p.fwhm:.4f}" if p.fwhm else "",
            f"{p.area:.0f}" if p.area else "",
        ])
    peaks_path = os.path.join(output_dir, f"{basename}_peaks.{args.format}")
    save_table(peak_rows, ["mz", "intensity", "rt", "snr", "fwhm", "area"],
               peaks_path, args.format)
    log(f"  Peaks: {len(all_peaks)} detected -> {peaks_path}", args.quiet)
    results["n_peaks"] = len(all_peaks)

    # --- Feature detection ---
    if args.features:
        log(f"  Feature detection (mz_tol={args.mz_tolerance}, rt_tol={args.rt_tolerance}s)...", args.quiet)
        feat_workflow = FeatureDetectionWorkflow(
            mz_tolerance=args.mz_tolerance,
            rt_tolerance=args.rt_tolerance,
            min_snr=args.snr,
            smooth=not args.no_smooth,
            baseline_correct=not args.no_baseline,
        )
        feature_map = feat_workflow.process(exp)

        feat_rows = []
        for f in sorted(feature_map, key=lambda x: x.intensity, reverse=True):
            feat_rows.append([
                f"{f.mz:.6f}", f"{f.intensity:.0f}", f"{f.rt:.2f}",
                f"{f.rt_min:.2f}", f"{f.rt_max:.2f}",
                f"{f.volume:.0f}" if f.volume else "",
                f"{f.quality:.3f}" if f.quality else "",
            ])
        feat_path = os.path.join(output_dir, f"{basename}_features.{args.format}")
        save_table(feat_rows,
                   ["mz", "intensity", "rt", "rt_start", "rt_end", "volume", "quality"],
                   feat_path, args.format)
        log(f"  Features: {len(list(feature_map))} detected -> {feat_path}", args.quiet)
        results["n_features"] = len(list(feature_map))

    # --- Spectral library search ---
    if args.search and ms2:
        log(f"  Searching against library: {args.search}...", args.quiet)
        from masskit.spectral_matching import SpectralLibrary
        lib = SpectralLibrary()
        lib.load_mgf(args.search)
        log(f"  Library loaded: {len(lib)} entries", args.quiet)

        match_rows = []
        for i, spec in enumerate(ms2):
            prec_mz = spec.precursors[0].mz if spec.precursors else 0.0
            matches = lib.search(
                spec.mz, spec.intensity,
                query_precursor_mz=prec_mz,
                min_score=args.min_score,
                top_n=1,
            )
            for m in matches:
                match_rows.append([
                    i, f"{spec.rt:.2f}", f"{prec_mz:.4f}",
                    m.library_name, f"{m.score:.4f}", m.matched_peaks,
                ])

        match_path = os.path.join(output_dir, f"{basename}_matches.{args.format}")
        save_table(match_rows,
                   ["scan", "rt", "precursor_mz", "library_match", "score", "matched_peaks"],
                   match_path, args.format)
        log(f"  Matches: {len(match_rows)} hits (score>={args.min_score}) -> {match_path}", args.quiet)
        results["n_matches"] = len(match_rows)

    # --- Spectrum annotation ---
    if args.annotate and ms2:
        log(f"  Annotating MS2 spectra with sequence: {args.annotate}...", args.quiet)
        from masskit.annotation import annotate_spectrum, format_annotation_table

        ann_rows = []
        for i, spec in enumerate(ms2[:50]):  # annotate up to 50 spectra
            ann = annotate_spectrum(spec, args.annotate, precursor_charge=args.charge)
            for a in ann.annotations:
                ann_rows.append([
                    i, f"{spec.rt:.2f}", a.label,
                    f"{a.mz_observed:.4f}", f"{a.mz_theoretical:.4f}",
                    f"{a.error_ppm:.1f}", f"{a.intensity:.0f}",
                ])

            if i == 0 and ann.annotations:
                log(f"  Scan 0: {ann.n_matched}/{ann.n_total_peaks} matched, "
                    f"coverage={ann.coverage:.0%}", args.quiet)

        ann_path = os.path.join(output_dir, f"{basename}_annotations.{args.format}")
        save_table(ann_rows,
                   ["scan", "rt", "ion", "mz_obs", "mz_theo", "error_ppm", "intensity"],
                   ann_path, args.format)
        log(f"  Annotations: {len(ann_rows)} fragment ions -> {ann_path}", args.quiet)

    # --- QC metrics ---
    log(f"  Computing QC metrics...", args.quiet)
    from masskit.qc import compute_qc_metrics
    qc = compute_qc_metrics(exp)
    qc_data = {
        "file": filepath,
        "n_spectra": n_spectra,
        "ms1_count": qc.ms1_count,
        "ms2_count": qc.ms2_count,
        "tic_cv": round(qc.tic_cv, 4),
        "median_peak_width": round(qc.median_peak_width, 2),
        "peak_capacity": qc.peak_capacity,
        "dynamic_range": round(qc.dynamic_range, 1),
    }
    qc_path = os.path.join(output_dir, f"{basename}_qc.json")
    with open(qc_path, "w") as f:
        json.dump(qc_data, f, indent=2)
    log(f"  QC: TIC_CV={qc.tic_cv:.4f}, peak_capacity={qc.peak_capacity} -> {qc_path}", args.quiet)
    results["qc"] = qc_data
    results["qc_obj"] = qc

    return results


def run_quantification(all_results, args, output_dir):
    """Run label-free quantification across multiple files."""
    from masskit.quantification import median_normalization, quantile_normalization, tic_normalization

    log(f"\n{'='*60}", args.quiet)
    log(f"  Multi-file Quantification ({len(all_results)} files)", args.quiet)
    log(f"{'='*60}", args.quiet)

    # For quantification, we need features from each file.
    # Re-run feature detection if not already done.
    log("  Note: Full consensus map quantification requires feature alignment.", args.quiet)
    log("  Use the Python API for advanced multi-file workflows:", args.quiet)
    log("    from masskit.quantification import FeatureAlignment, ConsensusMap", args.quiet)
    log("    from masskit.statistics import pca, anova", args.quiet)


def generate_report(all_results, args, output_dir):
    """Generate HTML analysis report."""
    from masskit.reporting import ReportBuilder, ReportConfig

    log(f"\n  Generating HTML report...", args.quiet)

    builder = ReportBuilder(ReportConfig(
        title="MassKit Analysis Report",
    ))

    for r in all_results:
        if r is None:
            continue
        builder.add_summary(
            n_spectra=r.get("n_spectra", 0),
            n_ms1=r.get("n_ms1", 0),
            n_ms2=r.get("n_ms2", 0),
            n_features=r.get("n_features", 0),
            rt_range=r.get("rt_range"),
            mz_range=r.get("mz_range"),
            extra={"File": r["file"], "Peaks": r.get("n_peaks", "N/A")},
        )
        if "qc_obj" in r:
            builder.add_qc_section(r["qc_obj"])

    report_path = os.path.join(output_dir, "analysis_report.html")
    builder.save_html(report_path)
    log(f"  Report saved: {report_path}", args.quiet)
    log(f"  Open in browser: file://{os.path.abspath(report_path)}", args.quiet)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Verify files exist
    for f in args.files:
        if not os.path.exists(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)

    log(f"MassKit Analysis", args.quiet)
    log(f"  Files: {len(args.files)}", args.quiet)
    log(f"  Output: {args.output_dir}/", args.quiet)
    t_start = time.time()

    # Analyze each file
    all_results = []
    for filepath in args.files:
        try:
            result = analyze_single_file(filepath, args, args.output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR processing {filepath}: {e}")
            all_results.append(None)

    # Multi-file quantification
    if args.quantify and len(args.files) > 1:
        run_quantification([r for r in all_results if r], args, args.output_dir)

    # Report
    if args.report:
        generate_report([r for r in all_results if r], args, args.output_dir)

    # Summary
    t_total = time.time() - t_start
    log(f"\n{'='*60}", args.quiet)
    log(f"  Analysis complete in {t_total:.1f}s", args.quiet)
    log(f"  Results in: {os.path.abspath(args.output_dir)}/", args.quiet)
    log(f"{'='*60}", args.quiet)

    # List output files
    if not args.quiet:
        print(f"\n  Output files:")
        for f in sorted(os.listdir(args.output_dir)):
            fpath = os.path.join(args.output_dir, f)
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"    {f:<40} {size_str}")


if __name__ == "__main__":
    main()
