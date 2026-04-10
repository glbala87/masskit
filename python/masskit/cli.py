"""
Command-line interface for MassKit toolkit.

Provides subcommands for common LC-MS data analysis tasks.

Usage:
    masskit info <file>           Show file information
    masskit peaks <file>          Pick peaks and export
    masskit convert <file> -o out Convert between formats
    masskit xic <file> --mz 500   Extract ion chromatogram
    masskit quantify <files...>   Label-free quantification
    masskit search <file> --lib   Spectral library search
    masskit qc <files...>         Quality control report
"""

import argparse
import sys
import os
import json
import csv
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _load_experiment(filepath: str):
    """Load experiment from mzML or mzXML file."""
    from .io import load_mzml, load_mzxml
    from .exceptions import FileFormatError
    from .validation import validate_file_path

    validate_file_path(filepath, must_exist=True)

    filepath_lower = filepath.lower()
    if filepath_lower.endswith(".mzxml"):
        return load_mzxml(filepath)
    elif filepath_lower.endswith(".mzml"):
        return load_mzml(filepath)
    else:
        # Try mzML first, then mzXML
        try:
            return load_mzml(filepath)
        except (ET.ParseError, KeyError, ValueError) as e:
            logger.debug("mzML parse failed for %s (%s), trying mzXML", filepath, e)
            try:
                return load_mzxml(filepath)
            except (ET.ParseError, KeyError, ValueError) as e2:
                raise FileFormatError(
                    filepath,
                    "Could not parse as mzML or mzXML. "
                    f"mzML error: {e}, mzXML error: {e2}"
                ) from e2


def cmd_info(args):
    """Show file information."""
    exp = _load_experiment(args.file)

    print(f"File: {args.file}")
    print(f"Spectra: {exp.num_spectra}")

    # MS level breakdown
    levels = {}
    for spec in exp.spectra:
        level = spec.ms_level
        levels[level] = levels.get(level, 0) + 1

    for level in sorted(levels.keys()):
        print(f"  MS{level}: {levels[level]} spectra")

    if exp.num_spectra > 0:
        rt_min = min(s.rt for s in exp.spectra)
        rt_max = max(s.rt for s in exp.spectra)
        print(f"RT range: {rt_min:.2f} - {rt_max:.2f} s")

        # m/z range from MS1
        ms1 = [s for s in exp.spectra if s.ms_level == 1]
        if ms1:
            import numpy as np
            all_mz_min = min(np.min(s.mz) for s in ms1 if len(s.mz) > 0)
            all_mz_max = max(np.max(s.mz) for s in ms1 if len(s.mz) > 0)
            print(f"m/z range: {all_mz_min:.4f} - {all_mz_max:.4f}")

    if exp.chromatograms:
        print(f"Chromatograms: {len(exp.chromatograms)}")

    if args.verbose:
        print("\nFirst 10 spectra:")
        for i, spec in enumerate(exp.spectra[:10]):
            print(
                f"  [{i}] MS{spec.ms_level} RT={spec.rt:.2f}s "
                f"peaks={len(spec.mz)} "
                f"TIC={spec.tic:.0f}"
            )


def cmd_peaks(args):
    """Pick peaks from a file."""
    from .algorithms import pick_peaks

    exp = _load_experiment(args.file)

    ms_level = args.ms_level
    spectra = [s for s in exp.spectra if s.ms_level == ms_level]

    if not spectra:
        print(f"No MS{ms_level} spectra found.", file=sys.stderr)
        return 1

    all_peaks = []
    for spec in spectra:
        peaks = pick_peaks(spec, min_snr=args.snr, min_intensity=args.min_intensity)
        for peak in peaks:
            all_peaks.append(peak)

    print(f"Found {len(all_peaks)} peaks across {len(spectra)} spectra", file=sys.stderr)

    # Output
    output = sys.stdout
    if args.output:
        output = open(args.output, "w", newline="")

    writer = csv.writer(output, delimiter="\t" if args.tsv else ",")
    writer.writerow(["mz", "rt", "intensity", "area", "snr", "fwhm", "charge"])

    for peak in all_peaks:
        writer.writerow([
            f"{peak.mz:.6f}",
            f"{peak.rt:.2f}",
            f"{peak.intensity:.2f}",
            f"{peak.area:.2f}",
            f"{peak.snr:.2f}",
            f"{peak.fwhm_mz:.6f}",
            peak.charge,
        ])

    if args.output:
        output.close()
        print(f"Written to {args.output}", file=sys.stderr)


def cmd_convert(args):
    """Convert between file formats."""
    exp = _load_experiment(args.file)

    output = args.output
    if not output:
        base = Path(args.file).stem
        if args.format == "mztab":
            output = f"{base}.mztab"
        else:
            output = f"{base}_converted.mzML"

    if args.format == "mztab":
        from .io import save_mztab
        from .workflows import FeatureDetectionWorkflow

        workflow = FeatureDetectionWorkflow()
        features = workflow.process(exp)
        save_mztab(features, output)
        print(f"Converted to mzTab: {output} ({len(features)} features)")
    else:
        print(f"Format '{args.format}' export not yet supported.", file=sys.stderr)
        return 1


def cmd_xic(args):
    """Extract ion chromatogram."""
    exp = _load_experiment(args.file)

    xic = exp.generate_xic(args.mz, args.tolerance, level=args.ms_level)

    if args.plot:
        from .visualization import plot_chromatogram
        import matplotlib.pyplot as plt

        plot_chromatogram(xic, title=f"XIC @ m/z {args.mz:.4f} ± {args.tolerance}")
        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {args.output}")
        else:
            plt.show()
    else:
        output = sys.stdout
        if args.output:
            output = open(args.output, "w", newline="")

        writer = csv.writer(output, delimiter="\t")
        writer.writerow(["rt", "intensity"])
        for rt, intensity in zip(xic.rt, xic.intensity):
            writer.writerow([f"{rt:.4f}", f"{intensity:.2f}"])

        if args.output:
            output.close()
            print(f"Written to {args.output}", file=sys.stderr)


def cmd_quantify(args):
    """Label-free quantification across multiple files."""
    from .quantification import FeatureAlignment, ConsensusMap
    from .workflows import FeatureDetectionWorkflow
    from .parallel import BatchProcessor

    files = args.files
    sample_names = [Path(f).stem for f in files]

    print(f"Processing {len(files)} files...", file=sys.stderr)

    # Detect features in each file
    feature_maps = []
    for i, filepath in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {Path(filepath).name}...", file=sys.stderr)
        exp = _load_experiment(filepath)
        workflow = FeatureDetectionWorkflow()
        features = workflow.process(exp)
        feature_maps.append(features)
        print(f"    {len(features)} features detected", file=sys.stderr)

    # Align features
    print("Aligning features...", file=sys.stderr)
    aligner = FeatureAlignment(
        mz_tolerance=args.mz_tolerance,
        rt_tolerance=args.rt_tolerance,
    )
    consensus = aligner.align(feature_maps, sample_names)

    # Filter and normalize
    consensus = consensus.filter_by_presence(args.min_presence)
    if args.normalize:
        consensus = consensus.normalize(method=args.normalize)

    print(f"Consensus: {consensus.n_features} features x {consensus.n_samples} samples",
          file=sys.stderr)

    # Output
    output = args.output or "quantification.tsv"
    df = consensus.to_dataframe()
    df.to_csv(output, sep="\t")
    print(f"Written to {output}", file=sys.stderr)


def cmd_search(args):
    """Spectral library search."""
    from .spectral_matching import SpectralLibrary
    from .algorithms import pick_peaks

    # Load library
    lib = SpectralLibrary()
    lib_path = args.library
    if lib_path.lower().endswith(".msp"):
        count = lib.load_msp(lib_path)
    else:
        count = lib.load_mgf(lib_path)
    print(f"Library: {count} spectra loaded from {lib_path}", file=sys.stderr)

    # Load query file
    exp = _load_experiment(args.file)
    ms2_spectra = [s for s in exp.spectra if s.ms_level == 2]

    if not ms2_spectra:
        print("No MS2 spectra found.", file=sys.stderr)
        return 1

    print(f"Searching {len(ms2_spectra)} MS2 spectra...", file=sys.stderr)

    output = sys.stdout
    if args.output:
        output = open(args.output, "w", newline="")

    writer = csv.writer(output, delimiter="\t")
    writer.writerow(["scan", "precursor_mz", "match_name", "score", "matched_peaks"])

    total_matches = 0
    for spec in ms2_spectra:
        precursor_mz = 0.0
        if spec.precursors:
            prec = spec.precursors[0]
            # Support both Precursor dataclass and dict (from MGF)
            precursor_mz = prec.mz if hasattr(prec, "mz") else prec.get("mz", 0.0)

        matches = lib.search(
            spec.mz, spec.intensity,
            query_precursor_mz=precursor_mz,
            method=args.method,
            tolerance=args.tolerance,
            min_score=args.min_score,
            top_n=args.top_n,
            precursor_tolerance=args.precursor_tolerance,
        )

        for match in matches:
            writer.writerow([
                spec.index,
                f"{precursor_mz:.4f}",
                match.library_name,
                f"{match.score:.4f}",
                match.matched_peaks,
            ])
            total_matches += 1

    if args.output:
        output.close()

    print(f"Found {total_matches} matches", file=sys.stderr)


def cmd_qc(args):
    """Quality control report."""
    import numpy as np

    files = args.files

    print(f"{'File':<40} {'Spectra':>8} {'MS1':>6} {'MS2':>6} "
          f"{'RT range':>14} {'TIC median':>12} {'TIC CV%':>8}")
    print("-" * 100)

    for filepath in files:
        try:
            exp = _load_experiment(filepath)
            name = Path(filepath).name
            if len(name) > 38:
                name = name[:35] + "..."

            n_total = exp.num_spectra
            ms1 = [s for s in exp.spectra if s.ms_level == 1]
            ms2 = [s for s in exp.spectra if s.ms_level == 2]

            rt_min = min(s.rt for s in exp.spectra) if exp.spectra else 0
            rt_max = max(s.rt for s in exp.spectra) if exp.spectra else 0
            rt_str = f"{rt_min:.1f}-{rt_max:.1f}s"

            tic_values = np.array([s.tic for s in ms1]) if ms1 else np.array([0])
            tic_median = np.median(tic_values)
            tic_cv = (np.std(tic_values) / np.mean(tic_values) * 100
                      if np.mean(tic_values) > 0 else 0)

            print(f"{name:<40} {n_total:>8} {len(ms1):>6} {len(ms2):>6} "
                  f"{rt_str:>14} {tic_median:>12.0f} {tic_cv:>7.1f}%")
        except (OSError, ValueError, ET.ParseError) as e:
            logger.error("Failed to process %s: %s", filepath, e)
            print(f"{Path(filepath).name:<40} ERROR: {e}", file=sys.stderr)

    if args.plot and len(files) > 1:
        _plot_qc(files, args.output)


def _plot_qc(files: List[str], output: Optional[str] = None):
    """Generate QC plots."""
    import numpy as np
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for QC plots", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    tic_data = []
    labels = []

    for filepath in files:
        try:
            exp = _load_experiment(filepath)
            ms1 = [s for s in exp.spectra if s.ms_level == 1]
            if ms1:
                rts = np.array([s.rt for s in ms1])
                tics = np.array([s.tic for s in ms1])
                tic_data.append((rts, tics))
                labels.append(Path(filepath).stem)
        except (OSError, ValueError, ET.ParseError) as e:
            logger.warning("Skipping %s in QC plot: %s", filepath, e)
            continue

    if not tic_data:
        return

    # TIC overlay
    ax = axes[0, 0]
    for (rts, tics), label in zip(tic_data, labels):
        ax.plot(rts, tics, linewidth=0.5, label=label)
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("TIC")
    ax.set_title("TIC Overlay")
    ax.legend(fontsize=6)

    # TIC boxplot
    ax = axes[0, 1]
    ax.boxplot([tics for _, tics in tic_data], labels=labels)
    ax.set_ylabel("TIC")
    ax.set_title("TIC Distribution")
    ax.tick_params(axis="x", rotation=45)

    # Spectrum count comparison
    ax = axes[1, 0]
    counts = [len(tics) for _, tics in tic_data]
    ax.bar(range(len(labels)), counts, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MS1 Spectra Count")
    ax.set_title("Spectra per Run")

    # TIC CV
    ax = axes[1, 1]
    cvs = []
    for _, tics in tic_data:
        cv = np.std(tics) / np.mean(tics) * 100 if np.mean(tics) > 0 else 0
        cvs.append(cv)
    colors = ["green" if cv < 20 else "orange" if cv < 40 else "red" for cv in cvs]
    ax.bar(range(len(labels)), cvs, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("TIC CV (%)")
    ax.set_title("TIC Stability")
    ax.axhline(20, color="green", linestyle="--", alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"QC plot saved to {output}", file=sys.stderr)
    else:
        plt.show()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="masskit",
        description="MassKit - LC-MS Data Analysis Toolkit",
    )
    parser.add_argument("--version", action="version", version="masskit 1.0.0")
    parser.add_argument(
        "--config",
        help="Path to JSON/YAML config file (LCMSConfig). "
             "Values can also be supplied via MASSKIT_* env vars.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    p = subparsers.add_parser("info", help="Show file information")
    p.add_argument("file", help="Input file (mzML or mzXML)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # peaks
    p = subparsers.add_parser("peaks", help="Pick peaks")
    p.add_argument("file", help="Input file")
    p.add_argument("-o", "--output", help="Output file")
    p.add_argument("--snr", type=float, default=3.0, help="Min SNR (default: 3.0)")
    p.add_argument("--min-intensity", type=float, default=0.0, help="Min intensity")
    p.add_argument("--ms-level", type=int, default=1, help="MS level (default: 1)")
    p.add_argument("--tsv", action="store_true", help="TSV output (default: CSV)")

    # convert
    p = subparsers.add_parser("convert", help="Convert file format")
    p.add_argument("file", help="Input file")
    p.add_argument("-o", "--output", help="Output file")
    p.add_argument("-f", "--format", default="mztab",
                   choices=["mztab"], help="Output format")

    # xic
    p = subparsers.add_parser("xic", help="Extract ion chromatogram")
    p.add_argument("file", help="Input file")
    p.add_argument("--mz", type=float, required=True, help="Target m/z")
    p.add_argument("--tolerance", type=float, default=0.01, help="m/z tolerance (Da)")
    p.add_argument("--ms-level", type=int, default=1, help="MS level")
    p.add_argument("-o", "--output", help="Output file")
    p.add_argument("--plot", action="store_true", help="Generate plot")

    # quantify
    p = subparsers.add_parser("quantify", help="Label-free quantification")
    p.add_argument("files", nargs="+", help="Input files")
    p.add_argument("-o", "--output", help="Output file")
    p.add_argument("--mz-tolerance", type=float, default=0.01, help="m/z tolerance")
    p.add_argument("--rt-tolerance", type=float, default=30.0, help="RT tolerance (s)")
    p.add_argument("--min-presence", type=float, default=0.5,
                   help="Min fraction of samples (default: 0.5)")
    p.add_argument("--normalize", choices=["median", "quantile", "tic"],
                   help="Normalization method")

    # search
    p = subparsers.add_parser("search", help="Spectral library search")
    p.add_argument("file", help="Query file (mzML/mzXML)")
    p.add_argument("--library", "-l", required=True, help="Library file (MGF/MSP)")
    p.add_argument("-o", "--output", help="Output file")
    p.add_argument("--method", default="cosine",
                   choices=["cosine", "modified_cosine", "entropy"])
    p.add_argument("--tolerance", type=float, default=0.02, help="Fragment tolerance")
    p.add_argument("--min-score", type=float, default=0.7, help="Min score")
    p.add_argument("--top-n", type=int, default=1, help="Top N matches per query")
    p.add_argument("--precursor-tolerance", type=float, default=0.5,
                   help="Precursor tolerance (Da)")

    # qc
    p = subparsers.add_parser("qc", help="Quality control report")
    p.add_argument("files", nargs="+", help="Input files")
    p.add_argument("--plot", action="store_true", help="Generate QC plots")
    p.add_argument("-o", "--output", help="Output plot file")

    return parser


def _setup_logging(level_name: str = "WARNING") -> None:
    """Configure logging for the CLI."""
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_cli_config(args):
    """Load LCMSConfig from --config flag, env vars, or defaults."""
    from .config import LCMSConfig
    if getattr(args, "config", None):
        config = LCMSConfig.from_file(args.config)
    else:
        config = LCMSConfig.discover()
    config.validate()
    return config


def _apply_config_defaults(args, config):
    """Backfill argparse args with config defaults where the CLI didn't override."""
    # peak picking
    if hasattr(args, "snr") and args.snr == 3.0:
        args.snr = config.peak_picking.min_snr
    if hasattr(args, "min_intensity") and args.min_intensity == 0.0:
        args.min_intensity = config.peak_picking.min_intensity
    # quantification
    if hasattr(args, "mz_tolerance") and args.mz_tolerance == 0.01:
        args.mz_tolerance = config.quantification.mz_tolerance
    if hasattr(args, "rt_tolerance") and args.rt_tolerance == 30.0:
        args.rt_tolerance = config.quantification.rt_tolerance
    if hasattr(args, "min_presence") and args.min_presence == 0.5:
        args.min_presence = config.quantification.min_presence
    return args


def main(argv: Optional[List[str]] = None):
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Load config (file > env > defaults), then apply to argparse defaults
    try:
        config = _load_cli_config(args)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    log_level = args.log_level or config.log_level
    _setup_logging(log_level)

    args = _apply_config_defaults(args, config)

    commands = {
        "info": cmd_info,
        "peaks": cmd_peaks,
        "convert": cmd_convert,
        "xic": cmd_xic,
        "quantify": cmd_quantify,
        "search": cmd_search,
        "qc": cmd_qc,
    }

    func = commands.get(args.command)
    if func:
        return func(args) or 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
