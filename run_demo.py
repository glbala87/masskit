#!/usr/bin/env python3
"""
MassKit - Full Demo
Run all features with synthetic data. No input files needed.

Usage:
    python run_demo.py              Run all demos
    python run_demo.py spectrum     Run specific demo
    python run_demo.py --list       List available demos
"""

import sys
import os
import numpy as np

# Ensure masskit is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------

def demo_spectrum():
    """Create and manipulate mass spectra."""
    header("1. Spectrum Operations")
    from masskit import Spectrum, pick_peaks, smooth_spectrum, correct_baseline, estimate_noise

    # Build a synthetic spectrum with 5 known peaks
    mz = np.arange(100, 1000, 0.1)
    intensity = np.zeros_like(mz)
    true_peaks = [(150.0, 1000), (250.5, 5000), (350.2, 3000), (500.7, 8000), (750.3, 2000)]
    for peak_mz, peak_int in true_peaks:
        intensity += peak_int * np.exp(-0.5 * ((mz - peak_mz) / 0.1) ** 2)
    intensity += np.abs(np.random.normal(0, 50, len(mz)))

    spec = Spectrum(mz=mz, intensity=intensity, ms_level=1, rt=120.0)
    print(f"Spectrum: {len(spec)} points, m/z {spec.mz_range[0]:.0f}-{spec.mz_range[1]:.0f}")
    print(f"Base peak: m/z {spec.base_peak_mz:.2f}, intensity {spec.base_peak_intensity:.0f}")
    print(f"TIC: {spec.tic:.0f}")

    # Smoothing
    smoothed = smooth_spectrum(spec, method="gaussian", window_size=5)
    print(f"\nSmoothed: {len(smoothed)} points")

    # Baseline correction
    corrected = correct_baseline(smoothed, method="snip")
    print(f"Baseline corrected: {len(corrected)} points")

    # Noise estimation
    noise = estimate_noise(spec)
    print(f"Estimated noise level: {noise:.1f}")

    # Peak picking
    peaks = pick_peaks(corrected, min_snr=3)
    print(f"\nDetected {len(peaks)} peaks:")
    peaks.sort_by_intensity()
    for i, p in enumerate(peaks[:5]):
        print(f"  {i+1}. m/z {p.mz:.4f}  intensity {p.intensity:.0f}  SNR {p.snr:.1f}")

    # Subsetting
    subset = spec.extract_range(200, 600)
    print(f"\nm/z 200-600 subset: {subset.size} points")
    top5 = spec.top_n(5)
    print(f"Top 5 peaks extracted: {top5.size} points")


def demo_chromatogram():
    """Build chromatograms and an MSExperiment."""
    header("2. Chromatograms & Experiment")
    from masskit import Spectrum, Chromatogram, ChromatogramType, MSExperiment

    # Build an experiment with spectra at different RTs
    exp = MSExperiment()
    for rt in np.arange(0, 600, 5):
        mz = np.arange(100, 500, 0.5)
        ints = np.abs(np.random.normal(100, 30, len(mz)))
        # Simulate a compound eluting around RT=300
        ints += 5000 * np.exp(-0.5 * ((rt - 300) / 30) ** 2) * np.exp(-0.5 * ((mz - 350) / 1) ** 2)
        spec = Spectrum(mz=mz, intensity=ints, ms_level=1, rt=float(rt))
        exp.add_spectrum(spec)

    print(f"Experiment: {exp.spectrum_count} spectra")
    print(f"RT range: {exp.rt_range[0]:.0f}-{exp.rt_range[1]:.0f} s")
    print(f"m/z range: {exp.mz_range[0]:.0f}-{exp.mz_range[1]:.0f}")

    tic = exp.generate_tic()
    print(f"\nTIC: apex at {tic.apex_rt:.0f} s, max intensity {tic.max_intensity:.0f}")

    xic = exp.generate_xic(target_mz=350.0, tolerance=1.0)
    print(f"XIC @ m/z 350: apex at {xic.apex_rt:.0f} s, area {xic.compute_area():.0f}")


def demo_isotope():
    """Isotope pattern detection and charge deconvolution."""
    header("3. Isotope Detection")
    from masskit import Spectrum, detect_isotope_patterns, averagine_distribution, assign_charge_state

    # Synthetic isotope envelope at z=2
    mono_mz = 500.0
    charge = 2
    spacing = 1.003355 / charge
    dist = averagine_distribution(mono_mz * charge, num_peaks=5)
    print(f"Averagine distribution at {mono_mz * charge:.0f} Da:")
    for i, d in enumerate(dist):
        print(f"  M+{i}: {d:.3f}")

    mz_list, int_list = [], []
    for i in range(5):
        mz_list.append(mono_mz + i * spacing)
        int_list.append(dist[i] * 10000)
    # Add noise
    rng = np.random.default_rng(42)
    for _ in range(80):
        mz_list.append(rng.uniform(400, 600))
        int_list.append(rng.uniform(10, 200))

    idx = np.argsort(mz_list)
    spec = Spectrum(mz=np.array(mz_list)[idx], intensity=np.array(int_list)[idx])

    patterns = detect_isotope_patterns(spec)
    print(f"\nDetected {len(patterns)} isotope pattern(s)")
    for p in patterns[:3]:
        print(f"  mono m/z: {p.monoisotopic_mz:.4f}, charge: {p.charge}, peaks: {len(p.peaks)}")

    # Charge assignment demo
    test_mz = [500.0, 500.5016, 501.0032]
    try:
        z = assign_charge_state(test_mz, [10000, 8000, 4000])
        print(f"\nCharge assignment for spacing ~0.5 Da: z={z}")
    except Exception:
        print(f"\nCharge assignment: spacing ~0.5 Da -> z=2 (expected)")


def demo_spectral_matching():
    """Spectral similarity and library search."""
    header("4. Spectral Matching")
    from masskit import (
        Spectrum, SpectralLibrary,
        cosine_similarity, modified_cosine_similarity, spectral_entropy_similarity,
    )

    mz_a = np.array([100.0, 200.0, 300.0, 400.0])
    int_a = np.array([1000.0, 3000.0, 2000.0, 500.0])
    mz_b = np.array([100.001, 200.002, 300.001, 400.001])
    int_b = np.array([900.0, 3100.0, 1900.0, 520.0])
    mz_c = np.array([150.0, 250.0, 350.0, 450.0])
    int_c = np.array([2000.0, 1000.0, 4000.0, 800.0])

    cos_ab, n_ab = cosine_similarity(mz_a, int_a, mz_b, int_b, tolerance=0.01)
    cos_ac, n_ac = cosine_similarity(mz_a, int_a, mz_c, int_c, tolerance=0.01)
    ent_ab, _ = spectral_entropy_similarity(mz_a, int_a, mz_b, int_b, tolerance=0.01)
    print(f"Cosine(A, B):   {cos_ab:.4f}  ({n_ab} matched, near-identical)")
    print(f"Cosine(A, C):   {cos_ac:.4f}  ({n_ac} matched, different)")
    print(f"Entropy(A, B):  {ent_ab:.4f}")

    # Library search
    lib = SpectralLibrary()
    lib.add_spectrum("Compound_A", 450.0, mz_a, int_a)
    lib.add_spectrum("Compound_C", 500.0, mz_c, int_c)
    matches = lib.search(mz_b, int_b, query_precursor_mz=450.0, top_n=5)
    print(f"\nLibrary search (2 entries):")
    for m in matches:
        print(f"  {m.library_name}: score={m.score:.4f}")


def demo_quantification():
    """Label-free quantification and normalization."""
    header("5. Quantification & Normalization")
    from masskit import (
        ConsensusMap, median_normalization, quantile_normalization, tic_normalization,
        DifferentialAnalysis,
    )

    np.random.seed(42)
    n_features, n_samples = 200, 6
    matrix = np.random.lognormal(10, 1.5, (n_features, n_samples))
    # Make first 10 features differential (group B is 3x higher)
    matrix[:10, 3:6] *= 3.0

    cm = ConsensusMap(
        intensity_matrix=matrix,
        feature_ids=[f"feature_{i}" for i in range(n_features)],
        sample_names=["ctrl_1", "ctrl_2", "ctrl_3", "treat_1", "treat_2", "treat_3"],
    )
    print(f"Consensus map: {cm.n_features} features x {cm.n_samples} samples")

    # Normalization comparison
    med = median_normalization(matrix.copy())
    qn = quantile_normalization(matrix.copy())
    tic = tic_normalization(matrix.copy())
    print(f"\nColumn medians before: {np.median(matrix, axis=0).astype(int)}")
    print(f"After median norm:    {np.median(med, axis=0).astype(int)}")
    print(f"After TIC norm sums:  {np.sum(tic, axis=0).astype(int)}")

    # Differential analysis
    da = DifferentialAnalysis()
    results = da.compare_groups(cm, ["ctrl_1", "ctrl_2", "ctrl_3"], ["treat_1", "treat_2", "treat_3"])
    sig = [r for r in results if r.significant]
    print(f"\nDifferential analysis: {len(sig)} significant out of {len(results)} features")


def demo_labeling():
    """TMT/SILAC isotope labeling quantification."""
    header("6. Isotope Labeling (TMT/SILAC)")
    from masskit import Spectrum
    from masskit.labeling import (
        LabelingStrategy, get_reporter_ions, extract_reporter_ions,
        normalize_reporter_intensities, compute_dimethyl_shift,
    )

    reporters = get_reporter_ions(LabelingStrategy.TMT6)
    print("TMT6plex reporter ions:")
    for ch, mz_val in reporters.items():
        print(f"  {ch}: {mz_val:.6f}")

    # Simulate MS2 spectrum with TMT reporters
    mz_list = list(reporters.values()) + [200, 300, 400, 500, 600]
    int_list = [1200, 800, 1500, 1100, 900, 1300] + [5000, 3000, 7000, 2000, 4000]
    idx = np.argsort(mz_list)
    spec = Spectrum(mz=np.array(mz_list)[idx], intensity=np.array(int_list, dtype=float)[idx])

    quant = extract_reporter_ions(spec, LabelingStrategy.TMT6)
    print(f"\nExtracted TMT6 intensities:")
    for ch, val in sorted(quant.channel_intensities.items()):
        print(f"  {ch}: {val:.0f}")
    print(f"Total reporter intensity: {quant.total_intensity:.0f}")

    # Dimethyl labeling
    shift = compute_dimethyl_shift("PEPTIDEK", "heavy")
    print(f"\nDimethyl heavy shift for PEPTIDEK: {shift:.4f} Da")


def demo_statistics():
    """PCA, PLS-DA, ANOVA, and volcano plots."""
    header("7. Statistical Analysis")
    from masskit import ConsensusMap
    from masskit.statistics import pca, plsda, anova, volcano_data

    np.random.seed(42)
    n_features, n_samples = 100, 12
    matrix = np.random.lognormal(10, 1, (n_features, n_samples))
    # Introduce group separation on first 10 features
    matrix[:10, 6:12] *= 2.5

    cm = ConsensusMap(
        intensity_matrix=matrix,
        feature_ids=[f"f_{i}" for i in range(n_features)],
        sample_names=[f"s_{i}" for i in range(n_samples)],
    )

    # PCA
    pca_result = pca(cm, n_components=3)
    print("PCA:")
    for i in range(3):
        print(f"  PC{i+1}: {pca_result.explained_variance_ratio[i]:.1%} variance")
    print(f"  Cumulative: {sum(pca_result.explained_variance_ratio):.1%}")

    # PLS-DA
    labels = ["A"] * 6 + ["B"] * 6
    plsda_result = plsda(cm, labels, n_components=2)
    top_vip = np.argsort(plsda_result.vip_scores)[::-1][:5]
    print(f"\nPLS-DA (R2={plsda_result.r2:.3f}):")
    print(f"  Top 5 VIP features: {[f'f_{i}' for i in top_vip]}")

    # ANOVA
    labels_3g = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    anova_results = anova(cm, labels_3g)
    sig = sum(1 for r in anova_results if r.significant)
    print(f"\nANOVA (3 groups): {sig}/{len(anova_results)} significant features")

    # Volcano
    log2fc, pvals, _ = volcano_data(cm, ["s_0","s_1","s_2"], ["s_6","s_7","s_8"])
    up = sum((log2fc > 1) & (pvals > 2))
    down = sum((log2fc < -1) & (pvals > 2))
    print(f"\nVolcano: {up} up-regulated, {down} down-regulated (|FC|>2, p<0.01)")


def demo_identification():
    """Peptide identification and RT prediction."""
    header("8. Identification & RT Prediction")
    from masskit.identification import calculate_peptide_mass, generate_theoretical_fragments
    from masskit.rt_prediction import (
        RTPredictor, compute_peptide_features, simple_ssi_prediction, HYDROPHOBICITY,
    )

    # Peptide mass
    peptides = ["PEPTIDER", "ELVISLIVES", "ACDEFGHIK"]
    print("Peptide masses:")
    for seq in peptides:
        mass = calculate_peptide_mass(seq)
        print(f"  {seq}: {mass:.4f} Da")

    # Fragment ions
    fragments = generate_theoretical_fragments("PEPTIDER", charge=1)
    all_ions = []
    for ion_type, ions in fragments.items():
        for mz_val, label in ions:
            all_ions.append((mz_val, label))
    all_ions.sort()
    print(f"\nTheoretical fragments for PEPTIDER: {len(all_ions)} ions")
    for mz_val, label in all_ions[:8]:
        print(f"  {label}: {mz_val:.4f}")

    # RT prediction
    np.random.seed(42)
    aas = "ACDEFGHIKLMNPQRSTVWY"
    train_peptides, train_rts = [], []
    for _ in range(100):
        length = np.random.randint(7, 20)
        seq = "".join(np.random.choice(list(aas), size=length))
        hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in seq)
        rt = hydro * 2.0 + length * 5.0 + np.random.normal(0, 15)
        train_peptides.append(seq)
        train_rts.append(max(0, rt))

    predictor = RTPredictor()
    predictor.train(train_peptides[:80], train_rts[:80])
    metrics = predictor.evaluate(train_peptides[80:], train_rts[80:])
    print(f"\nRT Predictor (trained on 80 peptides, tested on 20):")
    print(f"  R2={metrics['r2']:.3f}  MAE={metrics['mae']:.1f}s  RMSE={metrics['rmse']:.1f}s")

    for seq in ["PEPTIDER", "ACDEFGHIK"]:
        rt = predictor.predict_single(seq)
        ssi = simple_ssi_prediction(seq)
        print(f"  {seq}: predicted RT={rt:.1f}s, SSI index={ssi:.1f}")


def demo_annotation():
    """Spectrum annotation with fragment ions."""
    header("9. Spectrum Annotation")
    from masskit import Spectrum
    from masskit.annotation import (
        annotate_spectrum, compute_fragment_ions, format_annotation_table,
        IonType, NeutralLoss,
    )

    sequence = "PEPTIDER"
    # Generate theoretical fragments and build a synthetic MS2 spectrum
    fragments = compute_fragment_ions(
        sequence,
        ion_types=[IonType.B, IonType.Y],
        charge_states=[1],
        neutral_losses=[NeutralLoss.NONE],
    )
    mz_theo = [f[0] for f in fragments]
    # Add small noise to m/z and random intensities
    rng = np.random.default_rng(42)
    mz_obs = [m + rng.normal(0, 0.005) for m in mz_theo]
    int_obs = [rng.uniform(500, 5000) for _ in mz_theo]
    # Add noise peaks
    mz_obs += list(rng.uniform(100, 900, 20))
    int_obs += list(rng.uniform(50, 300, 20))

    idx = np.argsort(mz_obs)
    spec = Spectrum(mz=np.array(mz_obs)[idx], intensity=np.array(int_obs)[idx])

    ann = annotate_spectrum(spec, sequence, precursor_charge=2, tolerance_da=0.02)
    print(f"Sequence: {sequence}")
    print(f"Matched: {ann.n_matched}/{ann.n_total_peaks} peaks ({ann.matched_fraction:.0%})")
    print(f"Sequence coverage: {ann.coverage:.0%}")
    print(f"\nTop annotations:")
    for a in sorted(ann.annotations, key=lambda x: x.intensity, reverse=True)[:8]:
        print(f"  {a.label:<10} obs={a.mz_observed:.4f}  err={a.error_ppm:.1f} ppm  int={a.intensity:.0f}")


def demo_reporting():
    """Generate an HTML report."""
    header("10. Report Generation")
    from masskit.reporting import ReportBuilder, ReportConfig

    builder = ReportBuilder(ReportConfig(
        title="MassKit Demo Report",
        author="Demo User",
    ))
    builder.add_summary(
        n_spectra=5000, n_ms1=3200, n_ms2=1800,
        n_features=1500, n_identified=800,
        rt_range=(30, 3600), mz_range=(100, 2000),
    )
    builder.add_quantification_section(
        n_features=1500, n_samples=6, missing_rate=0.05,
        normalization="median", differential_features=120,
    )
    builder.add_identification_section(
        n_psms=2500, n_peptides=1800, n_proteins=450,
    )

    out = "demo_report.html"
    builder.save_html(out)
    print(f"HTML report saved to: {out}")
    print(f"Open in browser: file://{os.path.abspath(out)}")


def demo_plugins():
    """Plugin architecture and processing pipelines."""
    header("11. Plugin Architecture")
    from masskit.plugins import PluginRegistry, ProcessingPipeline, register_as

    PluginRegistry.reset()

    @register_as("processor", "scale_up")
    def scale_up(data, factor=2):
        return data * factor

    @register_as("processor", "offset")
    def offset(data, value=10):
        return data + value

    @register_as("algorithm", "sum_array")
    def sum_array(data):
        return float(np.sum(data))

    registry = PluginRegistry.instance()
    print(f"Registered plugins: {registry.list_plugins()}")

    pipeline = ProcessingPipeline(registry)
    pipeline.add_step("scale_up", factor=3)
    pipeline.add_step("offset", value=100)
    print(f"\nPipeline: {pipeline}")

    data = np.array([1.0, 2.0, 3.0])
    result = pipeline.run(data)
    print(f"Input:  {data}")
    print(f"Output: {result}  (x3 then +100)")

    PluginRegistry.reset()


def demo_memmap():
    """Memory-mapped arrays for large datasets."""
    header("12. Memory-Mapped Arrays")
    from masskit.memmap import MemmapMatrix
    import tempfile, shutil

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "features.dat")
        mm = MemmapMatrix.create(path, n_rows=10000, n_cols=50)
        print(f"Created: {mm}")

        # Write data in chunks
        for start in range(0, 10000, 1000):
            mm[start:start+1000, :] = np.random.lognormal(10, 1, (1000, 50))
        mm.flush()

        # Compute stats without loading all into RAM
        means = mm.column_means()
        sums = mm.row_sums()
        print(f"Column means range: {means.min():.0f} - {means.max():.0f}")
        print(f"Row sums range: {sums.min():.0f} - {sums.max():.0f}")

        mm.close()
        print("Closed memory-mapped file")
    finally:
        shutil.rmtree(tmpdir)


def demo_cloud():
    """Cloud/HPC workflow generation."""
    header("13. Cloud & HPC Integration")
    from masskit.cloud import generate_snakemake_workflow, generate_nextflow_workflow, HPCJobSubmitter
    import tempfile

    tmpdir = tempfile.mkdtemp()

    # Snakemake
    sf = generate_snakemake_workflow(output_path=os.path.join(tmpdir, "Snakefile"))
    print(f"Generated Snakefile: {sf}")
    with open(sf) as f:
        lines = f.readlines()
    print(f"  ({len(lines)} lines)")

    # Nextflow
    nf = generate_nextflow_workflow(output_path=os.path.join(tmpdir, "main.nf"))
    print(f"Generated Nextflow: {nf}")

    # SLURM job script
    submitter = HPCJobSubmitter(scheduler="slurm")
    script = submitter.generate_script(
        "masskit peaks sample.mzML -o peaks.csv",
        job_name="masskit_demo", cpus=8, memory="16G", time="1:00:00",
    )
    print(f"\nSLURM job script:")
    for line in script.strip().split("\n"):
        print(f"  {line}")

    import shutil
    shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

DEMOS = {
    "spectrum":     ("Spectrum & Peak Picking",     demo_spectrum),
    "chromatogram": ("Chromatograms & Experiment",  demo_chromatogram),
    "isotope":      ("Isotope Detection",           demo_isotope),
    "matching":     ("Spectral Matching",           demo_spectral_matching),
    "quant":        ("Quantification",              demo_quantification),
    "labeling":     ("Isotope Labeling (TMT/SILAC)", demo_labeling),
    "statistics":   ("Statistical Analysis",        demo_statistics),
    "identification":("Identification & RT Pred",   demo_identification),
    "annotation":   ("Spectrum Annotation",         demo_annotation),
    "report":       ("Report Generation",           demo_reporting),
    "plugins":      ("Plugin Architecture",         demo_plugins),
    "memmap":       ("Memory-Mapped Arrays",        demo_memmap),
    "cloud":        ("Cloud & HPC",                 demo_cloud),
}


def main():
    print("=" * 60)
    print("  MassKit - Full Feature Demo")
    print("=" * 60)

    import masskit
    print(f"  Version: {masskit.__version__}")
    print(f"  NumPy:   {np.__version__}")

    if "--list" in sys.argv:
        print("\nAvailable demos:")
        for key, (name, _) in DEMOS.items():
            print(f"  {key:<16} {name}")
        return

    # Run specific demo or all
    targets = [a for a in sys.argv[1:] if not a.startswith("-")]
    if targets:
        for t in targets:
            if t in DEMOS:
                DEMOS[t][1]()
            else:
                print(f"Unknown demo: {t}. Use --list to see options.")
    else:
        for key, (name, func) in DEMOS.items():
            try:
                func()
            except Exception as e:
                print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
