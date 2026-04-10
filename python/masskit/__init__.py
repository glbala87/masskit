"""
MassKit - Python LC-MS Data Analysis Toolkit

A Python interface for LC-MS (Liquid Chromatography-Mass Spectrometry) data
analysis, providing NumPy integration and high-level workflows.

Example:
    >>> import masskit
    >>> exp = masskit.load_mzml("sample.mzML")
    >>> spec = exp.spectrum(0)
    >>> peaks = masskit.pick_peaks(spec)
"""

__version__ = "1.0.0"
__author__ = "MassKit Contributors"

from .spectrum import Spectrum, SpectrumType, Polarity
from .chromatogram import Chromatogram, ChromatogramType
from .peak import Peak, PeakList
from .feature import Feature, FeatureMap
from .experiment import MSExperiment
from .io import load_mzml, load_mzxml, save_mztab
from .algorithms import (
    pick_peaks,
    centroid_spectrum,
    smooth_spectrum,
    correct_baseline,
    estimate_noise,
)
from .isotope import (
    IsotopePattern,
    DeconvolutedMass,
    detect_isotope_patterns,
    deconvolute_spectrum,
    averagine_distribution,
    assign_charge_state,
)
from .spectral_matching import (
    SpectralLibrary,
    SpectralMatch,
    cosine_similarity,
    modified_cosine_similarity,
    spectral_entropy_similarity,
)
from .quantification import (
    ConsensusMap,
    FeatureAlignment,
    RetentionTimeAlignment,
    DifferentialAnalysis,
    DifferentialFeature,
    median_normalization,
    quantile_normalization,
    tic_normalization,
)
from .parallel import (
    BatchProcessor,
    ParallelSpectrumProcessor,
    batch_peak_picking,
    batch_feature_detection,
    batch_process,
)
from .streaming import (
    IndexedMzMLReader,
    IndexedMzXMLReader,
    StreamingExperiment,
    ChunkedProcessor,
    FileIndex,
)
from .qc import (
    QCMetrics,
    compute_qc_metrics,
    compare_qc_metrics,
    generate_qc_report,
)
from .formats import (
    load_mgf,
    save_mgf,
    save_feature_table,
    load_feature_table,
    save_peak_list,
    load_peak_list,
    load_imzml_metadata,
    save_mzidentml,
    load_identification_table,
)
from .visualization import (
    plot_spectrum,
    plot_chromatogram,
    plot_heatmap,
    plot_peaks,
)
from .statistics import (
    pca,
    plsda,
    hierarchical_clustering,
    anova,
    volcano_data,
    PCAResult,
    PLSDAResult,
    ClusterResult,
    ANOVAResult,
)
from .identification import (
    SimpleDatabaseSearch,
    PeptideSpectrumMatch,
    ProteinGroup,
    target_decoy_fdr,
    protein_inference,
    generate_theoretical_fragments,
    calculate_peptide_mass,
)
from .rt_prediction import (
    RTPredictor,
    RTPrediction,
    compute_peptide_features,
    simple_ssi_prediction,
)
from .labeling import (
    LabelingStrategy,
    ReporterIonQuantification,
    SILACPair,
    extract_reporter_ions,
    batch_extract_reporters,
    find_silac_pairs,
    normalize_reporter_intensities,
    aggregate_protein_ratios,
)
from .annotation import (
    IonType,
    NeutralLoss,
    FragmentAnnotation,
    SpectrumAnnotation,
    annotate_spectrum,
    compute_fragment_ions,
    compute_immonium_ions,
)
from .reporting import (
    ReportBuilder,
    ReportConfig,
    generate_analysis_report,
)
from .memmap import (
    MemmapMatrix,
    MemmapSpectraStore,
)
from .plugins import (
    PluginRegistry,
    PluginInfo,
    ProcessingPipeline,
    register_as,
)

__all__ = [
    # Version
    "__version__",
    # Data structures
    "Spectrum",
    "SpectrumType",
    "Polarity",
    "Chromatogram",
    "ChromatogramType",
    "Peak",
    "PeakList",
    "Feature",
    "FeatureMap",
    "MSExperiment",
    # I/O
    "load_mzml",
    "load_mzxml",
    "save_mztab",
    # Algorithms
    "pick_peaks",
    "centroid_spectrum",
    "smooth_spectrum",
    "correct_baseline",
    "estimate_noise",
    # Isotope detection
    "IsotopePattern",
    "DeconvolutedMass",
    "detect_isotope_patterns",
    "deconvolute_spectrum",
    "averagine_distribution",
    "assign_charge_state",
    # Spectral matching
    "SpectralLibrary",
    "SpectralMatch",
    "cosine_similarity",
    "modified_cosine_similarity",
    "spectral_entropy_similarity",
    # Quantification
    "ConsensusMap",
    "FeatureAlignment",
    "RetentionTimeAlignment",
    "DifferentialAnalysis",
    "DifferentialFeature",
    "median_normalization",
    "quantile_normalization",
    "tic_normalization",
    # Parallel processing
    "BatchProcessor",
    "ParallelSpectrumProcessor",
    "batch_peak_picking",
    "batch_feature_detection",
    "batch_process",
    # Streaming
    "IndexedMzMLReader",
    "IndexedMzXMLReader",
    "StreamingExperiment",
    "ChunkedProcessor",
    "FileIndex",
    # Quality Control
    "QCMetrics",
    "compute_qc_metrics",
    "compare_qc_metrics",
    "generate_qc_report",
    # Additional Formats
    "load_mgf",
    "save_mgf",
    "save_feature_table",
    "load_feature_table",
    "save_peak_list",
    "load_peak_list",
    "load_imzml_metadata",
    "save_mzidentml",
    "load_identification_table",
    # Visualization
    "plot_spectrum",
    "plot_chromatogram",
    "plot_heatmap",
    "plot_peaks",
    # Statistics
    "pca",
    "plsda",
    "hierarchical_clustering",
    "anova",
    "volcano_data",
    "PCAResult",
    "PLSDAResult",
    "ClusterResult",
    "ANOVAResult",
    # Identification
    "SimpleDatabaseSearch",
    "PeptideSpectrumMatch",
    "ProteinGroup",
    "target_decoy_fdr",
    "protein_inference",
    "generate_theoretical_fragments",
    "calculate_peptide_mass",
    # RT Prediction
    "RTPredictor",
    "RTPrediction",
    "compute_peptide_features",
    "simple_ssi_prediction",
    # Labeling
    "LabelingStrategy",
    "ReporterIonQuantification",
    "SILACPair",
    "extract_reporter_ions",
    "batch_extract_reporters",
    "find_silac_pairs",
    "normalize_reporter_intensities",
    "aggregate_protein_ratios",
    # Annotation
    "IonType",
    "NeutralLoss",
    "FragmentAnnotation",
    "SpectrumAnnotation",
    "annotate_spectrum",
    "compute_fragment_ions",
    "compute_immonium_ions",
    # Reporting
    "ReportBuilder",
    "ReportConfig",
    "generate_analysis_report",
    # Memory-mapped arrays
    "MemmapMatrix",
    "MemmapSpectraStore",
    # Plugins
    "PluginRegistry",
    "PluginInfo",
    "ProcessingPipeline",
    "register_as",
]


def version_info():
    """Return detailed version information."""
    import sys
    info = {
        "masskit_version": __version__,
        "python_version": sys.version,
    }
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        pass
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    return info
