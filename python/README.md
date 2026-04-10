# MassKit

> **Mass** Spectrometry Tool**kit** — Your complete LC-MS data analysis solution

[![CI](https://github.com/masskit/masskit/actions/workflows/ci.yml/badge.svg)](https://github.com/masskit/masskit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/masskit.svg)](https://pypi.org/project/masskit/)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)](https://github.com/masskit/masskit)
[![Python](https://img.shields.io/pypi/pyversions/masskit)](https://pypi.org/project/masskit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MassKit** (**Mass** Spectrometry Tool**kit**) — a production-ready Python toolkit for LC-MS (Liquid Chromatography-Mass Spectrometry) data analysis. NumPy-native data structures, mzML/mzXML I/O (including legacy 0.99.x), peak picking, feature detection, quantification, identification, statistics, and reporting.

**479 tests | 80% coverage | zero deprecation warnings | tested against real public mzML files**

## Installation

```bash
pip install masskit              # core (numpy + scipy)
pip install masskit[full]        # + matplotlib, plotly, pandas, weasyprint
pip install masskit[cloud]       # + dask, boto3
pip install masskit[dev]         # + pytest, black, mypy, flake8
```

## Quick start

```python
import masskit

# Load mzML (supports 1.1 and legacy 0.99.x, any vendor)
exp = masskit.load_mzml("sample.mzML")
print(f"{exp.spectrum_count} spectra")

# Pick peaks
peaks = masskit.pick_peaks(exp.spectrum(0), min_snr=5)

# Extract ion chromatogram
xic = exp.generate_xic(mz=500.25, tolerance=0.01)

# Quality control
qc = masskit.compute_qc_metrics(exp)
print(f"TIC CV: {qc.tic_cv:.1f}%, Status: {qc.status()}")
```

## CLI

```bash
masskit info sample.mzML                          # file summary
masskit peaks sample.mzML -o peaks.csv --snr 5    # peak picking
masskit xic sample.mzML --mz 500.25               # extract ion chromatogram
masskit convert sample.mzML -f mztab              # format conversion
masskit qc *.mzML --plot                           # QC report
masskit quantify *.mzML -o quant.tsv               # label-free quantification
masskit search query.mzML -l library.msp           # spectral library search
```

## Configuration

MassKit supports centralized configuration via JSON/YAML files or environment variables:

```bash
# Config file (auto-discovered from ./masskit.json or ~/.masskit/config.json)
masskit --config masskit.json peaks sample.mzML

# Environment variables (MASSKIT_ prefix, double underscore for nesting)
MASSKIT_PEAK_PICKING__MIN_SNR=5.0 masskit peaks sample.mzML
MASSKIT_SEARCH__FDR_THRESHOLD=0.05 masskit search query.mzML -l lib.msp
```

```python
from masskit.config import LCMSConfig

config = LCMSConfig.discover()          # auto-find config
config = LCMSConfig.from_file("masskit.json")
config = LCMSConfig.from_env()          # from MASSKIT_* env vars
config.peak_picking.min_snr = 5.0
config.validate()
```

## Core features

### Data I/O

```python
import masskit

# Load from any mzML version (1.1, 0.99.x, indexed or plain)
exp = masskit.load_mzml("sample.mzML")
exp = masskit.load_mzml("sample.mzML", ms_levels=[1], rt_range=(60, 600))

# mzXML support
exp = masskit.load_mzxml("sample.mzXML")

# MGF spectral libraries
exp = masskit.load_mgf("tandem.mgf")
masskit.save_mgf("out.mgf", exp)

# Streaming for large files (memory-bounded)
from masskit.streaming import StreamingExperiment
with StreamingExperiment("large.mzML") as exp:
    for spec in exp.filter(ms_level=1, rt_range=(60, 600)):
        process(spec)
```

### Peak picking & feature detection

```python
from masskit.workflows import PeakPickingWorkflow, FeatureDetectionWorkflow

# Single-spectrum peak picking
peaks = masskit.pick_peaks(spec, min_snr=5)

# Full workflow (smooth → baseline correct → pick → fit)
wf = PeakPickingWorkflow(min_snr=5, smooth=True, baseline_correct=True)
peaks = wf.process(spec)

# 2D feature detection across an experiment
fd = FeatureDetectionWorkflow(mz_tolerance=0.01, rt_tolerance=30.0)
features = fd.process(exp)
```

### Quantification

```python
from masskit.quantification import FeatureAlignment, ConsensusMap, DifferentialAnalysis

# Align features across runs
aligner = FeatureAlignment(mz_tolerance=0.01, rt_tolerance=30.0)
consensus = aligner.align([features_a, features_b, features_c],
                          sample_names=["A", "B", "C"])

# Normalize
consensus = consensus.normalize(method="median")

# Differential analysis
da = DifferentialAnalysis()
results = da.compare_groups(consensus,
                            group1_samples=["A"], group2_samples=["B", "C"])
```

### Isotope labeling (TMT, SILAC, iTRAQ, dimethyl)

```python
from masskit.labeling import extract_reporter_ions, LabelingStrategy

quant = extract_reporter_ions(ms2_spec, LabelingStrategy.TMT10)
print(quant.channel_intensities)

# SILAC pairs
from masskit.labeling import find_silac_pairs
pairs = find_silac_pairs(features, label_type="heavy_light")
```

### Spectral matching

```python
lib = masskit.SpectralLibrary()
lib.load_mgf("library.mgf")   # or lib.load_msp("library.msp")

matches = lib.search(query_mz, query_int,
                     query_precursor_mz=500.0,
                     method="cosine",    # or "modified_cosine", "entropy"
                     top_n=10, min_score=0.7)
```

### Identification

```python
from masskit.identification import SimpleDatabaseSearch, target_decoy_fdr

searcher = SimpleDatabaseSearch()
searcher.load_fasta("proteins.fasta", enzyme="trypsin")
psms = searcher.search(exp, precursor_tolerance_ppm=10)
filtered = target_decoy_fdr(psms, fdr_threshold=0.01)
```

### Statistics

```python
from masskit.statistics import pca, plsda, anova

result = pca(consensus, n_components=3)
print(f"PC1: {result.explained_variance_ratio[0]:.1%}")

plsda_result = plsda(consensus, labels=["A","A","B","B"], n_components=2)

anova_results = anova(consensus, labels=["A","A","B","B","C","C"])
significant = [r for r in anova_results if r.significant]
```

### Reporting

```python
from masskit.reporting import ReportBuilder, ReportConfig

builder = ReportBuilder(ReportConfig(title="My Analysis", author="Lab"))
builder.add_summary(n_spectra=5000, n_ms1=3000, n_ms2=2000)
builder.add_qc_section(qc_metrics)
builder.add_identification_section(n_psms=500, n_peptides=200, n_proteins=50)
builder.save_html("report.html")    # or .save_pdf("report.pdf")
```

### Cloud & HPC

```python
from masskit.cloud import DaskBackend, S3FileHandler, HPCJobSubmitter

# Distributed processing
backend = DaskBackend(n_workers=16)
results = backend.map(process_file, file_list)

# S3 streaming
s3 = S3FileHandler(bucket="my-lcms-data")
files = s3.list_files(prefix="raw/", suffix=".mzML")

# Generate workflow templates
from masskit.cloud import generate_snakemake_workflow
generate_snakemake_workflow(output_path="Snakefile")

# HPC job submission
sub = HPCJobSubmitter(scheduler="slurm")
script = sub.generate_script("masskit peaks sample.mzML -o peaks.csv",
                             cpus=8, memory="16G", time="1:00:00")
```

### Plugin system

```python
from masskit.plugins import PluginRegistry, register_as, ProcessingPipeline

@register_as("processor", "my_filter")
def my_filter(data, threshold=100):
    return data[data > threshold]

pipeline = ProcessingPipeline(PluginRegistry.instance())
pipeline.add_step("my_filter", threshold=50)
result = pipeline.run(data)
```

## Modules

| Module | Description |
|--------|-------------|
| `masskit.io` | mzML (1.1 + 0.99.x), mzXML, mzTab readers/writers |
| `masskit.streaming` | Indexed/streaming readers for large files |
| `masskit.algorithms` | Peak picking, smoothing, baseline correction, centroiding |
| `masskit.workflows` | High-level peak/feature detection pipelines |
| `masskit.quantification` | Feature alignment, normalization, differential analysis |
| `masskit.identification` | Peptide DB search, FDR, protein inference |
| `masskit.spectral_matching` | Library search (cosine, modified cosine, entropy) |
| `masskit.labeling` | TMT, iTRAQ, SILAC, dimethyl quantification |
| `masskit.isotope` | Isotope pattern detection, charge deconvolution |
| `masskit.annotation` | b/y/a/c/x/z fragment ion annotation |
| `masskit.statistics` | PCA, PLS-DA, ANOVA, volcano plots |
| `masskit.rt_prediction` | ML-based retention time prediction |
| `masskit.reporting` | HTML/PDF report generation |
| `masskit.cloud` | Dask, S3, Snakemake/Nextflow, SLURM/PBS |
| `masskit.parallel` | Multi-file batch processing |
| `masskit.memmap` | Memory-mapped arrays for large datasets |
| `masskit.plugins` | Extensible plugin architecture |
| `masskit.qc` | Quality control metrics |
| `masskit.config` | Centralized configuration (JSON/YAML/env) |
| `masskit.validation` | Input validation and path sanitization |
| `masskit.exceptions` | Typed exception hierarchy |

## API reference

### Data structures

| Class | Description |
|-------|-------------|
| `Spectrum` | Mass spectrum with m/z and intensity arrays |
| `Chromatogram` | Intensity vs. retention time data |
| `Peak` / `PeakList` | Detected peak (m/z, RT, area, FWHM, SNR) |
| `Feature` / `FeatureMap` | 2D feature spanning multiple scans |
| `MSExperiment` | Container for a complete LC-MS run |
| `ConsensusMap` | Aligned intensity matrix across samples |
| `SpectralLibrary` / `SpectralMatch` | Library search infrastructure |
| `PeptideSpectrumMatch` / `ProteinGroup` | Identification results |
| `QCMetrics` | Quality control metrics |
| `LCMSConfig` | Centralized configuration |

### Key functions

| Function | Description |
|----------|-------------|
| `load_mzml()` / `load_mzxml()` | Load standard MS files |
| `pick_peaks()` | Detect peaks with SNR filtering |
| `smooth_spectrum()` | Gaussian, Savitzky-Golay, moving average |
| `correct_baseline()` | SNIP, top-hat, rolling ball |
| `detect_isotope_patterns()` | Isotope envelope detection |
| `cosine_similarity()` | Spectral cosine similarity |
| `annotate_spectrum()` | Fragment ion annotation |
| `compute_qc_metrics()` | Run quality control |
| `pca()` / `plsda()` / `anova()` | Statistical analysis |
| `target_decoy_fdr()` | FDR estimation |
| `generate_analysis_report()` | One-call report generation |

## Multi-language support

MassKit also includes C++ and Java implementations:

```bash
# C++ core library
cd core && cmake -B build && cmake --build build

# Java (Maven)
cd java && mvn package

# Run Java CLI
java -jar java/target/lcms-java-1.0-SNAPSHOT-jar-with-dependencies.jar
```

## Docker

```bash
docker build -t masskit .
docker run --rm -v $(pwd)/data:/data masskit info /data/sample.mzML
docker run -it --rm masskit python -c "import masskit; print(masskit.__version__)"
```

## Requirements

- **Python** >= 3.8, NumPy >= 1.20, SciPy >= 1.7
- **C++** C++17, CMake >= 3.14
- **Java** 11+, Maven 3.6+

## Testing

```bash
# Python (479 tests, 80% coverage)
cd python && pip install -e ".[dev]"
pytest tests/ -v --cov=masskit

# Run benchmarks (100MB+ and GB-scale)
pytest tests/test_benchmark_smoke.py --runslow

# Test against real public mzML files
MASSKIT_REAL_DATA=1 pytest tests/test_real_public_data.py

# C++
cd core/build && ctest

# Java
cd java && mvn test
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{masskit,
  title = {MassKit: A Production-Ready LC-MS Data Analysis Toolkit},
  year = {2026},
  url = {https://github.com/masskit/masskit}
}
```
