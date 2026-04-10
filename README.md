# MassKit

> **Mass** Spectrometry Tool**kit** — Your complete LC-MS data analysis solution

[![CI](https://github.com/masskit/masskit/actions/workflows/ci.yml/badge.svg)](https://github.com/masskit/masskit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/masskit.svg)](https://pypi.org/project/masskit/)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)](https://github.com/masskit/masskit)
[![Python](https://img.shields.io/pypi/pyversions/masskit)](https://pypi.org/project/masskit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MassKit** (**Mass** Spectrometry Tool**kit**) is a production-ready, cross-platform toolkit for **LC-MS (Liquid Chromatography-Mass Spectrometry)** data analysis. Implementations in **Python** (primary), **C++** (core), and **Java** (CLI + GUI), with a unified API for mass spectrometry data processing, quantification, identification, and statistical analysis.

**479 tests | 80% coverage | zero deprecation warnings | tested against real public mzML files from HUPO-PSI**

---

## Quick start

```bash
pip install masskit
```

```python
import masskit

exp = masskit.load_mzml("sample.mzML")
peaks = masskit.pick_peaks(exp.spectrum(0), min_snr=5)
qc = masskit.compute_qc_metrics(exp)
print(f"{exp.spectrum_count} spectra, TIC CV: {qc.tic_cv:.1f}%, Status: {qc.status()}")
```

```bash
masskit info sample.mzML
masskit peaks sample.mzML -o peaks.csv --snr 5
masskit quantify *.mzML -o quant.tsv
masskit search query.mzML -l library.msp
masskit qc *.mzML --plot
```

---

## Features

### Core capabilities
- **File formats**: mzML (1.1 + legacy 0.99.x with auto-detection of precision/endianness), mzXML, MGF, MSP, mzTab, CSV/TSV
- **Data structures**: Spectrum, Chromatogram, Peak, Feature, MSExperiment — all NumPy-native
- **Signal processing**: Peak picking, Gaussian/Savitzky-Golay smoothing, SNIP baseline correction, centroiding
- **Streaming**: Indexed random-access readers for GB-scale files with bounded memory

### Analysis
- **Isotope detection**: Averagine model, charge state deconvolution
- **Spectral matching**: Cosine, modified cosine, spectral entropy similarity with MGF/MSP library search
- **Quantification**: RT alignment, feature linking, consensus maps, median/quantile/TIC normalization, differential analysis
- **Labeling**: TMT (6/10/11/16/18-plex), iTRAQ (4/8), SILAC (2/3-plex), dimethyl
- **Identification**: In-memory database search, fragment matching, target-decoy FDR, protein inference
- **Annotation**: b/y/a/c/x/z ion series, neutral losses, immonium ions
- **RT prediction**: Ridge regression on sequence features

### Statistics & reporting
- **Multivariate**: PCA, PLS-DA, hierarchical clustering, ANOVA with BH-FDR
- **Reports**: Styled HTML/PDF with embedded figures, tables, and QC status
- **QC metrics**: TIC stability, peak capacity, mass accuracy, scan rate

### Infrastructure
- **Parallel processing**: Batch file processing, spectrum-level parallelism
- **Cloud/HPC**: Dask distributed, S3 streaming, Snakemake/Nextflow templates, SLURM/PBS job submission
- **Configuration**: Centralized config via JSON/YAML files, `MASSKIT_*` env vars, auto-discovery
- **Validation**: Path traversal detection, array validation, typed exception hierarchy
- **Plugins**: Register custom algorithms, readers, writers, and pipelines
- **Memory-mapped**: numpy memmap for large feature matrices
- **CLI**: `masskit` command with subcommands for all major operations
- **Docker**: Multi-stage Dockerfile, published to GHCR
- **CI/CD**: Cross-platform (Ubuntu/macOS/Windows), Python 3.9-3.12 matrix, pip-audit, Dependabot, PyPI publish

---

## Installation

```bash
# Core
pip install masskit

# Extras
pip install masskit[full]        # matplotlib, plotly, pandas, weasyprint
pip install masskit[cloud]       # dask, boto3
pip install masskit[dev]         # pytest, black, mypy, flake8

# C++
cd core && cmake -B build && cmake --build build

# Java
cd java && mvn package
```

---

## Python API

```python
import masskit

# Load any mzML version (1.1, 0.99.x, indexed, any vendor)
exp = masskit.load_mzml("sample.mzML")

# Peak picking
peaks = masskit.pick_peaks(exp.spectrum(0), min_snr=5)

# Feature detection
from masskit.workflows import FeatureDetectionWorkflow
features = FeatureDetectionWorkflow(mz_tolerance=0.01).process(exp)

# Quantification across runs
from masskit.quantification import FeatureAlignment
consensus = FeatureAlignment().align([feat_a, feat_b], ["run_A", "run_B"])

# TMT reporter ions
from masskit.labeling import extract_reporter_ions, LabelingStrategy
quant = extract_reporter_ions(ms2_spec, LabelingStrategy.TMT10)

# Spectral library search
lib = masskit.SpectralLibrary()
lib.load_mgf("library.mgf")
matches = lib.search(query_mz, query_int, query_precursor_mz=500.0, top_n=10)

# Identification
from masskit.identification import SimpleDatabaseSearch, target_decoy_fdr
searcher = SimpleDatabaseSearch()
searcher.load_fasta("proteins.fasta")
psms = searcher.search(exp, precursor_tolerance_ppm=10)
filtered = target_decoy_fdr(psms, fdr_threshold=0.01)

# Statistics
from masskit.statistics import pca, anova
pca_result = pca(consensus, n_components=3)
anova_results = anova(consensus, labels=group_labels)

# Streaming for large files
from masskit.streaming import StreamingExperiment
with StreamingExperiment("large.mzML") as exp:
    for spec in exp.filter(ms_level=1):
        process(spec)

# HTML report
from masskit.reporting import generate_analysis_report
generate_analysis_report(
    experiment_summary={"n_spectra": 5000, "n_ms1": 3000},
    qc_metrics=qc,
    output_path="report.html",
)
```

## CLI tool

```bash
masskit info sample.mzML                           # file summary
masskit info sample.mzML -v                        # verbose (first 10 spectra)
masskit peaks sample.mzML -o peaks.csv --snr 5     # peak picking
masskit peaks sample.mzML --tsv --ms-level 2       # MS2 peaks as TSV
masskit xic sample.mzML --mz 500.25 --plot         # XIC with plot
masskit convert sample.mzML -f mztab               # convert to mzTab
masskit quantify *.mzML -o quant.tsv --normalize median
masskit search query.mzML -l lib.msp --method entropy --min-score 0.7
masskit qc run1.mzML run2.mzML run3.mzML --plot -o qc.png
masskit --config masskit.json peaks sample.mzML     # use config file
masskit --log-level DEBUG info sample.mzML          # verbose logging
```

## Configuration

```bash
# Environment variables
MASSKIT_PEAK_PICKING__MIN_SNR=5.0 masskit peaks sample.mzML
MASSKIT_SEARCH__FDR_THRESHOLD=0.05 masskit search query.mzML -l lib.msp

# Config file (JSON or YAML)
masskit --config masskit.json peaks sample.mzML
```

Auto-discovery checks: `./masskit.json` → `~/.masskit/config.json` → `MASSKIT_*` env vars → defaults.

## Cloud & HPC

```python
from masskit.cloud import DaskBackend, S3FileHandler, HPCJobSubmitter

backend = DaskBackend(n_workers=16)
results = backend.map(process_file, file_list)

s3 = S3FileHandler(bucket="lcms-data")
files = s3.list_files(prefix="raw/", suffix=".mzML")

sub = HPCJobSubmitter(scheduler="slurm")
script = sub.generate_script("masskit peaks s.mzML", cpus=8, memory="16G")
```

## Docker

```bash
docker build -t masskit .
docker run --rm -v $(pwd)/data:/data masskit info /data/sample.mzML
docker run -it --rm masskit python -c "import masskit; print(masskit.__version__)"
```

---

## Project structure

```
masskit/
├── python/
│   ├── masskit/               # Python package (28 modules)
│   │   ├── io.py              # mzML/mzXML I/O (1.1 + 0.99.x)
│   │   ├── streaming.py       # Indexed/streaming readers
│   │   ├── algorithms.py      # Signal processing
│   │   ├── workflows.py       # Peak/feature pipelines
│   │   ├── quantification.py  # LFQ, alignment, normalization
│   │   ├── identification.py  # DB search, FDR, protein inference
│   │   ├── spectral_matching.py
│   │   ├── labeling.py        # TMT, SILAC, iTRAQ, dimethyl
│   │   ├── statistics.py      # PCA, PLS-DA, ANOVA
│   │   ├── annotation.py      # Fragment ion annotation
│   │   ├── reporting.py       # HTML/PDF reports
│   │   ├── cloud.py           # Dask, S3, Snakemake, HPC
│   │   ├── config.py          # Centralized configuration
│   │   ├── validation.py      # Input validation
│   │   ├── exceptions.py      # Typed exceptions
│   │   ├── cli.py             # CLI entry point
│   │   └── ...
│   ├── tests/                 # 479 pytest tests (80% coverage)
│   ├── benchmarks/            # 100MB + GB-scale benchmarks
│   ├── setup.py
│   └── pyproject.toml
├── core/                      # C++ core library (C++17, CMake)
├── java/                      # Java implementation (Maven, JUnit 5)
├── Dockerfile
├── .github/
│   ├── workflows/ci.yml       # Cross-platform CI matrix
│   ├── workflows/publish.yml  # PyPI + Docker publish
│   └── dependabot.yml         # Automated dependency updates
└── README.md
```

---

## Testing

```bash
# Python (479 tests, 80% coverage)
cd python && pip install -e ".[dev]"
pytest tests/ -v --cov=masskit

# Benchmarks (100MB in-memory + GB-scale streaming)
pytest tests/test_benchmark_smoke.py --runslow

# Real public mzML files (HUPO-PSI reference data)
MASSKIT_REAL_DATA=1 pytest tests/test_real_public_data.py

# C++
cd core/build && ctest

# Java
cd java && mvn test
```

---

## Requirements

| Language | Minimum | Dependencies |
|----------|---------|-------------|
| Python | >= 3.8 | numpy >= 1.20, scipy >= 1.7 |
| C++ | C++17 | CMake >= 3.14 |
| Java | 11+ | Maven 3.6+ |

Optional Python extras: matplotlib, plotly, dash, pandas, weasyprint, dask, boto3.

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
