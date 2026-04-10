#!/usr/bin/env python
"""
Setup script for MassKit - LC-MS Data Analysis Toolkit.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description (prefer the package-local README so that
# `python -m build` works without the parent repo being available)
_here = Path(__file__).parent
for _candidate in (_here / "README.md", _here.parent / "README.md"):
    if _candidate.exists():
        long_description = _candidate.read_text(encoding="utf-8")
        break
else:
    long_description = "MassKit - LC-MS Data Analysis Toolkit"

setup(
    name="masskit",
    version="1.0.0",
    author="MassKit Contributors",
    author_email="",
    description="MassKit - LC-MS Data Analysis Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/masskit/masskit",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    keywords=[
        "mass-spectrometry",
        "proteomics",
        "metabolomics",
        "lcms",
        "mzml",
        "mzxml",
        "bioinformatics",
        "masskit",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0,<3.0.0",
        "scipy>=1.7.0,<2.0.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.4.0,<4.0.0",
        ],
        "interactive": [
            "plotly>=5.0.0,<6.0.0",
        ],
        "dashboard": [
            "plotly>=5.0.0,<6.0.0",
            "dash>=2.0.0,<3.0.0",
        ],
        "dataframe": [
            "pandas>=1.3.0,<3.0.0",
        ],
        "report": [
            "weasyprint>=52.0,<64.0",
        ],
        "cloud": [
            "dask[distributed]>=2022.1.0,<2025.0.0",
            "boto3>=1.20.0,<2.0.0",
        ],
        "full": [
            "matplotlib>=3.4.0,<4.0.0",
            "plotly>=5.0.0,<6.0.0",
            "pandas>=1.3.0,<3.0.0",
            "weasyprint>=52.0,<64.0",
        ],
        "dev": [
            "pytest>=7.0.0,<9.0.0",
            "pytest-cov>=3.0.0,<6.0.0",
            "black>=22.0.0,<25.0.0",
            "isort>=5.10.0,<6.0.0",
            "mypy>=0.950,<2.0.0",
            "flake8>=4.0.0,<8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "masskit=masskit.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/masskit/masskit/issues",
        "Source": "https://github.com/masskit/masskit",
    },
)
