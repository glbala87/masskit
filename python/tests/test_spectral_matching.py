"""Tests for spectral matching and similarity scoring."""

import pytest
import numpy as np

from masskit.spectral_matching import (
    cosine_similarity,
    modified_cosine_similarity,
    spectral_entropy_similarity,
    SpectralLibrary,
    SpectralMatch,
)
from masskit.spectrum import Spectrum


class TestCosineSimilarity:
    def test_identical_spectra(self):
        mz = np.array([100.0, 200.0, 300.0])
        ints = np.array([1000.0, 2000.0, 500.0])
        score, matched = cosine_similarity(mz, ints, mz, ints)
        assert abs(score - 1.0) < 0.01
        assert matched == 3

    def test_orthogonal_spectra(self):
        mz1 = np.array([100.0, 200.0])
        int1 = np.array([1000.0, 500.0])
        mz2 = np.array([300.0, 400.0])
        int2 = np.array([500.0, 1000.0])
        score, matched = cosine_similarity(mz1, int1, mz2, int2)
        assert score < 0.1

    def test_similar_spectra(self):
        mz1 = np.array([100.0, 200.0, 300.0])
        int1 = np.array([1000.0, 2000.0, 500.0])
        mz2 = np.array([100.001, 200.001, 300.001])
        int2 = np.array([900.0, 2100.0, 480.0])
        score, matched = cosine_similarity(mz1, int1, mz2, int2, tolerance=0.01)
        assert score > 0.9


class TestModifiedCosine:
    def test_with_precursor_shift(self):
        mz1 = np.array([100.0, 200.0, 300.0])
        int1 = np.array([1000.0, 2000.0, 500.0])
        mz2 = np.array([114.0, 214.0, 314.0])
        int2 = np.array([1000.0, 2000.0, 500.0])
        score, matched = modified_cosine_similarity(
            mz1, int1, 400.0,
            mz2, int2, 414.0,
            tolerance=0.02,
        )
        assert score > 0.8


class TestEntropySimilarity:
    def test_identical(self):
        mz = np.array([100.0, 200.0, 300.0])
        ints = np.array([1000.0, 2000.0, 500.0])
        score, matched = spectral_entropy_similarity(mz, ints, mz, ints)
        assert score > 0.95

    def test_different(self):
        mz1 = np.array([100.0, 200.0])
        int1 = np.array([1000.0, 500.0])
        mz2 = np.array([300.0, 400.0])
        int2 = np.array([800.0, 1200.0])
        score, matched = spectral_entropy_similarity(mz1, int1, mz2, int2)
        assert score < 0.2


class TestSpectralLibrary:
    def test_create_library(self):
        lib = SpectralLibrary()
        assert len(lib) == 0

    def test_add_and_search(self):
        lib = SpectralLibrary()
        mz = np.array([100.0, 200.0, 300.0])
        ints = np.array([1000.0, 2000.0, 500.0])
        lib.add_spectrum(
            name="test_compound",
            precursor_mz=400.0,
            mz=mz,
            intensity=ints,
        )
        assert len(lib) == 1

        query_mz = np.array([100.001, 200.001, 300.001])
        query_int = np.array([950.0, 2050.0, 520.0])
        matches = lib.search(
            query_mz, query_int,
            query_precursor_mz=400.0,
            top_n=5,
        )
        assert len(matches) >= 1
        assert matches[0].score > 0.8
