"""Tests for quantification, alignment, and normalization."""

import pytest
import numpy as np

from masskit.quantification import (
    ConsensusMap,
    median_normalization,
    quantile_normalization,
    tic_normalization,
    DifferentialAnalysis,
)


class TestConsensusMap:
    def _make_consensus(self, n_features=100, n_samples=6):
        np.random.seed(42)
        matrix = np.random.lognormal(10, 2, (n_features, n_samples))
        feature_ids = [f"feature_{i}" for i in range(n_features)]
        sample_names = [f"sample_{i}" for i in range(n_samples)]
        return ConsensusMap(
            intensity_matrix=matrix,
            feature_ids=feature_ids,
            sample_names=sample_names,
        )

    def test_create(self):
        cm = self._make_consensus()
        assert cm.n_features == 100
        assert cm.n_samples == 6

    def test_filter_by_presence(self):
        cm = self._make_consensus()
        cm.intensity_matrix[0:10, 0:3] = 0  # Add missing values
        filtered = cm.filter_by_presence(min_fraction=0.7)
        assert filtered.n_features <= cm.n_features

    def test_shape(self):
        cm = self._make_consensus()
        assert cm.shape == (100, 6)


class TestNormalization:
    def _make_matrix(self):
        np.random.seed(42)
        return np.random.lognormal(10, 2, (50, 4))

    def test_median_normalization(self):
        matrix = self._make_matrix()
        normalized = median_normalization(matrix)
        assert normalized.shape == matrix.shape
        # Column medians should be more similar after normalization
        medians = np.median(normalized, axis=0)
        assert np.std(medians) < np.std(np.median(matrix, axis=0))

    def test_quantile_normalization(self):
        matrix = self._make_matrix()
        normalized = quantile_normalization(matrix)
        assert normalized.shape == matrix.shape

    def test_tic_normalization(self):
        matrix = self._make_matrix()
        normalized = tic_normalization(matrix)
        assert normalized.shape == matrix.shape
        # Column sums should be equal after TIC normalization
        col_sums = np.sum(normalized, axis=0)
        np.testing.assert_allclose(col_sums, col_sums[0], rtol=0.01)


class TestDifferentialAnalysis:
    def test_differential(self):
        np.random.seed(42)
        n_features = 50
        matrix = np.random.lognormal(10, 1, (n_features, 6))
        # Make first 5 features truly differential
        matrix[:5, 3:6] *= 4.0

        cm = ConsensusMap(
            intensity_matrix=matrix,
            feature_ids=[f"f_{i}" for i in range(n_features)],
            sample_names=[f"s_{i}" for i in range(6)],
        )

        da = DifferentialAnalysis()
        results = da.compare_groups(
            cm,
            group1_samples=["s_0", "s_1", "s_2"],
            group2_samples=["s_3", "s_4", "s_5"],
        )
        assert len(results) == n_features
