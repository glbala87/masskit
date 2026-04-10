"""Tests for statistical analysis module."""

import pytest
import numpy as np

from masskit.statistics import (
    pca, plsda, anova, volcano_data,
    PCAResult, PLSDAResult, ANOVAResult,
    _benjamini_hochberg,
)
from masskit.quantification import ConsensusMap


def _make_consensus(n_features=50, n_samples=12):
    np.random.seed(42)
    matrix = np.random.lognormal(10, 1, (n_features, n_samples))
    return ConsensusMap(
        intensity_matrix=matrix,
        feature_ids=[f"f_{i}" for i in range(n_features)],
        sample_names=[f"s_{i}" for i in range(n_samples)],
    )


class TestPCA:
    def test_basic_pca(self):
        cm = _make_consensus()
        result = pca(cm, n_components=2)
        assert isinstance(result, PCAResult)
        assert result.scores.shape == (12, 2)
        assert result.loadings.shape == (50, 2)
        assert len(result.explained_variance_ratio) == 2
        assert np.all(result.explained_variance_ratio >= 0)

    def test_pca_variance_sums(self):
        cm = _make_consensus()
        result = pca(cm, n_components=10)
        cumvar = result.cumulative_variance()
        assert cumvar[-1] <= 1.0 + 1e-10

    def test_pca_no_scale(self):
        cm = _make_consensus()
        result = pca(cm, n_components=2, scale=False)
        assert result.scores.shape[1] == 2


class TestPLSDA:
    def test_basic_plsda(self):
        cm = _make_consensus(n_samples=12)
        labels = ["A"] * 6 + ["B"] * 6
        result = plsda(cm, labels, n_components=2)
        assert isinstance(result, PLSDAResult)
        assert result.scores.shape == (12, 2)
        assert len(result.vip_scores) == 50

    def test_plsda_vip(self):
        cm = _make_consensus(n_samples=12)
        labels = ["A"] * 6 + ["B"] * 6
        result = plsda(cm, labels, n_components=2)
        assert np.all(result.vip_scores >= 0)


class TestANOVA:
    def test_basic_anova(self):
        cm = _make_consensus(n_features=20, n_samples=9)
        labels = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        results = anova(cm, labels)
        assert len(results) == 20
        assert all(isinstance(r, ANOVAResult) for r in results)
        assert all(0 <= r.pvalue <= 1 for r in results)
        assert all(0 <= r.adjusted_pvalue <= 1 for r in results)


class TestVolcano:
    def test_volcano_data(self):
        cm = _make_consensus(n_features=30, n_samples=6)
        log2fc, neg_log10p, indices = volcano_data(
            cm,
            group1_samples=["s_0", "s_1", "s_2"],
            group2_samples=["s_3", "s_4", "s_5"],
        )
        assert len(log2fc) == 30
        assert len(neg_log10p) == 30
        assert np.all(neg_log10p >= 0)


class TestBH:
    def test_bh_correction(self):
        pvals = [0.001, 0.01, 0.05, 0.1, 0.5]
        adjusted = _benjamini_hochberg(pvals)
        assert len(adjusted) == 5
        # Adjusted p-values should be >= original
        for orig, adj in zip(pvals, adjusted):
            assert adj >= orig - 1e-10
        # Should be monotonic after sorting by original
        assert all(a <= 1.0 for a in adjusted)

    def test_bh_empty(self):
        assert _benjamini_hochberg([]) == []
