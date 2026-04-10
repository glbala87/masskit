"""Tests for RT prediction module."""

import pytest
import numpy as np

from masskit.rt_prediction import (
    compute_peptide_features,
    RTPredictor,
    RTPrediction,
    simple_ssi_prediction,
    HYDROPHOBICITY,
)


class TestFeatures:
    def test_basic_features(self):
        features = compute_peptide_features("ACDEFGHIK")
        assert "length" in features
        assert features["length"] == 9
        assert "hydro_mean" in features
        assert "charge_ph7" in features
        assert "aromaticity" in features

    def test_aa_composition(self):
        features = compute_peptide_features("AAAA")
        assert features["aa_A"] == 1.0
        assert features["aa_K"] == 0.0

    def test_hydrophobicity_range(self):
        features = compute_peptide_features("PEPTIDER")
        assert -5 < features["hydro_mean"] < 5


class TestRTPredictor:
    def _make_training_data(self, n=50):
        np.random.seed(42)
        peptides = []
        rts = []
        aas = "ACDEFGHIKLMNPQRSTVWY"
        for i in range(n):
            length = np.random.randint(6, 20)
            seq = "".join(np.random.choice(list(aas), size=length))
            peptides.append(seq)
            # RT correlates with hydrophobicity
            hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in seq)
            rt = hydro * 2.0 + length * 5.0 + np.random.normal(0, 20)
            rts.append(max(0, rt))
        return peptides, rts

    def test_train(self):
        peptides, rts = self._make_training_data()
        predictor = RTPredictor()
        predictor.train(peptides, rts)
        assert predictor.is_trained

    def test_predict(self):
        peptides, rts = self._make_training_data()
        predictor = RTPredictor()
        predictor.train(peptides, rts)
        results = predictor.predict(["PEPTIDER", "ACDEFGHIK"])
        assert len(results) == 2
        assert all(isinstance(r, RTPrediction) for r in results)

    def test_predict_single(self):
        peptides, rts = self._make_training_data()
        predictor = RTPredictor()
        predictor.train(peptides, rts)
        rt = predictor.predict_single("PEPTIDER")
        assert isinstance(rt, float)

    def test_evaluate(self):
        peptides, rts = self._make_training_data(100)
        train_p, test_p = peptides[:80], peptides[80:]
        train_r, test_r = rts[:80], rts[80:]

        predictor = RTPredictor()
        predictor.train(train_p, train_r)
        metrics = predictor.evaluate(test_p, test_r)
        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics

    def test_insufficient_data(self):
        predictor = RTPredictor()
        with pytest.raises(ValueError):
            predictor.train(["AA", "BB"], [1.0, 2.0])

    def test_not_trained(self):
        predictor = RTPredictor()
        with pytest.raises(RuntimeError):
            predictor.predict(["PEPTIDE"])


class TestSSI:
    def test_ssi_prediction(self):
        rt1 = simple_ssi_prediction("AAAA")  # Hydrophobic
        rt2 = simple_ssi_prediction("KKKK")  # Hydrophilic
        # Hydrophobic peptides elute later
        assert rt1 > rt2

    def test_ssi_length_effect(self):
        rt_short = simple_ssi_prediction("AA")
        rt_long = simple_ssi_prediction("AAAAAAAAAA")
        assert rt_long > rt_short
