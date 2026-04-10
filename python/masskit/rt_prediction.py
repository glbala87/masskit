"""
Retention time prediction for peptides.

Provides ML-based RT prediction using peptide sequence features
including amino acid composition, hydrophobicity, and sequence-based
descriptors.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import numpy as np


# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "D": -3.5,
    "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
}

# Amino acid single-letter codes
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class RTPrediction:
    """Retention time prediction result."""
    peptide: str = ""
    predicted_rt: float = 0.0
    confidence: float = 0.0
    features: Optional[Dict[str, float]] = None

    def __repr__(self) -> str:
        return f"RTPrediction('{self.peptide}', RT={self.predicted_rt:.2f}s)"


def compute_peptide_features(sequence: str) -> Dict[str, float]:
    """
    Compute sequence-based features for RT prediction.

    Args:
        sequence: Peptide sequence (one-letter code)

    Returns:
        Dict of feature name -> value

    Features computed:
        - Length
        - Amino acid composition (20 features)
        - Mean hydrophobicity
        - N/C-terminal hydrophobicity
        - Charge at pH 7
        - Molecular weight estimate
        - Aromaticity
        - Isoelectric point estimate
    """
    seq = sequence.upper()
    n = len(seq)

    features: Dict[str, float] = {}
    features["length"] = float(n)

    # Amino acid composition
    for aa in AMINO_ACIDS:
        count = seq.count(aa)
        features[f"aa_{aa}"] = count / n if n > 0 else 0.0

    # Hydrophobicity
    hydro_values = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq]
    features["hydro_mean"] = float(np.mean(hydro_values)) if hydro_values else 0.0
    features["hydro_sum"] = float(np.sum(hydro_values))
    features["hydro_max"] = float(np.max(hydro_values)) if hydro_values else 0.0
    features["hydro_min"] = float(np.min(hydro_values)) if hydro_values else 0.0

    # N/C terminal hydrophobicity (first/last 3 residues)
    if n >= 3:
        features["hydro_nterm"] = float(np.mean(
            [HYDROPHOBICITY.get(aa, 0) for aa in seq[:3]]))
        features["hydro_cterm"] = float(np.mean(
            [HYDROPHOBICITY.get(aa, 0) for aa in seq[-3:]]))
    else:
        features["hydro_nterm"] = features["hydro_mean"]
        features["hydro_cterm"] = features["hydro_mean"]

    # Charge at pH 7
    pos_charge = seq.count("K") + seq.count("R") + seq.count("H") * 0.1
    neg_charge = seq.count("D") + seq.count("E")
    features["charge_ph7"] = float(pos_charge - neg_charge + 1)  # +1 for N-term

    # Aromaticity
    aromatic = seq.count("F") + seq.count("W") + seq.count("Y")
    features["aromaticity"] = aromatic / n if n > 0 else 0.0

    # Bulkiness (fraction of large residues)
    bulky = sum(1 for aa in seq if aa in "FWYLIMR")
    features["bulkiness"] = bulky / n if n > 0 else 0.0

    # Proline content (affects secondary structure)
    features["proline_fraction"] = seq.count("P") / n if n > 0 else 0.0

    return features


class RTPredictor:
    """
    Retention time predictor using linear regression on peptide features.

    Example:
        >>> predictor = RTPredictor()
        >>> predictor.train(peptides, observed_rts)
        >>> predictions = predictor.predict(["PEPTIDER", "ANOTHERK"])
    """

    def __init__(self):
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._feature_names: List[str] = []
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._is_trained: bool = False
        self._r2: float = 0.0
        self._mae: float = 0.0

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def r2_score(self) -> float:
        return self._r2

    @property
    def mean_absolute_error(self) -> float:
        return self._mae

    def train(
        self,
        peptides: List[str],
        observed_rts: List[float],
        regularization: float = 1.0,
    ) -> None:
        """
        Train the RT prediction model.

        Args:
            peptides: List of peptide sequences
            observed_rts: Corresponding observed retention times
            regularization: Ridge regression regularization strength
        """
        if len(peptides) != len(observed_rts):
            raise ValueError("peptides and observed_rts must have same length")

        if len(peptides) < 5:
            raise ValueError("Need at least 5 training peptides")

        # Compute features
        all_features = [compute_peptide_features(p) for p in peptides]
        self._feature_names = sorted(all_features[0].keys())

        X = np.array([[f[name] for name in self._feature_names]
                       for f in all_features])
        y = np.array(observed_rts)

        # Standardize
        self._feature_mean = np.mean(X, axis=0)
        self._feature_std = np.std(X, axis=0)
        self._feature_std[self._feature_std == 0] = 1.0
        X_scaled = (X - self._feature_mean) / self._feature_std

        # Ridge regression: w = (X^T X + lambda I)^-1 X^T y
        n_features = X_scaled.shape[1]
        XtX = X_scaled.T @ X_scaled + regularization * np.eye(n_features)
        Xty = X_scaled.T @ y

        try:
            self._weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self._weights = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        self._bias = np.mean(y) - X_scaled.mean(axis=0) @ self._weights

        # Compute training metrics
        y_pred = X_scaled @ self._weights + self._bias
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self._r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        self._mae = float(np.mean(np.abs(y - y_pred)))

        self._is_trained = True

    def predict(self, peptides: List[str]) -> List[RTPrediction]:
        """
        Predict retention times for peptides.

        Args:
            peptides: List of peptide sequences

        Returns:
            List of RTPrediction objects
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        results = []
        for peptide in peptides:
            features = compute_peptide_features(peptide)
            x = np.array([features[name] for name in self._feature_names])
            x_scaled = (x - self._feature_mean) / self._feature_std

            predicted_rt = float(x_scaled @ self._weights + self._bias)

            results.append(RTPrediction(
                peptide=peptide,
                predicted_rt=predicted_rt,
                confidence=self._r2,
                features=features,
            ))

        return results

    def predict_single(self, peptide: str) -> float:
        """Predict RT for a single peptide. Returns RT value."""
        results = self.predict([peptide])
        return results[0].predicted_rt

    def evaluate(
        self,
        peptides: List[str],
        observed_rts: List[float],
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Returns:
            Dict with 'r2', 'mae', 'rmse', 'median_ae'
        """
        predictions = self.predict(peptides)
        pred_rts = np.array([p.predicted_rt for p in predictions])
        obs_rts = np.array(observed_rts)

        errors = pred_rts - obs_rts
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((obs_rts - np.mean(obs_rts)) ** 2)

        return {
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "median_ae": float(np.median(np.abs(errors))),
            "n_samples": len(peptides),
        }


def simple_ssi_prediction(sequence: str) -> float:
    """
    Simple Sequence-Specific Index RT prediction.

    Uses a hydrophobicity-based index for quick RT estimation.
    Good for approximate ordering, not absolute RT values.

    Args:
        sequence: Peptide sequence

    Returns:
        Predicted RT index (arbitrary units, higher = later elution)
    """
    # Sequence-Specific Retention Calculator (SSRCalc-like)
    # Simplified version using Kyte-Doolittle + length correction
    seq = sequence.upper()
    n = len(seq)

    hydro_sum = sum(HYDROPHOBICITY.get(aa, 0) for aa in seq)

    # Length correction (longer peptides elute later)
    length_factor = np.log(n) * 2.0 if n > 0 else 0.0

    # N-terminal effect
    nterm_bonus = HYDROPHOBICITY.get(seq[0], 0) * 0.5 if seq else 0.0

    return hydro_sum + length_factor + nterm_bonus
