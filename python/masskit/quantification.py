"""
Label-free quantification (LFQ) for LC-MS data.

Provides retention time alignment, feature alignment across runs,
consensus map generation, intensity normalization, and differential analysis.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import numpy as np

from .feature import Feature, FeatureMap


@dataclass
class DifferentialFeature:
    """Result of differential analysis for a single feature."""
    feature_index: int = 0
    log2_fc: float = 0.0
    pvalue: float = 1.0
    adjusted_pvalue: float = 1.0
    mean_group1: float = 0.0
    mean_group2: float = 0.0
    significant: bool = False

    def __repr__(self) -> str:
        sig = "*" if self.significant else ""
        return (
            f"DifferentialFeature(idx={self.feature_index}, "
            f"log2FC={self.log2_fc:.3f}, p={self.adjusted_pvalue:.4f}{sig})"
        )


class ConsensusMap:
    """
    Matrix of features (rows) x samples (columns) with intensities.

    Represents aligned features across multiple LC-MS runs.

    Example:
        >>> cmap = ConsensusMap(feature_ids, sample_names, intensity_matrix)
        >>> cmap.normalize(method='median')
        >>> df = cmap.to_dataframe()
    """

    def __init__(
        self,
        feature_ids: List[str],
        sample_names: List[str],
        intensity_matrix: np.ndarray,
        mz_values: Optional[np.ndarray] = None,
        rt_values: Optional[np.ndarray] = None,
    ):
        self.feature_ids = feature_ids
        self.sample_names = sample_names
        self.intensity_matrix = np.asarray(intensity_matrix, dtype=np.float64)
        self.mz_values = mz_values
        self.rt_values = rt_values

    @property
    def n_features(self) -> int:
        return self.intensity_matrix.shape[0]

    @property
    def n_samples(self) -> int:
        return self.intensity_matrix.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.intensity_matrix.shape

    def filter_by_presence(self, min_fraction: float = 0.7) -> "ConsensusMap":
        """
        Filter features by minimum presence across samples.

        Args:
            min_fraction: Minimum fraction of samples with non-zero intensity

        Returns:
            New ConsensusMap with filtered features
        """
        presence = np.sum(self.intensity_matrix > 0, axis=1) / self.n_samples
        mask = presence >= min_fraction

        return ConsensusMap(
            feature_ids=[fid for fid, m in zip(self.feature_ids, mask) if m],
            sample_names=self.sample_names,
            intensity_matrix=self.intensity_matrix[mask],
            mz_values=self.mz_values[mask] if self.mz_values is not None else None,
            rt_values=self.rt_values[mask] if self.rt_values is not None else None,
        )

    def fill_missing(self, method: str = "min") -> "ConsensusMap":
        """
        Fill missing (zero) values.

        Args:
            method: Fill method:
                - 'min': Fill with column minimum
                - 'half_min': Fill with half of column minimum
                - 'mean': Fill with column mean
                - 'median': Fill with column median

        Returns:
            New ConsensusMap with filled values
        """
        matrix = self.intensity_matrix.copy()

        for j in range(self.n_samples):
            col = matrix[:, j]
            nonzero = col[col > 0]
            if len(nonzero) == 0:
                continue

            if method == "min":
                fill_val = np.min(nonzero)
            elif method == "half_min":
                fill_val = np.min(nonzero) / 2
            elif method == "mean":
                fill_val = np.mean(nonzero)
            elif method == "median":
                fill_val = np.median(nonzero)
            else:
                raise ValueError(f"Unknown fill method: {method}")

            col[col == 0] = fill_val

        return ConsensusMap(
            feature_ids=self.feature_ids,
            sample_names=self.sample_names,
            intensity_matrix=matrix,
            mz_values=self.mz_values,
            rt_values=self.rt_values,
        )

    def normalize(self, method: str = "median") -> "ConsensusMap":
        """
        Normalize intensities.

        Args:
            method: Normalization method ('median', 'quantile', 'tic')

        Returns:
            New ConsensusMap with normalized values
        """
        matrix = self.intensity_matrix.copy()

        if method == "median":
            matrix = median_normalization(matrix)
        elif method == "quantile":
            matrix = quantile_normalization(matrix)
        elif method == "tic":
            matrix = tic_normalization(matrix)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return ConsensusMap(
            feature_ids=self.feature_ids,
            sample_names=self.sample_names,
            intensity_matrix=matrix,
            mz_values=self.mz_values,
            rt_values=self.rt_values,
        )

    def log_transform(self, base: float = 2.0) -> "ConsensusMap":
        """Log-transform intensities (adds 1 to avoid log(0))."""
        matrix = np.log(self.intensity_matrix + 1) / np.log(base)
        return ConsensusMap(
            feature_ids=self.feature_ids,
            sample_names=self.sample_names,
            intensity_matrix=matrix,
            mz_values=self.mz_values,
            rt_values=self.rt_values,
        )

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        df = pd.DataFrame(
            self.intensity_matrix,
            index=self.feature_ids,
            columns=self.sample_names,
        )
        if self.mz_values is not None:
            df.insert(0, "mz", self.mz_values)
        if self.rt_values is not None:
            df.insert(1 if self.mz_values is not None else 0, "rt", self.rt_values)
        return df

    def __repr__(self) -> str:
        return f"ConsensusMap({self.n_features} features x {self.n_samples} samples)"


class RetentionTimeAlignment:
    """
    Retention time alignment between LC-MS runs.

    Example:
        >>> aligner = RetentionTimeAlignment()
        >>> aligner.fit(reference_features, target_features)
        >>> corrected_rt = aligner.transform(rt_values)
    """

    def __init__(self):
        self._coefficients = None
        self._method = None
        self._anchor_pairs = None

    def align_linear(
        self,
        reference_features: FeatureMap,
        target_features: FeatureMap,
        mz_tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        Linear RT alignment.

        Args:
            reference_features: Reference run features
            target_features: Target run features to align
            mz_tolerance: m/z tolerance for matching features

        Returns:
            Array of corrected RT values for target features
        """
        self._method = "linear"
        pairs = self._find_anchor_pairs(
            reference_features, target_features, mz_tolerance
        )
        self._anchor_pairs = pairs

        if len(pairs) < 2:
            self._coefficients = np.array([0.0, 1.0])
            return np.array([f.rt for f in target_features])

        ref_rts = np.array([p[0] for p in pairs])
        tgt_rts = np.array([p[1] for p in pairs])

        # Linear fit: ref_rt = a + b * tgt_rt
        self._coefficients = np.polyfit(tgt_rts, ref_rts, 1)

        corrected = np.polyval(self._coefficients,
                               np.array([f.rt for f in target_features]))
        return corrected

    def align_loess(
        self,
        reference_features: FeatureMap,
        target_features: FeatureMap,
        mz_tolerance: float = 0.01,
        span: float = 0.5,
    ) -> np.ndarray:
        """
        LOESS-based RT alignment.

        Args:
            reference_features: Reference run features
            target_features: Target run features to align
            mz_tolerance: m/z tolerance for matching features
            span: LOESS smoothing span (0-1)

        Returns:
            Array of corrected RT values for target features
        """
        self._method = "loess"
        pairs = self._find_anchor_pairs(
            reference_features, target_features, mz_tolerance
        )
        self._anchor_pairs = pairs

        if len(pairs) < 4:
            return self.align_linear(
                reference_features, target_features, mz_tolerance
            )

        ref_rts = np.array([p[0] for p in pairs])
        tgt_rts = np.array([p[1] for p in pairs])
        deviations = ref_rts - tgt_rts

        target_rt_values = np.array([f.rt for f in target_features])
        corrections = self._loess_predict(tgt_rts, deviations, target_rt_values, span)

        return target_rt_values + corrections

    def transform(self, rt_values: np.ndarray) -> np.ndarray:
        """Apply learned alignment to new RT values."""
        if self._method == "linear" and self._coefficients is not None:
            return np.polyval(self._coefficients, rt_values)
        elif self._method == "loess" and self._anchor_pairs is not None:
            ref_rts = np.array([p[0] for p in self._anchor_pairs])
            tgt_rts = np.array([p[1] for p in self._anchor_pairs])
            deviations = ref_rts - tgt_rts
            corrections = self._loess_predict(tgt_rts, deviations, rt_values, 0.5)
            return rt_values + corrections
        return rt_values.copy()

    def compute_rt_deviation(self) -> float:
        """Compute median RT deviation after alignment."""
        if self._anchor_pairs is None or len(self._anchor_pairs) == 0:
            return 0.0
        deviations = [abs(p[0] - p[1]) for p in self._anchor_pairs]
        return float(np.median(deviations))

    def _find_anchor_pairs(
        self,
        ref_features: FeatureMap,
        tgt_features: FeatureMap,
        mz_tolerance: float,
    ) -> List[Tuple[float, float]]:
        """Find matching feature pairs between runs."""
        pairs = []
        used = set()

        for ref_feat in ref_features:
            best_idx = -1
            best_diff = float("inf")

            for j, tgt_feat in enumerate(tgt_features):
                if j in used:
                    continue
                mz_diff = abs(ref_feat.mz - tgt_feat.mz)
                if mz_diff <= mz_tolerance and mz_diff < best_diff:
                    best_diff = mz_diff
                    best_idx = j

            if best_idx >= 0:
                pairs.append((ref_feat.rt, tgt_features[best_idx].rt))
                used.add(best_idx)

        return pairs

    @staticmethod
    def _loess_predict(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_pred: np.ndarray,
        span: float,
    ) -> np.ndarray:
        """Simple LOESS (locally weighted regression) prediction."""
        n = len(x_train)
        k = max(3, int(np.ceil(span * n)))
        predictions = np.zeros(len(x_pred))

        for i, xp in enumerate(x_pred):
            distances = np.abs(x_train - xp)
            nearest_idx = np.argsort(distances)[:k]
            nearest_x = x_train[nearest_idx]
            nearest_y = y_train[nearest_idx]
            nearest_d = distances[nearest_idx]

            max_d = nearest_d[-1]
            if max_d == 0:
                max_d = 1.0

            # Tricube weights
            u = nearest_d / max_d
            weights = (1 - u**3)**3
            weights = np.maximum(weights, 0)

            # Weighted linear regression
            w_sum = np.sum(weights)
            if w_sum == 0:
                predictions[i] = np.mean(nearest_y)
                continue

            w_x = np.sum(weights * nearest_x) / w_sum
            w_y = np.sum(weights * nearest_y) / w_sum
            w_xx = np.sum(weights * (nearest_x - w_x)**2)

            if w_xx == 0:
                predictions[i] = w_y
            else:
                slope = np.sum(weights * (nearest_x - w_x) * (nearest_y - w_y)) / w_xx
                predictions[i] = w_y + slope * (xp - w_x)

        return predictions


class FeatureAlignment:
    """
    Align features across multiple LC-MS runs.

    Example:
        >>> aligner = FeatureAlignment(mz_tolerance=0.01, rt_tolerance=30.0)
        >>> consensus = aligner.align(feature_maps, sample_names)
    """

    def __init__(
        self,
        mz_tolerance: float = 0.01,
        rt_tolerance: float = 30.0,
        use_rt_alignment: bool = True,
    ):
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.use_rt_alignment = use_rt_alignment

    def align(
        self,
        feature_maps: List[FeatureMap],
        sample_names: Optional[List[str]] = None,
    ) -> ConsensusMap:
        """
        Align features across multiple runs.

        Args:
            feature_maps: List of FeatureMap objects (one per run)
            sample_names: Optional sample names

        Returns:
            ConsensusMap with aligned features
        """
        n_samples = len(feature_maps)
        if sample_names is None:
            sample_names = [f"sample_{i}" for i in range(n_samples)]

        if n_samples == 0:
            return ConsensusMap([], sample_names, np.empty((0, 0)))

        # Use first run as reference for RT alignment
        if self.use_rt_alignment and n_samples > 1:
            rt_aligner = RetentionTimeAlignment()
            for i in range(1, n_samples):
                corrected_rts = rt_aligner.align_linear(
                    feature_maps[0], feature_maps[i], self.mz_tolerance
                )
                for j, feat in enumerate(feature_maps[i]):
                    if j < len(corrected_rts):
                        feat.rt = corrected_rts[j]

        # Build consensus features by greedy matching
        # Start with features from the first sample
        consensus_features: List[Dict] = []
        for feat in feature_maps[0]:
            consensus_features.append({
                "mz": feat.mz,
                "rt": feat.rt,
                "intensities": {0: feat.intensity},
            })

        # Match features from remaining samples
        for sample_idx in range(1, n_samples):
            used_consensus = set()
            for feat in feature_maps[sample_idx]:
                best_match = -1
                best_dist = float("inf")

                for ci, cf in enumerate(consensus_features):
                    if ci in used_consensus:
                        continue
                    mz_diff = abs(feat.mz - cf["mz"])
                    rt_diff = abs(feat.rt - cf["rt"])

                    if mz_diff <= self.mz_tolerance and rt_diff <= self.rt_tolerance:
                        dist = mz_diff / self.mz_tolerance + rt_diff / self.rt_tolerance
                        if dist < best_dist:
                            best_dist = dist
                            best_match = ci

                if best_match >= 0:
                    cf = consensus_features[best_match]
                    cf["intensities"][sample_idx] = feat.intensity
                    # Update consensus m/z and rt as weighted averages
                    n = len(cf["intensities"])
                    cf["mz"] = (cf["mz"] * (n - 1) + feat.mz) / n
                    cf["rt"] = (cf["rt"] * (n - 1) + feat.rt) / n
                    used_consensus.add(best_match)
                else:
                    consensus_features.append({
                        "mz": feat.mz,
                        "rt": feat.rt,
                        "intensities": {sample_idx: feat.intensity},
                    })

        # Build intensity matrix
        n_features = len(consensus_features)
        matrix = np.zeros((n_features, n_samples))
        feature_ids = []
        mz_values = np.zeros(n_features)
        rt_values = np.zeros(n_features)

        for i, cf in enumerate(consensus_features):
            feature_ids.append(f"F{i:06d}")
            mz_values[i] = cf["mz"]
            rt_values[i] = cf["rt"]
            for sample_idx, intensity in cf["intensities"].items():
                matrix[i, sample_idx] = intensity

        return ConsensusMap(
            feature_ids=feature_ids,
            sample_names=sample_names,
            intensity_matrix=matrix,
            mz_values=mz_values,
            rt_values=rt_values,
        )


class DifferentialAnalysis:
    """
    Differential expression analysis between sample groups.

    Example:
        >>> da = DifferentialAnalysis(alpha=0.05)
        >>> results = da.compare_groups(
        ...     consensus_map, group1=["s1","s2"], group2=["s3","s4"]
        ... )
        >>> significant = [r for r in results if r.significant]
    """

    def __init__(self, alpha: float = 0.05, min_fold_change: float = 1.0):
        self.alpha = alpha
        self.min_fold_change = min_fold_change

    def compare_groups(
        self,
        consensus_map: ConsensusMap,
        group1_samples: List[str],
        group2_samples: List[str],
    ) -> List[DifferentialFeature]:
        """
        Compare two groups of samples.

        Args:
            consensus_map: Aligned feature matrix
            group1_samples: Sample names in group 1
            group2_samples: Sample names in group 2

        Returns:
            List of DifferentialFeature results
        """
        from scipy import stats

        idx1 = [consensus_map.sample_names.index(s) for s in group1_samples
                if s in consensus_map.sample_names]
        idx2 = [consensus_map.sample_names.index(s) for s in group2_samples
                if s in consensus_map.sample_names]

        if len(idx1) < 2 or len(idx2) < 2:
            raise ValueError("Each group needs at least 2 samples for t-test")

        results = []
        pvalues = []

        for i in range(consensus_map.n_features):
            vals1 = consensus_map.intensity_matrix[i, idx1]
            vals2 = consensus_map.intensity_matrix[i, idx2]

            mean1 = np.mean(vals1)
            mean2 = np.mean(vals2)

            # Log2 fold change
            if mean1 > 0 and mean2 > 0:
                log2_fc = np.log2(mean2 / mean1)
            elif mean2 > 0:
                log2_fc = float("inf")
            elif mean1 > 0:
                log2_fc = float("-inf")
            else:
                log2_fc = 0.0

            # Welch's t-test
            if np.std(vals1) == 0 and np.std(vals2) == 0:
                pvalue = 1.0
            else:
                _, pvalue = stats.ttest_ind(vals1, vals2, equal_var=False)
                if np.isnan(pvalue):
                    pvalue = 1.0

            results.append(DifferentialFeature(
                feature_index=i,
                log2_fc=log2_fc,
                pvalue=pvalue,
                mean_group1=mean1,
                mean_group2=mean2,
            ))
            pvalues.append(pvalue)

        # Benjamini-Hochberg correction
        adjusted = self._benjamini_hochberg(pvalues)
        for i, result in enumerate(results):
            result.adjusted_pvalue = adjusted[i]
            result.significant = (
                adjusted[i] < self.alpha
                and abs(result.log2_fc) >= self.min_fold_change
            )

        return results

    @staticmethod
    def _benjamini_hochberg(pvalues: List[float]) -> List[float]:
        """Benjamini-Hochberg FDR correction."""
        n = len(pvalues)
        if n == 0:
            return []

        indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
        adjusted = [0.0] * n

        prev_adj = 1.0
        for rank_idx in range(n - 1, -1, -1):
            orig_idx, pval = indexed[rank_idx]
            rank = rank_idx + 1
            adj = min(prev_adj, pval * n / rank)
            adj = min(adj, 1.0)
            adjusted[orig_idx] = adj
            prev_adj = adj

        return adjusted


# =========================================================================
# Normalization functions
# =========================================================================

def median_normalization(matrix: np.ndarray) -> np.ndarray:
    """
    Median ratio normalization.

    Scales each sample so that the median intensity across features is equal.
    """
    result = matrix.copy()
    medians = np.zeros(matrix.shape[1])

    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        nonzero = col[col > 0]
        medians[j] = np.median(nonzero) if len(nonzero) > 0 else 1.0

    target_median = np.median(medians[medians > 0]) if np.any(medians > 0) else 1.0

    for j in range(matrix.shape[1]):
        if medians[j] > 0:
            result[:, j] *= target_median / medians[j]

    return result


def quantile_normalization(matrix: np.ndarray) -> np.ndarray:
    """
    Quantile normalization.

    Forces the distribution of intensities to be the same across all samples.
    """
    result = matrix.copy()
    n_features, n_samples = matrix.shape

    # Rank each column
    ranks = np.zeros_like(matrix)
    sorted_vals = np.zeros_like(matrix)

    for j in range(n_samples):
        order = np.argsort(matrix[:, j])
        sorted_vals[:, j] = matrix[order, j]
        ranks[order, j] = np.arange(n_features)

    # Compute row means of sorted values
    row_means = np.mean(sorted_vals, axis=1)

    # Replace values with row means at corresponding ranks
    for j in range(n_samples):
        result[:, j] = row_means[ranks[:, j].astype(int)]

    return result


def tic_normalization(
    matrix: np.ndarray,
    tic_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    TIC-based normalization.

    Args:
        matrix: Intensity matrix (features x samples)
        tic_values: Optional pre-computed TIC values per sample.
                    If None, computed as column sums.
    """
    result = matrix.copy()

    if tic_values is None:
        tic_values = np.sum(matrix, axis=0)

    target_tic = np.mean(tic_values[tic_values > 0]) if np.any(tic_values > 0) else 1.0

    for j in range(matrix.shape[1]):
        if tic_values[j] > 0:
            result[:, j] *= target_tic / tic_values[j]

    return result
