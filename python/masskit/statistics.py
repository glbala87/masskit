"""
Statistical analysis module for LC-MS data.

Provides PCA, PLS-DA, hierarchical clustering, ANOVA,
volcano plots, and other multivariate statistical methods.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import numpy as np

from .quantification import ConsensusMap


@dataclass
class PCAResult:
    """Result of Principal Component Analysis."""
    scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    loadings: np.ndarray = field(default_factory=lambda: np.empty(0))
    explained_variance: np.ndarray = field(default_factory=lambda: np.empty(0))
    explained_variance_ratio: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_components: int = 0
    sample_names: List[str] = field(default_factory=list)
    feature_ids: List[str] = field(default_factory=list)

    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)


@dataclass
class PLSDAResult:
    """Result of Partial Least Squares Discriminant Analysis."""
    scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    loadings: np.ndarray = field(default_factory=lambda: np.empty(0))
    vip_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_components: int = 0
    r2: float = 0.0
    q2: float = 0.0
    sample_names: List[str] = field(default_factory=list)
    group_labels: List[str] = field(default_factory=list)


@dataclass
class ClusterResult:
    """Result of hierarchical clustering."""
    labels: np.ndarray = field(default_factory=lambda: np.empty(0))
    linkage_matrix: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_clusters: int = 0
    sample_names: List[str] = field(default_factory=list)
    dendrogram_data: Optional[Dict] = None


@dataclass
class ANOVAResult:
    """Result of ANOVA analysis for a single feature."""
    feature_index: int = 0
    f_statistic: float = 0.0
    pvalue: float = 1.0
    adjusted_pvalue: float = 1.0
    group_means: Dict[str, float] = field(default_factory=dict)
    significant: bool = False


def pca(
    consensus_map: ConsensusMap,
    n_components: int = 2,
    scale: bool = True,
) -> PCAResult:
    """
    Perform Principal Component Analysis on a consensus map.

    Args:
        consensus_map: Feature intensity matrix
        n_components: Number of components
        scale: Auto-scale (mean-center and unit variance)

    Returns:
        PCAResult with scores, loadings, and variance explained

    Example:
        >>> result = pca(consensus, n_components=3)
        >>> print(f"PC1 explains {result.explained_variance_ratio[0]:.1%}")
    """
    X = consensus_map.intensity_matrix.T.copy()  # samples x features
    n_samples, n_features = X.shape

    n_components = min(n_components, n_samples, n_features)

    # Center
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Scale
    if scale:
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        X_centered /= std

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    scores = U[:, :n_components] * S[:n_components]
    loadings = Vt[:n_components, :].T

    total_var = np.sum(S ** 2)
    explained_var = S[:n_components] ** 2
    explained_ratio = explained_var / total_var if total_var > 0 else np.zeros(n_components)

    return PCAResult(
        scores=scores,
        loadings=loadings,
        explained_variance=explained_var,
        explained_variance_ratio=explained_ratio,
        n_components=n_components,
        sample_names=consensus_map.sample_names,
        feature_ids=consensus_map.feature_ids,
    )


def plsda(
    consensus_map: ConsensusMap,
    group_labels: List[str],
    n_components: int = 2,
) -> PLSDAResult:
    """
    Perform PLS-DA (Partial Least Squares Discriminant Analysis).

    Args:
        consensus_map: Feature intensity matrix
        group_labels: Group label for each sample
        n_components: Number of latent variables

    Returns:
        PLSDAResult with scores, loadings, and VIP scores
    """
    X = consensus_map.intensity_matrix.T.copy()  # samples x features
    n_samples, n_features = X.shape

    # Encode group labels as dummy matrix
    unique_groups = sorted(set(group_labels))
    Y = np.zeros((n_samples, len(unique_groups)))
    for i, label in enumerate(group_labels):
        Y[i, unique_groups.index(label)] = 1.0

    # Center
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # NIPALS PLS algorithm
    T = np.zeros((n_samples, n_components))  # X scores
    P = np.zeros((n_features, n_components))  # X loadings
    W = np.zeros((n_features, n_components))  # X weights
    Q = np.zeros((Y.shape[1], n_components))  # Y loadings

    Xk = X_centered.copy()
    Yk = Y_centered.copy()

    for k in range(n_components):
        # Initial u = first column of Y
        u = Yk[:, 0].copy()

        for _ in range(100):  # Max iterations
            # X weight
            w = Xk.T @ u
            w_norm = np.linalg.norm(w)
            if w_norm == 0:
                break
            w /= w_norm

            # X score
            t = Xk @ w

            # Y loading
            q = Yk.T @ t
            t_norm = t @ t
            if t_norm == 0:
                break
            q /= t_norm

            # Y score
            u_new = Yk @ q
            q_norm = q @ q
            if q_norm == 0:
                break
            u_new /= q_norm

            if np.linalg.norm(u_new - u) < 1e-10:
                u = u_new
                break
            u = u_new

        # X loading
        p = Xk.T @ t / (t @ t) if (t @ t) > 0 else np.zeros(n_features)

        T[:, k] = t
        P[:, k] = p
        W[:, k] = w
        Q[:, k] = q

        # Deflate
        Xk -= np.outer(t, p)
        Yk -= np.outer(t, q)

    # VIP scores
    vip = _compute_vip(W, T, Q, Y_centered)

    # R2
    Y_pred = T @ Q.T + Y_mean
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - Y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return PLSDAResult(
        scores=T,
        loadings=P,
        vip_scores=vip,
        n_components=n_components,
        r2=r2,
        sample_names=consensus_map.sample_names,
        group_labels=group_labels,
    )


def _compute_vip(W, T, Q, Y):
    """Compute Variable Importance in Projection scores."""
    n_features = W.shape[0]
    n_components = W.shape[1]

    ss_y = np.zeros(n_components)
    for k in range(n_components):
        ss_y[k] = np.sum((np.outer(T[:, k], Q[:, k])) ** 2)

    total_ss = np.sum(ss_y)
    if total_ss == 0:
        return np.ones(n_features)

    vip = np.zeros(n_features)
    for j in range(n_features):
        s = 0.0
        for k in range(n_components):
            w_norm = np.linalg.norm(W[:, k])
            if w_norm > 0:
                s += (W[j, k] / w_norm) ** 2 * ss_y[k]
        vip[j] = np.sqrt(n_features * s / total_ss)

    return vip


def hierarchical_clustering(
    consensus_map: ConsensusMap,
    method: str = "ward",
    metric: str = "euclidean",
    n_clusters: int = 2,
) -> ClusterResult:
    """
    Hierarchical clustering of samples.

    Args:
        consensus_map: Feature intensity matrix
        method: Linkage method ('ward', 'complete', 'average', 'single')
        metric: Distance metric ('euclidean', 'correlation', 'cosine')
        n_clusters: Number of clusters to cut

    Returns:
        ClusterResult with labels and linkage matrix
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    X = consensus_map.intensity_matrix.T  # samples x features

    # Compute distance matrix
    if metric == "correlation":
        dist = pdist(X, metric="correlation")
    elif metric == "cosine":
        dist = pdist(X, metric="cosine")
    else:
        dist = pdist(X, metric="euclidean")

    # Handle NaN distances
    dist = np.nan_to_num(dist, nan=0.0)

    # Linkage
    Z = linkage(dist, method=method)

    # Cut
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    return ClusterResult(
        labels=labels,
        linkage_matrix=Z,
        n_clusters=n_clusters,
        sample_names=consensus_map.sample_names,
    )


def anova(
    consensus_map: ConsensusMap,
    group_labels: List[str],
    alpha: float = 0.05,
) -> List[ANOVAResult]:
    """
    One-way ANOVA across multiple groups.

    Args:
        consensus_map: Feature intensity matrix
        group_labels: Group label for each sample
        alpha: Significance level

    Returns:
        List of ANOVAResult for each feature
    """
    from scipy import stats

    unique_groups = sorted(set(group_labels))
    group_indices = {}
    for g in unique_groups:
        group_indices[g] = [i for i, l in enumerate(group_labels) if l == g]

    results = []
    pvalues = []

    for i in range(consensus_map.n_features):
        groups_data = []
        group_means = {}

        for g in unique_groups:
            vals = consensus_map.intensity_matrix[i, group_indices[g]]
            groups_data.append(vals)
            group_means[g] = float(np.mean(vals))

        if len(groups_data) < 2 or any(len(g) < 2 for g in groups_data):
            f_stat, pval = 0.0, 1.0
        else:
            f_stat, pval = stats.f_oneway(*groups_data)
            if np.isnan(pval):
                pval = 1.0

        results.append(ANOVAResult(
            feature_index=i,
            f_statistic=float(f_stat),
            pvalue=float(pval),
            group_means=group_means,
        ))
        pvalues.append(float(pval))

    # BH correction
    adjusted = _benjamini_hochberg(pvalues)
    for i, r in enumerate(results):
        r.adjusted_pvalue = adjusted[i]
        r.significant = adjusted[i] < alpha

    return results


def volcano_data(
    consensus_map: ConsensusMap,
    group1_samples: List[str],
    group2_samples: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute volcano plot data (log2FC vs -log10 p-value).

    Args:
        consensus_map: Feature intensity matrix
        group1_samples: Sample names in group 1
        group2_samples: Sample names in group 2

    Returns:
        Tuple of (log2_fc, neg_log10_pvalue, feature_indices)
    """
    from scipy import stats

    idx1 = [consensus_map.sample_names.index(s) for s in group1_samples
            if s in consensus_map.sample_names]
    idx2 = [consensus_map.sample_names.index(s) for s in group2_samples
            if s in consensus_map.sample_names]

    log2_fc = np.zeros(consensus_map.n_features)
    pvalues = np.zeros(consensus_map.n_features)

    for i in range(consensus_map.n_features):
        vals1 = consensus_map.intensity_matrix[i, idx1]
        vals2 = consensus_map.intensity_matrix[i, idx2]

        mean1 = np.mean(vals1)
        mean2 = np.mean(vals2)

        if mean1 > 0 and mean2 > 0:
            log2_fc[i] = np.log2(mean2 / mean1)
        elif mean2 > 0:
            log2_fc[i] = 10.0
        elif mean1 > 0:
            log2_fc[i] = -10.0

        if np.std(vals1) == 0 and np.std(vals2) == 0:
            pvalues[i] = 1.0
        else:
            _, pval = stats.ttest_ind(vals1, vals2, equal_var=False)
            pvalues[i] = pval if not np.isnan(pval) else 1.0

    neg_log10_p = -np.log10(np.maximum(pvalues, 1e-300))
    feature_indices = np.arange(consensus_map.n_features)

    return (log2_fc, neg_log10_p, feature_indices)


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
        adj = min(prev_adj, pval * n / rank, 1.0)
        adjusted[orig_idx] = adj
        prev_adj = adj
    return adjusted
