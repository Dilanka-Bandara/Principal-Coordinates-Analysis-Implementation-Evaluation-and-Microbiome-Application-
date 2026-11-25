"""
Quantitative evaluation of a PCoA ordination.

Three families of question are answered here:

*Embedding quality*  - how faithfully do k principal coordinates reproduce the
  original dissimilarities?  (``kruskal_stress``, ``reconstruction_error``,
  ``shepard_correlation``, ``procrustes_m2``)

*Statistical structure* - is the grouping we see in the plot more than noise?
  (``permanova``, ``anosim``, ``mantel_test``)

*Cluster recovery* - do unsupervised clusters in the ordination agree with the
  known biological labels?  (``silhouette_from_distances``, ``cluster_agreement``)

All permutation tests take a ``random_state`` so results are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PermutationTestResult",
    "permanova",
    "anosim",
    "mantel_test",
    "kruskal_stress",
    "reconstruction_error",
    "shepard_correlation",
    "procrustes_m2",
    "silhouette_from_distances",
    "cluster_agreement",
]


def _condensed(D: np.ndarray) -> np.ndarray:
    """Upper-triangular (i < j) entries of a square matrix, flattened."""
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


@dataclass
class PermutationTestResult:
    statistic: float
    p_value: float
    effect_size: float
    n_permutations: int
    name: str

    def __str__(self) -> str:
        return (
            f"{self.name}: statistic={self.statistic:.4f}, "
            f"effect size={self.effect_size:.4f}, "
            f"p={self.p_value:.4g} ({self.n_permutations} permutations)"
        )


# --------------------------------------------------------------------------- #
# statistical structure
# --------------------------------------------------------------------------- #
def permanova(
    D: np.ndarray,
    groups,
    n_permutations: int = 999,
    random_state: int | None = 0,
) -> PermutationTestResult:
    r"""
    PERMANOVA (Anderson, 2001): permutational multivariate ANOVA on distances.

    Partitions the total sum of squares of the distance matrix into
    between- and within-group components and compares them with a pseudo-F:

    .. math:: F = \frac{\mathrm{tr}(HGH)/(a-1)}{\mathrm{tr}((I-H)G(I-H))/(n-a)}

    where ``G`` is Gower's centred matrix and ``H`` the hat matrix of the
    group design.  The null distribution is built by permuting group labels,
    which is valid without any multivariate-normality assumption.

    ``effect_size`` is :math:`R^2` = SS_between / SS_total, i.e. the fraction of
    multivariate variation attributable to the grouping.
    """
    from .core import gower_center

    rng = np.random.default_rng(random_state)
    groups = np.asarray(groups)
    n = D.shape[0]
    if groups.shape[0] != n:
        raise ValueError("groups and distance matrix disagree on n")
    levels = np.unique(groups)
    a = levels.size
    if a < 2:
        raise ValueError("PERMANOVA needs at least two groups")
    if a >= n:
        raise ValueError("need more samples than groups")

    G = gower_center(-0.5 * D ** 2)
    ss_total = np.trace(G)

    # tr(HGH) = tr(HG) because H is idempotent and the trace is cyclic.  For a
    # group-indicator design H is block-constant, so the whole statistic
    # reduces to group-wise sums of G: an O(n^2 a) step per permutation instead
    # of the O(n^3) cost of forming H G H explicitly.  With 999 permutations on
    # 675 samples that is the difference between seconds and minutes.
    def _pseudo_f(labels: np.ndarray) -> tuple[float, float]:
        X = (labels[:, None] == levels[None, :]).astype(np.float64)
        sizes = X.sum(axis=0)
        block = X.T @ G @ X                       # (a, a); diagonal = within-group sums
        ss_between = float(np.sum(np.diag(block) / np.where(sizes > 0, sizes, 1.0)))
        ss_within = ss_total - ss_between
        f = (ss_between / (a - 1)) / (ss_within / (n - a))
        return f, ss_between / ss_total

    f_obs, r2 = _pseudo_f(groups)

    count = 1  # +1 for the observed value (Hope, 1968 - avoids p = 0)
    perm = groups.copy()
    for _ in range(n_permutations):
        rng.shuffle(perm)
        f_perm, _ = _pseudo_f(perm)
        if f_perm >= f_obs:
            count += 1
    p = count / (n_permutations + 1)

    return PermutationTestResult(f_obs, p, r2, n_permutations, "PERMANOVA (pseudo-F)")


def anosim(
    D: np.ndarray,
    groups,
    n_permutations: int = 999,
    random_state: int | None = 0,
) -> PermutationTestResult:
    r"""
    ANOSIM (Clarke, 1993): rank-based test of group separation.

    .. math:: R = \frac{\bar r_B - \bar r_W}{n(n-1)/4}

    ``R`` close to 1 means between-group dissimilarities are consistently
    larger than within-group ones; ``R`` near 0 means no separation.
    Rank-based, so it is insensitive to the distance scale - a useful
    robustness check against PERMANOVA, which uses raw magnitudes.
    """
    rng = np.random.default_rng(random_state)
    groups = np.asarray(groups)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    d = D[iu]
    ranks = np.argsort(np.argsort(d)) + 1.0  # 1-based ranks, ties broken arbitrarily
    denom = n * (n - 1) / 4.0

    def _r(labels: np.ndarray) -> float:
        within = labels[iu[0]] == labels[iu[1]]
        return float((ranks[~within].mean() - ranks[within].mean()) / denom)

    r_obs = _r(groups)
    count = 1
    perm = groups.copy()
    for _ in range(n_permutations):
        rng.shuffle(perm)
        if _r(perm) >= r_obs:
            count += 1
    p = count / (n_permutations + 1)
    return PermutationTestResult(r_obs, p, r_obs, n_permutations, "ANOSIM (R)")


def mantel_test(
    D1: np.ndarray,
    D2: np.ndarray,
    n_permutations: int = 999,
    random_state: int | None = 0,
) -> PermutationTestResult:
    """
    Mantel test: Pearson correlation between two distance matrices, with a
    p-value from permuting the rows/columns of one of them.

    Used here to ask how much of the original dissimilarity structure survives
    in the reduced-dimension reconstruction, and to compare metrics with each
    other.
    """
    rng = np.random.default_rng(random_state)
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = D1[iu], D2[iu]
    r_obs = float(np.corrcoef(a, b)[0, 1])

    idx = np.arange(n)
    count = 1
    for _ in range(n_permutations):
        rng.shuffle(idx)
        bb = D2[np.ix_(idx, idx)][iu]
        if float(np.corrcoef(a, bb)[0, 1]) >= r_obs:
            count += 1
    p = count / (n_permutations + 1)
    return PermutationTestResult(r_obs, p, r_obs ** 2, n_permutations, "Mantel (r)")


# --------------------------------------------------------------------------- #
# embedding quality
# --------------------------------------------------------------------------- #
def kruskal_stress(D_original: np.ndarray, D_embedded: np.ndarray) -> float:
    r"""
    Kruskal's stress-1.

    .. math:: S_1 = \sqrt{\frac{\sum_{i<j}(d_{ij} - \hat d_{ij})^2}{\sum_{i<j} d_{ij}^2}}

    Conventional reading: <0.05 excellent, <0.10 good, <0.20 fair, >0.20 poor.
    """
    d, dh = _condensed(D_original), _condensed(D_embedded)
    denom = np.sum(d ** 2)
    return float(np.sqrt(np.sum((d - dh) ** 2) / denom)) if denom > 0 else 0.0


def reconstruction_error(D_original: np.ndarray, D_embedded: np.ndarray) -> dict:
    """Frobenius / RMSE / MAE / relative error between two distance matrices."""
    d, dh = _condensed(D_original), _condensed(D_embedded)
    diff = d - dh
    return {
        "frobenius": float(np.linalg.norm(diff)),
        "relative_frobenius": float(np.linalg.norm(diff) / np.linalg.norm(d)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
    }


def shepard_correlation(D_original: np.ndarray, D_embedded: np.ndarray) -> dict:
    """Pearson and Spearman correlation of the Shepard diagram (d vs d-hat)."""
    from scipy.stats import spearmanr

    d, dh = _condensed(D_original), _condensed(D_embedded)
    return {
        "pearson_r": float(np.corrcoef(d, dh)[0, 1]),
        "spearman_rho": float(spearmanr(d, dh).statistic),
        "r_squared": float(np.corrcoef(d, dh)[0, 1] ** 2),
    }


def procrustes_m2(Y1: np.ndarray, Y2: np.ndarray) -> float:
    r"""
    Procrustes :math:`m^2` between two configurations after optimal
    translation, uniform scaling and rotation/reflection.

    0 = identical up to a similarity transform.  Used to prove that our PCoA
    agrees with an independent reference implementation, and that PCoA on
    Euclidean distances equals PCA.
    """
    k = min(Y1.shape[1], Y2.shape[1])
    A = Y1[:, :k] - Y1[:, :k].mean(axis=0)
    B = Y2[:, :k] - Y2[:, :k].mean(axis=0)
    A /= np.linalg.norm(A)
    B /= np.linalg.norm(B)
    U, s, Vt = np.linalg.svd(A.T @ B)
    return float(1.0 - s.sum() ** 2)


# --------------------------------------------------------------------------- #
# cluster recovery
# --------------------------------------------------------------------------- #
def silhouette_from_distances(D: np.ndarray, labels) -> dict:
    """
    Silhouette coefficient computed directly on the dissimilarity matrix
    (no coordinates needed), plus the per-group means.

    Range [-1, 1]; >0 means samples sit closer to their own group than to the
    nearest other group.  Microbiome silhouettes are typically small even when
    PERMANOVA is highly significant - the groups overlap but their centroids
    still differ, which is worth saying explicitly in the interpretation.
    """
    labels = np.asarray(labels)
    levels = np.unique(labels)
    n = D.shape[0]
    sil = np.zeros(n)
    for i in range(n):
        own = labels[i]
        same = labels == own
        same[i] = False
        if same.sum() == 0:
            sil[i] = 0.0
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == lv].mean() for lv in levels if lv != own)
        sil[i] = (b - a) / max(a, b)
    return {
        "mean": float(sil.mean()),
        "per_group": {str(lv): float(sil[labels == lv].mean()) for lv in levels},
        "values": sil,
    }


def cluster_agreement(coords: np.ndarray, labels, n_clusters: int | None = None,
                      random_state: int = 0) -> dict:
    """
    k-means on the principal coordinates, scored against the known labels with
    the adjusted Rand index and adjusted mutual information.

    This is the "did the ordination actually recover the biology?" check: a
    high ARI means the geometry alone is enough to rediscover the design factor.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    labels = np.asarray(labels)
    k = n_clusters or np.unique(labels).size
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(coords)
    return {
        "n_clusters": int(k),
        "adjusted_rand_index": float(adjusted_rand_score(labels, km.labels_)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(labels, km.labels_)),
        "assignments": km.labels_,
    }
