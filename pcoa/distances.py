"""
Pairwise dissimilarity measures for community (OTU) abundance tables.

Every metric here is implemented from first principles with NumPy so that the
PCoA pipeline is self-contained; `scipy.spatial.distance` is used only as an
independent cross-check inside the test-suite.

Design notes
------------
* All routines take a samples x features matrix ``X`` (rows = samples) and
  return a dense, symmetric ``(n, n)`` distance matrix with a zero diagonal.
* The naive broadcast ``|X[:, None, :] - X[None, :, :]|`` materialises an
  ``n x n x p`` array.  For the assignment data (675 x 6696) that is ~24 GB,
  so every O(n^2 p) metric is evaluated in **row blocks** whose size is
  derived from an explicit memory budget (see ``_rows_per_block``).
* Metrics that can be written in terms of inner products (Euclidean,
  Hellinger, Aitchison, Jaccard) use the Gram-matrix identity instead and are
  therefore O(n^2 p) in FLOPs but O(n^2 + np) in memory.

References
----------
Bray & Curtis (1957) Ecol. Monogr. 27:325-349.
Legendre & Legendre (2012) Numerical Ecology, 3rd ed., Ch. 7.
Aitchison (1986) The Statistical Analysis of Compositional Data.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pairwise_distances",
    "bray_curtis",
    "euclidean",
    "jaccard",
    "hellinger",
    "aitchison",
    "jensen_shannon",
    "AVAILABLE_METRICS",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_matrix(X) -> np.ndarray:
    """Validate and coerce the input to a 2-D float64 C-contiguous array."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D (samples x features) array, got ndim={X.ndim}")
    if X.shape[0] < 2:
        raise ValueError("need at least two samples to compute a distance matrix")
    if not np.all(np.isfinite(X)):
        raise ValueError("input contains NaN or inf; clean the table before use")
    return np.ascontiguousarray(X)


def _rows_per_block(n: int, p: int, memory_budget_mb: float) -> int:
    """
    How many rows can be broadcast at once against the full table while keeping
    the temporary ``(b, n, p)`` float64 array inside ``memory_budget_mb``?
    """
    bytes_per_row_block = 8.0 * n * p
    b = int(memory_budget_mb * 1024 ** 2 / max(bytes_per_row_block, 1.0))
    return max(1, min(b, n))


def _symmetrise(D: np.ndarray) -> np.ndarray:
    """Force exact symmetry and a zero diagonal (kills float round-off)."""
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def _gram_to_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Euclidean distances via the identity  d_ij^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i.x_j.

    O(n^2 p) FLOPs but only O(n^2) extra memory, and it hands the work to BLAS.
    """
    G = X @ X.T
    sq = np.diag(G).copy()
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(D2, 0.0, out=D2)          # clip tiny negative round-off
    return _symmetrise(np.sqrt(D2))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def euclidean(X, **_) -> np.ndarray:
    """Plain Euclidean distance. PCoA on this is algebraically identical to PCA."""
    return _gram_to_euclidean(_as_matrix(X))


def bray_curtis(X, memory_budget_mb: float = 256.0, **_) -> np.ndarray:
    r"""
    Bray-Curtis dissimilarity.

    .. math:: d_{ij} = \frac{\sum_k |x_{ik} - x_{jk}|}{\sum_k (x_{ik} + x_{jk})}

    Requires non-negative data.  Note that for **closed** compositional rows
    (each row sums to 1) the denominator is constant at 2, so Bray-Curtis
    degenerates to half the L1 distance - a useful sanity check on this dataset.
    """
    X = _as_matrix(X)
    if np.any(X < 0):
        raise ValueError("Bray-Curtis is undefined for negative abundances")
    n, p = X.shape
    totals = X.sum(axis=1)
    D = np.zeros((n, n), dtype=np.float64)
    b = _rows_per_block(n, p, memory_budget_mb)
    for start in range(0, n, b):
        stop = min(start + b, n)
        num = np.abs(X[start:stop, None, :] - X[None, :, :]).sum(axis=2)
        den = totals[start:stop, None] + totals[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            D[start:stop] = np.where(den > 0, num / den, 0.0)
    return _symmetrise(D)


def jaccard(X, presence_threshold: float = 0.0, **_) -> np.ndarray:
    r"""
    Binary (presence/absence) Jaccard dissimilarity.

    .. math:: d_{ij} = 1 - \frac{|A \cap B|}{|A \cup B|}

    ``presence_threshold`` is the abundance above which a taxon counts as
    present.  Set it above the zero-replacement floor when working with
    imputed compositional data, otherwise every taxon is "present" everywhere
    and the metric collapses to zero.
    """
    X = _as_matrix(X)
    B = (X > presence_threshold).astype(np.float64)
    inter = B @ B.T
    counts = np.diag(inter).copy()
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(union > 0, 1.0 - inter / union, 0.0)
    return _symmetrise(D)


def hellinger(X, **_) -> np.ndarray:
    r"""
    Hellinger distance: Euclidean distance on square-rooted relative abundances.

    .. math:: d_{ij} = \sqrt{\sum_k \left(\sqrt{p_{ik}} - \sqrt{p_{jk}}\right)^2}

    Euclidean-embeddable by construction, so PCoA on it yields no negative
    eigenvalues.  Standard in ecology for down-weighting dominant taxa.
    """
    X = _as_matrix(X)
    if np.any(X < 0):
        raise ValueError("Hellinger is undefined for negative abundances")
    totals = X.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return _gram_to_euclidean(np.sqrt(X / totals))


def aitchison(X, pseudocount: float | None = None, **_) -> np.ndarray:
    r"""
    Aitchison distance: Euclidean distance in centred-log-ratio (clr) space.

    .. math:: \mathrm{clr}(x)_k = \log x_k - \frac{1}{p}\sum_j \log x_j

    This is the scale-invariant metric for compositional data.  Zeros must be
    replaced first; pass ``pseudocount`` to do it here, or supply already
    zero-replaced data (as in the assignment CSVs).
    """
    X = _as_matrix(X)
    if pseudocount is not None:
        X = X + float(pseudocount)
    if np.any(X <= 0):
        raise ValueError(
            "Aitchison needs strictly positive data; supply a pseudocount or "
            "use a zero-replacement method first"
        )
    L = np.log(X)
    clr = L - L.mean(axis=1, keepdims=True)
    return _gram_to_euclidean(clr)


def jensen_shannon(X, memory_budget_mb: float = 256.0, **_) -> np.ndarray:
    r"""
    Jensen-Shannon distance, the square root of the JS divergence (base 2).

    Bounded in [0, 1] and a true metric; a probabilistic alternative to
    Bray-Curtis for compositional profiles.
    """
    X = _as_matrix(X)
    if np.any(X < 0):
        raise ValueError("Jensen-Shannon is undefined for negative values")
    totals = X.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    P = X / totals
    n, p = P.shape
    D = np.zeros((n, n))
    b = _rows_per_block(n, p, memory_budget_mb / 3.0)  # 3 temporaries in flight

    def _xlog2x(a):
        return np.where(a > 0, a * np.log2(np.where(a > 0, a, 1.0)), 0.0)

    ent = -_xlog2x(P).sum(axis=1)  # Shannon entropy of each profile
    for start in range(0, n, b):
        stop = min(start + b, n)
        M = 0.5 * (P[start:stop, None, :] + P[None, :, :])
        ent_M = -_xlog2x(M).sum(axis=2)
        jsd = ent_M - 0.5 * (ent[start:stop, None] + ent[None, :])
        D[start:stop] = np.sqrt(np.clip(jsd, 0.0, None))
    return _symmetrise(D)


AVAILABLE_METRICS = {
    "braycurtis": bray_curtis,
    "euclidean": euclidean,
    "jaccard": jaccard,
    "hellinger": hellinger,
    "aitchison": aitchison,
    "jensenshannon": jensen_shannon,
}


def pairwise_distances(X, metric: str = "braycurtis", **kwargs) -> np.ndarray:
    """
    Dispatch to a named metric.

    Parameters
    ----------
    X : array (n_samples, n_features)
    metric : {'braycurtis','euclidean','jaccard','hellinger','aitchison','jensenshannon'}
    **kwargs : forwarded to the metric (e.g. ``memory_budget_mb``,
        ``presence_threshold``, ``pseudocount``).

    Returns
    -------
    D : ndarray (n_samples, n_samples), symmetric, zero diagonal.
    """
    key = metric.lower().replace("-", "").replace("_", "")
    if key not in AVAILABLE_METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; choose from {sorted(AVAILABLE_METRICS)}"
        )
    return AVAILABLE_METRICS[key](X, **kwargs)
