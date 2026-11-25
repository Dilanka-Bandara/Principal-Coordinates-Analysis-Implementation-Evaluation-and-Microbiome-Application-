"""
Principal Coordinates Analysis (PCoA / classical metric MDS).

The algorithm, exactly as specified in Gower (1966) and restated in
Shi et al. (2020), the paper the assignment dataset comes from:

    1.  A  = -1/2 * D^2                      (element-wise on the distances)
    2.  G  = (I - 11'/n) A (I - 11'/n)       Gower's doubly-centred matrix
    3.  G  = U L U'                          symmetric eigendecomposition
    4.  Y  = U_k L_k^{1/2}                   coordinates on the k leading axes

Key facts the implementation relies on
--------------------------------------
* ``G`` is real symmetric, so ``numpy.linalg.eigh`` is used rather than the
  general ``eig``: it is ~2x faster, numerically stable, and returns real
  eigenvalues in ascending order.
* If ``D`` is Euclidean-embeddable, ``G`` is positive semi-definite and
  ``G = Y_c Y_c'``.  Non-Euclidean dissimilarities (Bray-Curtis, Jaccard, ...)
  produce **negative eigenvalues**, which have no real-coordinate
  representation.  We quantify them and optionally remove them with the
  Lingoes (1971) or Cailliez (1983) additive constant corrections.
* ``G`` always has at least one zero eigenvalue because the centring matrix
  annihilates the constant vector; the rank of a valid configuration is at
  most n-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["PCoAResult", "pcoa", "gower_center", "cailliez_constant", "lingoes_constant"]


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class PCoAResult:
    """Everything a downstream step (plotting, testing, evaluation) needs."""

    coordinates: np.ndarray          # (n, k) sample scores on the retained axes
    eigenvalues: np.ndarray          # (n,) all eigenvalues, descending
    proportion_explained: np.ndarray # (k,) fraction of variation per retained axis
    gower: np.ndarray                # (n, n) Gower-centred matrix
    n_negative: int                  # how many eigenvalues fell below tolerance
    negative_fraction: float         # sum|neg| / sum|all| - non-Euclideanity index
    correction: Optional[str] = None # 'lingoes' | 'cailliez' | None
    correction_constant: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.coordinates.shape[0]

    @property
    def n_axes(self) -> int:
        return self.coordinates.shape[1]

    def cumulative_explained(self) -> np.ndarray:
        return np.cumsum(self.proportion_explained)

    def reconstructed_distances(self, n_axes: Optional[int] = None) -> np.ndarray:
        """Euclidean distances implied by the first ``n_axes`` principal coordinates."""
        Y = self.coordinates[:, : (n_axes or self.n_axes)]
        sq = (Y ** 2).sum(axis=1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (Y @ Y.T)
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2)
        np.fill_diagonal(D, 0.0)
        return 0.5 * (D + D.T)

    def summary(self, n_axes: int = 5) -> str:
        k = min(n_axes, self.n_axes)
        lines = [
            f"PCoA on {self.n_samples} samples, {self.n_axes} retained axes",
            f"  correction            : {self.correction or 'none'}"
            + (f" (c = {self.correction_constant:.6g})" if self.correction else ""),
            f"  negative eigenvalues  : {self.n_negative} "
            f"({self.negative_fraction:.2%} of |trace|)",
            "  axis   eigenvalue    % explained   cumulative %",
        ]
        cum = self.cumulative_explained()
        for i in range(k):
            lines.append(
                f"  PCo{i + 1:<4d} {self.eigenvalues[i]: .6e}   "
                f"{100 * self.proportion_explained[i]:8.3f}   {100 * cum[i]:10.3f}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# steps 1-2
# --------------------------------------------------------------------------- #
def _validate_distance_matrix(D: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distance matrix must be square, got shape {D.shape}")
    if not np.all(np.isfinite(D)):
        raise ValueError("distance matrix contains NaN or inf")
    if np.abs(D - D.T).max() > tol:
        raise ValueError("distance matrix is not symmetric within tolerance")
    if np.abs(np.diag(D)).max() > tol:
        raise ValueError("distance matrix has a non-zero diagonal")
    if D.min() < -tol:
        raise ValueError("distance matrix contains negative entries")
    return 0.5 * (D + D.T)


def gower_center(A: np.ndarray) -> np.ndarray:
    r"""
    Apply Gower's double centring :math:`G = (I - 11'/n) A (I - 11'/n)`.

    Implemented as row/column/grand-mean subtraction, which is O(n^2) instead
    of the O(n^3) cost of forming and multiplying the centring matrix::

        G_ij = A_ij - rowmean_i - colmean_j + grandmean
    """
    row = A.mean(axis=1, keepdims=True)
    col = A.mean(axis=0, keepdims=True)
    G = A - row - col + A.mean()
    return 0.5 * (G + G.T)  # enforce exact symmetry


# --------------------------------------------------------------------------- #
# additive-constant corrections for non-Euclidean dissimilarities
# --------------------------------------------------------------------------- #
def lingoes_constant(D: np.ndarray) -> float:
    """
    Lingoes (1971) constant: ``c = |lambda_min|`` of the Gower matrix of ``D``.

    The corrected distances are ``sqrt(D^2 + 2c)`` off the diagonal, which makes
    the configuration Euclidean.
    """
    G = gower_center(-0.5 * D ** 2)
    lam_min = np.linalg.eigvalsh(G)[0]
    return float(max(-lam_min, 0.0))


def cailliez_constant(D: np.ndarray) -> float:
    """
    Cailliez (1983) constant: smallest ``c`` such that ``D + c(1 - I)`` is Euclidean.

    ``c`` is the largest real eigenvalue of the ``2n x 2n`` block matrix

        [  0      2*G(D^2) ]
        [ -I     -4*G(D)   ]

    where ``G(.)`` denotes Gower centring of ``-1/2`` the (squared) distances.
    """
    n = D.shape[0]
    G2 = gower_center(-0.5 * D ** 2)
    G1 = gower_center(-0.5 * D)
    top = np.hstack([np.zeros((n, n)), 2.0 * G2])
    bot = np.hstack([-np.eye(n), -4.0 * G1])
    M = np.vstack([top, bot])
    lam = np.linalg.eigvals(M)
    return float(max(np.real(lam).max(), 0.0))


def _apply_correction(D: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    if method == "lingoes":
        c = lingoes_constant(D)
        Dc = np.sqrt(np.maximum(D ** 2 + 2.0 * c, 0.0))
    elif method == "cailliez":
        c = cailliez_constant(D)
        Dc = D + c
    else:
        raise ValueError(f"unknown correction {method!r}; use 'lingoes' or 'cailliez'")
    np.fill_diagonal(Dc, 0.0)
    return Dc, c


# --------------------------------------------------------------------------- #
# steps 3-4
# --------------------------------------------------------------------------- #
def pcoa(
    D: np.ndarray,
    n_components: Optional[int] = None,
    correction: Optional[str] = None,
    eigen_solver: str = "full",
    tol: float = 1e-10,
    scale_by_eigenvalue: bool = True,
) -> PCoAResult:
    """
    Run Principal Coordinates Analysis on a pairwise distance matrix.

    Parameters
    ----------
    D : array (n, n)
        Symmetric dissimilarity matrix with a zero diagonal.
    n_components : int, optional
        Number of principal coordinates to retain.  ``None`` keeps every axis
        with a positive eigenvalue.
    correction : {'lingoes', 'cailliez'}, optional
        Additive-constant correction applied to ``D`` before the decomposition
        to remove negative eigenvalues.
    eigen_solver : {'full', 'partial'}
        ``'full'`` uses ``numpy.linalg.eigh`` (all n eigenpairs, O(n^3)).
        ``'partial'`` uses ``scipy.sparse.linalg.eigsh`` to extract only the
        leading ``n_components`` eigenpairs - much cheaper for large n, but it
        cannot report the negative-eigenvalue diagnostics.
    tol : float
        Eigenvalues with ``|lambda| <= tol * max|lambda|`` are treated as zero.
    scale_by_eigenvalue : bool
        If True (default, and the standard convention) coordinates are
        ``U sqrt(L)`` so that inter-point Euclidean distances reproduce ``D``.
        If False, raw unit-norm eigenvectors are returned.

    Returns
    -------
    PCoAResult
    """
    D = _validate_distance_matrix(D)
    n = D.shape[0]

    correction_constant = 0.0
    if correction is not None:
        D, correction_constant = _apply_correction(D, correction)

    # --- step 1 & 2 -------------------------------------------------------- #
    A = -0.5 * (D ** 2)
    G = gower_center(A)

    # --- step 3 ------------------------------------------------------------ #
    if eigen_solver == "full":
        eigvals, eigvecs = np.linalg.eigh(G)          # ascending
        order = np.argsort(eigvals)[::-1]             # -> descending
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    elif eigen_solver == "partial":
        from scipy.sparse.linalg import eigsh

        k = min(n_components or 10, n - 2)
        eigvals, eigvecs = eigsh(G, k=k, which="LA")
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    else:
        raise ValueError(f"unknown eigen_solver {eigen_solver!r}")

    # deterministic sign convention: largest-magnitude loading is positive
    for j in range(eigvecs.shape[1]):
        if eigvecs[np.argmax(np.abs(eigvecs[:, j])), j] < 0:
            eigvecs[:, j] *= -1.0

    # --- diagnostics ------------------------------------------------------- #
    scale = np.abs(eigvals).max() if eigvals.size else 1.0
    zero_tol = tol * scale
    positive = eigvals > zero_tol
    negative = eigvals < -zero_tol
    n_negative = int(negative.sum())
    total_abs = np.abs(eigvals).sum()
    negative_fraction = float(np.abs(eigvals[negative]).sum() / total_abs) if total_abs else 0.0

    # --- step 4 ------------------------------------------------------------ #
    k = int(positive.sum()) if n_components is None else min(n_components, int(positive.sum()))
    if k == 0:
        raise RuntimeError("no positive eigenvalues: the distance matrix is degenerate")

    lam = eigvals[:k]
    U = eigvecs[:, :k]
    coords = U * np.sqrt(lam) if scale_by_eigenvalue else U

    # Proportion explained uses the positive eigenvalues as the denominator
    # (Legendre & Legendre's recommendation when negatives are left in place).
    denom = eigvals[positive].sum()
    proportion = lam / denom

    return PCoAResult(
        coordinates=coords,
        eigenvalues=eigvals,
        proportion_explained=proportion,
        gower=G,
        n_negative=n_negative,
        negative_fraction=negative_fraction,
        correction=correction,
        correction_constant=correction_constant,
        metadata={
            "n_samples": n,
            "eigen_solver": eigen_solver,
            "positive_eigenvalue_sum": float(denom),
            "abs_eigenvalue_sum": float(total_abs),
        },
    )


def apcoa(
    D: np.ndarray,
    covariates: np.ndarray,
    n_components: Optional[int] = None,
    add_intercept: bool = True,
) -> PCoAResult:
    r"""
    Covariate-adjusted PCoA (Shi et al., 2020, Bioinformatics 36:4099).

    Removes the linear effect of nuisance covariates ``X`` from Gower's matrix
    before the decomposition:

    .. math::
        H = X (X'X)^{-1} X', \qquad \Delta^* = (I - H)\, G\, (I - H)

    The projection of :math:`\Delta^*` is then read exactly like a normal PCoA
    plot, but with the confounder's structure projected out.  Use it when a
    nuisance factor (cage, donor, sequencing run) visually dominates the
    ordination and masks the effect of interest.

    Parameters
    ----------
    D : array (n, n)          pairwise distance matrix
    covariates : array (n, s) design matrix of nuisance covariates
                              (dummy-code categoricals before passing them in)
    """
    D = _validate_distance_matrix(D)
    X = np.asarray(covariates, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != D.shape[0]:
        raise ValueError("covariate matrix and distance matrix disagree on n")
    if add_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    G = gower_center(-0.5 * D ** 2)
    # pseudo-inverse keeps us safe against collinear dummy columns
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    R = np.eye(X.shape[0]) - H
    delta = R @ G @ R
    delta = 0.5 * (delta + delta.T)

    eigvals, eigvecs = np.linalg.eigh(delta)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    for j in range(eigvecs.shape[1]):
        if eigvecs[np.argmax(np.abs(eigvecs[:, j])), j] < 0:
            eigvecs[:, j] *= -1.0

    scale = np.abs(eigvals).max()
    positive = eigvals > 1e-10 * scale
    k = int(positive.sum()) if n_components is None else min(n_components, int(positive.sum()))
    lam, U = eigvals[:k], eigvecs[:, :k]
    total_abs = np.abs(eigvals).sum()

    return PCoAResult(
        coordinates=U * np.sqrt(lam),
        eigenvalues=eigvals,
        proportion_explained=lam / eigvals[positive].sum(),
        gower=delta,
        n_negative=int((eigvals < -1e-10 * scale).sum()),
        negative_fraction=float(np.abs(eigvals[eigvals < 0]).sum() / total_abs),
        metadata={"method": "aPCoA", "n_covariate_columns": int(X.shape[1])},
    )
