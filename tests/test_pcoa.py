"""
Correctness tests for the PCoA implementation.

Run with:  pytest -q tests/

The suite is built around cases where the right answer is known independently
of our code:

* configurations with an analytic solution (a square, collinear points),
* the algebraic identity PCoA(Euclidean) == PCA,
* agreement with SciPy's distance routines and (if installed) scikit-bio's
  reference PCoA,
* invariants that must hold for any valid implementation (symmetry, isometry,
  rotation/translation invariance, eigenvalue-trace identities).
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist, squareform

from pcoa import (
    cluster_agreement,
    gower_center,
    kruskal_stress,
    pairwise_distances,
    pcoa,
    permanova,
    procrustes_m2,
    reconstruction_error,
)
from pcoa.core import apcoa, cailliez_constant, lingoes_constant

RNG = np.random.default_rng(20250101)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def square_points():
    """Unit square: PCoA must recover it up to rotation/reflection."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


@pytest.fixture
def gaussian_blobs():
    """Three well-separated clusters in 5-D with known labels."""
    centres = np.array([[0, 0, 0, 0, 0], [8, 0, 0, 0, 0], [0, 8, 0, 0, 0]], float)
    X = np.vstack([c + RNG.normal(scale=0.5, size=(25, 5)) for c in centres])
    y = np.repeat([0, 1, 2], 25)
    return X, y


@pytest.fixture
def abundance_table():
    """Sparse, closed compositional table resembling the assignment data."""
    X = RNG.random((30, 40)) ** 4
    X[RNG.random((30, 40)) < 0.6] = 0.0
    X = X + 1e-12
    return X / X.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# distance metrics: agreement with SciPy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", ["euclidean", "braycurtis"])
def test_metric_matches_scipy(abundance_table, metric):
    ours = pairwise_distances(abundance_table, metric=metric)
    theirs = squareform(pdist(abundance_table, metric=metric))
    assert np.allclose(ours, theirs, atol=1e-9), f"{metric} disagrees with SciPy"


def test_jensen_shannon_matches_scipy(abundance_table):
    """SciPy defaults to natural log; we use base 2 so the distance is in [0, 1]."""
    from scipy.spatial.distance import jensenshannon

    X = abundance_table[:8]
    ours = pairwise_distances(X, metric="jensenshannon")
    for i in range(len(X)):
        for j in range(len(X)):
            assert np.isclose(ours[i, j], jensenshannon(X[i], X[j], base=2), atol=1e-10)
    assert ours.max() <= 1.0 + 1e-12


def test_jaccard_matches_scipy_on_binary(abundance_table):
    ours = pairwise_distances(abundance_table, metric="jaccard", presence_threshold=1e-9)
    binary = (abundance_table > 1e-9).astype(float)
    theirs = squareform(pdist(binary, metric="jaccard"))
    assert np.allclose(ours, theirs, atol=1e-12)


def test_hellinger_equals_euclidean_on_sqrt(abundance_table):
    ours = pairwise_distances(abundance_table, metric="hellinger")
    theirs = cdist(np.sqrt(abundance_table), np.sqrt(abundance_table), "euclidean")
    assert np.allclose(ours, theirs, atol=1e-12)


def test_braycurtis_is_half_l1_for_closed_data(abundance_table):
    """Rows sum to 1 => denominator is always 2 => BC == L1/2."""
    bc = pairwise_distances(abundance_table, metric="braycurtis")
    l1 = squareform(pdist(abundance_table, metric="cityblock"))
    assert np.allclose(bc, 0.5 * l1, atol=1e-12)


def test_blocking_does_not_change_result(abundance_table):
    """The memory budget is a performance knob, never a numerical one."""
    big = pairwise_distances(abundance_table, metric="braycurtis", memory_budget_mb=512)
    small = pairwise_distances(abundance_table, metric="braycurtis", memory_budget_mb=1e-4)
    assert np.allclose(big, small, atol=1e-14)


def test_distance_matrix_properties(abundance_table):
    for metric in ["braycurtis", "euclidean", "hellinger", "jensenshannon"]:
        D = pairwise_distances(abundance_table, metric=metric)
        assert np.allclose(D, D.T), f"{metric} not symmetric"
        assert np.allclose(np.diag(D), 0.0), f"{metric} has non-zero diagonal"
        assert D.min() >= 0.0, f"{metric} produced a negative distance"


def test_triangle_inequality_for_metric_distances(abundance_table):
    """Euclidean, Hellinger and Jensen-Shannon are true metrics."""
    for metric in ["euclidean", "hellinger", "jensenshannon"]:
        D = pairwise_distances(abundance_table, metric=metric)
        assert (D[:, :, None] + D[None, :, :].transpose(1, 0, 2) + 1e-9
                >= D[:, None, :]).all(), f"{metric} violates the triangle inequality"


def test_metric_rejects_bad_input():
    with pytest.raises(ValueError):
        pairwise_distances(np.array([[1.0, np.nan]]), metric="euclidean")
    with pytest.raises(ValueError):
        pairwise_distances(np.array([[1.0, -2.0], [0.5, 0.5]]), metric="braycurtis")
    with pytest.raises(ValueError):
        pairwise_distances(np.ones((3, 3)), metric="not_a_metric")


# --------------------------------------------------------------------------- #
# Gower centring
# --------------------------------------------------------------------------- #
def test_gower_centering_matches_explicit_formula():
    A = RNG.normal(size=(12, 12))
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    C = np.eye(n) - np.ones((n, n)) / n
    assert np.allclose(gower_center(A), C @ A @ C, atol=1e-12)


def test_gower_rows_and_columns_sum_to_zero():
    A = RNG.normal(size=(9, 9))
    G = gower_center(0.5 * (A + A.T))
    assert np.allclose(G.sum(axis=0), 0.0, atol=1e-10)
    assert np.allclose(G.sum(axis=1), 0.0, atol=1e-10)


# --------------------------------------------------------------------------- #
# PCoA core: analytic and algebraic ground truth
# --------------------------------------------------------------------------- #
def test_recovers_unit_square(square_points):
    D = pairwise_distances(square_points, metric="euclidean")
    res = pcoa(D, n_components=2)
    assert res.n_axes == 2
    # exactly two non-zero eigenvalues, equal by symmetry of the square
    assert np.isclose(res.eigenvalues[0], res.eigenvalues[1], atol=1e-10)
    assert abs(res.eigenvalues[2]) < 1e-10
    assert procrustes_m2(res.coordinates, square_points) < 1e-12


def test_reconstructs_original_distances_exactly(gaussian_blobs):
    """For Euclidean input, all axes must reproduce D to machine precision."""
    X, _ = gaussian_blobs
    D = pairwise_distances(X, metric="euclidean")
    res = pcoa(D)
    err = reconstruction_error(D, res.reconstructed_distances())
    assert err["max_abs_error"] < 1e-8
    assert kruskal_stress(D, res.reconstructed_distances()) < 1e-9


def test_pcoa_on_euclidean_equals_pca(gaussian_blobs):
    """The identity that makes PCoA a generalisation of PCA."""
    from sklearn.decomposition import PCA

    X, _ = gaussian_blobs
    D = pairwise_distances(X, metric="euclidean")
    res = pcoa(D, n_components=4)
    scores = PCA(n_components=4).fit_transform(X)
    assert procrustes_m2(res.coordinates, scores) < 1e-12
    # eigenvalues of G equal (n-1) * PCA explained variances
    pca = PCA(n_components=4).fit(X)
    assert np.allclose(res.eigenvalues[:4],
                       pca.explained_variance_ * (X.shape[0] - 1), rtol=1e-8)


def test_no_negative_eigenvalues_for_euclidean_input(gaussian_blobs):
    X, _ = gaussian_blobs
    res = pcoa(pairwise_distances(X, metric="euclidean"))
    assert res.n_negative == 0
    assert res.negative_fraction < 1e-12


def test_negative_eigenvalues_appear_for_braycurtis(abundance_table):
    res = pcoa(pairwise_distances(abundance_table, metric="braycurtis"))
    assert res.n_negative > 0, "Bray-Curtis is not Euclidean-embeddable in general"
    assert 0 < res.negative_fraction < 1


def test_trace_identity(gaussian_blobs):
    """sum(eigenvalues) == trace(G) == total variation in the configuration."""
    X, _ = gaussian_blobs
    res = pcoa(pairwise_distances(X, metric="euclidean"))
    assert np.isclose(res.eigenvalues.sum(), np.trace(res.gower), rtol=1e-10)


def test_invariance_to_rotation_and_translation(gaussian_blobs):
    X, _ = gaussian_blobs
    Q, _ = np.linalg.qr(RNG.normal(size=(5, 5)))
    Xt = X @ Q + 17.0
    a = pcoa(pairwise_distances(X, metric="euclidean"), n_components=3)
    b = pcoa(pairwise_distances(Xt, metric="euclidean"), n_components=3)
    assert np.allclose(a.eigenvalues[:3], b.eigenvalues[:3], rtol=1e-8)
    assert procrustes_m2(a.coordinates, b.coordinates) < 1e-12


def test_proportion_explained_sums_to_one(abundance_table):
    res = pcoa(pairwise_distances(abundance_table, metric="hellinger"))
    assert np.isclose(res.proportion_explained.sum(), 1.0, atol=1e-10)
    assert np.all(np.diff(res.proportion_explained) <= 1e-12), "axes not in descending order"


def test_partial_solver_matches_full_solver(gaussian_blobs):
    X, _ = gaussian_blobs
    D = pairwise_distances(X, metric="euclidean")
    full = pcoa(D, n_components=3, eigen_solver="full")
    part = pcoa(D, n_components=3, eigen_solver="partial")
    assert np.allclose(full.eigenvalues[:3], part.eigenvalues[:3], rtol=1e-6)
    assert procrustes_m2(full.coordinates, part.coordinates) < 1e-12


def test_deterministic_across_runs(abundance_table):
    D = pairwise_distances(abundance_table, metric="braycurtis")
    a, b = pcoa(D, n_components=5), pcoa(D, n_components=5)
    assert np.array_equal(a.coordinates, b.coordinates)


def test_pcoa_rejects_invalid_matrices():
    with pytest.raises(ValueError):
        pcoa(np.array([[0.0, 1.0], [2.0, 0.0]]))          # asymmetric
    with pytest.raises(ValueError):
        pcoa(np.array([[1.0, 1.0], [1.0, 1.0]]))          # non-zero diagonal
    with pytest.raises(ValueError):
        pcoa(np.ones((3, 4)))                             # not square


# --------------------------------------------------------------------------- #
# corrections for non-Euclidean dissimilarities
# --------------------------------------------------------------------------- #
def test_lingoes_removes_negative_eigenvalues(abundance_table):
    D = pairwise_distances(abundance_table, metric="braycurtis")
    assert pcoa(D).n_negative > 0
    assert pcoa(D, correction="lingoes").negative_fraction < 1e-9


def test_cailliez_removes_negative_eigenvalues(abundance_table):
    D = pairwise_distances(abundance_table, metric="braycurtis")
    assert pcoa(D, correction="cailliez").negative_fraction < 1e-9


def test_corrections_are_zero_for_euclidean_input(gaussian_blobs):
    X, _ = gaussian_blobs
    D = pairwise_distances(X, metric="euclidean")
    assert lingoes_constant(D) < 1e-8
    assert cailliez_constant(D) < 1e-6


def test_correction_preserves_rank_order_of_samples(abundance_table):
    """An additive constant must not reshuffle which samples are most similar."""
    from scipy.stats import spearmanr

    D = pairwise_distances(abundance_table, metric="braycurtis")
    a = pcoa(D, n_components=2).coordinates
    b = pcoa(D, n_components=2, correction="lingoes").coordinates
    iu = np.triu_indices(len(a), 1)
    da = np.linalg.norm(a[iu[0]] - a[iu[1]], axis=1)
    db = np.linalg.norm(b[iu[0]] - b[iu[1]], axis=1)
    assert spearmanr(da, db).statistic > 0.95


# --------------------------------------------------------------------------- #
# covariate-adjusted PCoA
# --------------------------------------------------------------------------- #
def test_apcoa_removes_the_confounder_signal():
    """
    Build data where a nuisance factor dominates a weaker effect of interest;
    aPCoA must suppress the nuisance and expose the effect.
    """
    n_per = 20
    batch = np.repeat([0, 1], 2 * n_per)                # strong confounder
    effect = np.tile(np.repeat([0, 1], n_per), 2)       # weak effect of interest
    X = RNG.normal(scale=0.4, size=(4 * n_per, 6))
    X[:, 0] += 10.0 * batch
    X[:, 1] += 1.0 * effect
    D = pairwise_distances(X, metric="euclidean")

    plain = pcoa(D, n_components=2)
    adjusted = apcoa(D, batch[:, None], n_components=2)

    def separation(coords, labels):
        a, b = coords[labels == 0], coords[labels == 1]
        return np.linalg.norm(a.mean(0) - b.mean(0))

    # before adjustment the batch dominates axis 1
    assert separation(plain.coordinates, batch) > separation(plain.coordinates, effect)
    # after adjustment the batch separation is essentially gone
    assert separation(adjusted.coordinates, batch) < 1e-6
    assert separation(adjusted.coordinates, effect) > 0.5


# --------------------------------------------------------------------------- #
# evaluation utilities
# --------------------------------------------------------------------------- #
def test_permanova_detects_real_groups(gaussian_blobs):
    X, y = gaussian_blobs
    D = pairwise_distances(X, metric="euclidean")
    res = permanova(D, y, n_permutations=199, random_state=0)
    assert res.p_value <= 0.01
    assert res.effect_size > 0.8


def test_permanova_null_is_not_significant(gaussian_blobs):
    X, _ = gaussian_blobs
    D = pairwise_distances(X, metric="euclidean")
    random_labels = RNG.permutation(np.repeat([0, 1, 2], 25))
    res = permanova(D, random_labels, n_permutations=199, random_state=1)
    assert res.p_value > 0.05
    assert res.effect_size < 0.1


def test_permanova_is_reproducible(gaussian_blobs):
    X, y = gaussian_blobs
    D = pairwise_distances(X, metric="euclidean")
    a = permanova(D, y, n_permutations=99, random_state=7)
    b = permanova(D, y, n_permutations=99, random_state=7)
    assert a.p_value == b.p_value and a.statistic == b.statistic


def test_permanova_r2_matches_manual_sum_of_squares(gaussian_blobs):
    """Cross-check the trace formulation against the textbook SS partition."""
    X, y = gaussian_blobs
    D = pairwise_distances(X, metric="euclidean")
    n = len(y)
    ss_t = (D ** 2)[np.triu_indices(n, 1)].sum() / n
    ss_w = 0.0
    for lv in np.unique(y):
        idx = np.where(y == lv)[0]
        sub = D[np.ix_(idx, idx)]
        ss_w += (sub ** 2)[np.triu_indices(len(idx), 1)].sum() / len(idx)
    manual_r2 = (ss_t - ss_w) / ss_t
    res = permanova(D, y, n_permutations=9, random_state=0)
    assert np.isclose(res.effect_size, manual_r2, rtol=1e-8)


def test_clusters_are_recovered_from_coordinates(gaussian_blobs):
    X, y = gaussian_blobs
    res = pcoa(pairwise_distances(X, metric="euclidean"), n_components=3)
    assert cluster_agreement(res.coordinates, y)["adjusted_rand_index"] > 0.95


def test_stress_decreases_with_more_axes(abundance_table):
    D = pairwise_distances(abundance_table, metric="braycurtis")
    res = pcoa(D)
    stresses = [kruskal_stress(D, res.reconstructed_distances(k)) for k in (2, 5, 10)]
    assert stresses[0] > stresses[1] > stresses[2]


def test_procrustes_is_zero_for_a_scaled_rotation(square_points):
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert procrustes_m2(square_points, 3.2 * square_points @ R + 5.0) < 1e-12


# --------------------------------------------------------------------------- #
# optional: agreement with the scikit-bio reference implementation
# --------------------------------------------------------------------------- #
def test_matches_scikit_bio_reference(abundance_table):
    skbio = pytest.importorskip("skbio", reason="scikit-bio not installed")
    D = pairwise_distances(abundance_table, metric="braycurtis")
    ref = skbio.stats.ordination.pcoa(skbio.DistanceMatrix(D), number_of_dimensions=3)
    ours = pcoa(D, n_components=3)
    assert np.allclose(ours.eigenvalues[:3], ref.eigvals.values[:3], rtol=1e-8)
    assert procrustes_m2(ours.coordinates, ref.samples.values[:, :3]) < 1e-12
