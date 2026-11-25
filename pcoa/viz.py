"""
Plotting helpers for PCoA results.

Every function returns the Matplotlib ``Axes`` (or ``Figure``) so the caller can
keep customising, and every axis label carries the percentage of variation
explained - an ordination plot without that number is uninterpretable, because
the axes have no units and their relative scale is the whole message.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

__all__ = [
    "set_style",
    "scree_plot",
    "ordination_plot",
    "shepard_diagram",
    "biplot",
    "distance_heatmap",
    "confidence_ellipse",
]

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#4878CF", "#EE854A", "#6ACC64",
]


def set_style() -> None:
    """Consistent, print-friendly defaults for the whole report."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
        }
    )


def axis_label(result, axis: int) -> str:
    pct = 100 * result.proportion_explained[axis]
    return f"PCo{axis + 1} ({pct:.1f}%)"


def scree_plot(result, n_axes: int = 20, ax=None):
    """
    Eigenvalue scree with the cumulative curve, plus a broken-stick reference.

    The broken-stick model is the standard stopping rule in ordination: keep
    axis k only while its eigenvalue exceeds the value expected if the total
    variation were split at random, i.e. ``bs_k = (1/m) * sum_{i=k..m} 1/i``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 3.6))
    pos = result.eigenvalues[result.eigenvalues > 0]
    k = min(n_axes, pos.size)
    prop = pos[:k] / pos.sum()
    idx = np.arange(1, k + 1)

    m = pos.size
    bs = np.array([np.sum(1.0 / np.arange(i, m + 1)) / m for i in range(1, k + 1)])

    ax.bar(idx, 100 * prop, color=PALETTE[0], alpha=0.85, label="eigenvalue")
    ax.plot(idx, 100 * bs, "o--", color=PALETTE[3], ms=4, lw=1.2,
            label="broken-stick expectation")
    ax.set_xlabel("Principal coordinate axis")
    ax.set_ylabel("% of variation")
    ax.set_title("Scree plot")
    ax.set_xticks(idx[:: max(1, k // 10)])

    twin = ax.twinx()
    twin.plot(idx, 100 * np.cumsum(prop), "-", color=PALETTE[2], lw=1.6,
              label="cumulative %")
    twin.set_ylabel("cumulative %")
    twin.set_ylim(0, 105)
    twin.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    return ax


def confidence_ellipse(x, y, ax, n_std: float = 1.96, **kwargs):
    """Covariance ellipse enclosing ~95% of a bivariate normal fit to (x, y)."""
    if x.size < 3:
        return None
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    e = Ellipse((x.mean(), y.mean()), w, h, angle=angle, **kwargs)
    ax.add_patch(e)
    return e


def ordination_plot(
    result,
    labels=None,
    title: str = "PCoA ordination",
    axes: tuple[int, int] = (0, 1),
    ax=None,
    ellipses: bool = True,
    centroids: bool = True,
    legend_title: str | None = None,
    point_size: float = 22,
):
    """Scatter of two principal coordinates, coloured by a design factor."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.6))
    i, j = axes
    X, Y = result.coordinates[:, i], result.coordinates[:, j]

    if labels is None:
        ax.scatter(X, Y, s=point_size, c=PALETTE[0], alpha=0.75,
                   edgecolor="white", linewidth=0.4)
    else:
        labels = np.asarray(labels)
        for c, lv in enumerate(np.unique(labels)):
            m = labels == lv
            col = PALETTE[c % len(PALETTE)]
            ax.scatter(X[m], Y[m], s=point_size, color=col, alpha=0.78,
                       edgecolor="white", linewidth=0.4, label=str(lv))
            if ellipses:
                confidence_ellipse(X[m], Y[m], ax, facecolor=col, alpha=0.10,
                                   edgecolor=col, lw=1.0)
            if centroids:
                ax.scatter(X[m].mean(), Y[m].mean(), marker="X", s=95,
                           color=col, edgecolor="black", linewidth=0.7, zorder=5)
        ax.legend(title=legend_title, loc="best", ncol=1)

    ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.6, alpha=0.5)
    ax.set_xlabel(axis_label(result, i))
    ax.set_ylabel(axis_label(result, j))
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")  # equal aspect: distances are the point
    return ax


def shepard_diagram(D_original, D_embedded, ax=None, sample: int = 40000,
                    random_state: int = 0, title: str = "Shepard diagram"):
    """
    Original vs reconstructed dissimilarity for every sample pair.  Points on
    the 1:1 line mean the low-dimensional embedding is faithful.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.4))
    iu = np.triu_indices_from(D_original, k=1)
    d, dh = D_original[iu], D_embedded[iu]
    if d.size > sample:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(d.size, sample, replace=False)
        d, dh = d[sel], dh[sel]
    ax.scatter(d, dh, s=2, alpha=0.12, color=PALETTE[0], edgecolor="none", rasterized=True)
    lo, hi = 0, max(d.max(), dh.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], "--", color=PALETTE[3], lw=1.3, label="1:1")
    ax.set_xlabel("Original dissimilarity")
    ax.set_ylabel("Reconstructed distance")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


def biplot(
    result,
    X,
    taxa,
    labels=None,
    n_arrows: int = 8,
    axes: tuple[int, int] = (0, 1),
    ax=None,
    title: str = "PCoA biplot",
):
    """
    Overlay taxon vectors on the ordination.

    Because PCoA works from a distance matrix, taxa have no loadings of their
    own.  We use the standard *weighted-averages* projection (Legendre &
    Legendre, s.9.3.4): correlate each taxon's abundance profile with the
    sample scores, and draw the strongest ones as arrows.  An arrow points
    towards the region of the plot where that taxon is enriched.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))
    ordination_plot(result, labels=labels, title=title, axes=axes, ax=ax)

    i, j = axes
    Y = result.coordinates[:, [i, j]]
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    sx = Xc.std(axis=0)
    sy = Yc.std(axis=0)
    sx[sx == 0] = 1.0
    corr = (Xc.T @ Yc) / (X.shape[0] * sx[:, None] * sy[None, :])

    strength = np.linalg.norm(corr, axis=1)
    top = np.argsort(strength)[::-1][:n_arrows]
    scale = 0.85 * max(np.abs(Y).max(), 1e-12)
    for t in top:
        dx, dy = corr[t] * scale
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2, alpha=0.85))
        ax.text(dx * 1.06, dy * 1.06, taxa[t], fontsize=7.5, color="#222222",
                ha="center", va="center")
    return ax


def distance_heatmap(D, labels=None, ax=None, title: str = "Dissimilarity matrix"):
    """Distance matrix with samples ordered by group, to show block structure."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.4))
    order = np.argsort(np.asarray(labels), kind="stable") if labels is not None \
        else np.arange(D.shape[0])
    im = ax.imshow(D[np.ix_(order, order)], cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("samples (ordered by group)")
    ax.set_ylabel("samples")
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.82, label="dissimilarity")
    if labels is not None:
        lab = np.asarray(labels)[order]
        bounds = np.flatnonzero(lab[1:] != lab[:-1]) if lab.size > 1 else np.array([], int)
        for b in bounds:
            ax.axhline(b + 0.5, color="white", lw=0.7, alpha=0.8)
            ax.axvline(b + 0.5, color="white", lw=0.7, alpha=0.8)
    return ax
