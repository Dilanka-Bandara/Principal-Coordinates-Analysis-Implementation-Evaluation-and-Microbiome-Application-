"""
pcoa - a from-scratch implementation of Principal Coordinates Analysis.

EC9600 Practical Assignment 1.

Quick start
-----------
>>> from pcoa import load_microbiome, pairwise_distances, pcoa
>>> data = load_microbiome("data/MicrobiomeNoMetadata.csv",
...                        "data/MicrobiomeWithMetadata.csv")
>>> D = pairwise_distances(data.abundances, metric="braycurtis")
>>> res = pcoa(D, n_components=10)
>>> print(res.summary())
"""

from .core import PCoAResult, apcoa, cailliez_constant, gower_center, lingoes_constant, pcoa
from .data import MicrobiomeData, dummy_code, load_microbiome, preprocess, validate_table
from .distances import AVAILABLE_METRICS, pairwise_distances
from .evaluation import (
    anosim,
    cluster_agreement,
    kruskal_stress,
    mantel_test,
    permanova,
    procrustes_m2,
    reconstruction_error,
    shepard_correlation,
    silhouette_from_distances,
)

__version__ = "1.0.0"

__all__ = [
    "PCoAResult",
    "pcoa",
    "apcoa",
    "gower_center",
    "lingoes_constant",
    "cailliez_constant",
    "pairwise_distances",
    "AVAILABLE_METRICS",
    "MicrobiomeData",
    "load_microbiome",
    "validate_table",
    "preprocess",
    "dummy_code",
    "permanova",
    "anosim",
    "mantel_test",
    "kruskal_stress",
    "reconstruction_error",
    "shepard_correlation",
    "procrustes_m2",
    "silhouette_from_distances",
    "cluster_agreement",
    "__version__",
]
