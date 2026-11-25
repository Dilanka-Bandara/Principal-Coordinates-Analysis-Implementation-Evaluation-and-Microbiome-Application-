"""
Loading, validation and preprocessing of the microbiome OTU tables.

The two CSVs supplied with the assignment hold the same 675 x 6696 table of
**relative abundances** (every row sums to 1); ``MicrobiomeWithMetadata.csv``
prefixes it with five integer-coded design factors and stores the abundances
rounded to three significant figures, while ``MicrobiomeNoMetadata.csv`` keeps
full double precision.  We therefore take the abundances from the
no-metadata file and the labels from the with-metadata file.

One property drives most of the preprocessing decisions: the table contains no
exact zeros.  Roughly 97.6% of entries are drawn from a uniform distribution on
(0, 1e-10] - i.e. the authors applied **zero replacement** so that log-ratio
methods would be defined.  Those entries are structural zeros for
count-based metrics and must be treated as such, otherwise a
presence/absence metric like Jaccard sees every taxon in every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["MicrobiomeData", "load_microbiome", "validate_table", "preprocess"]

METADATA_COLUMNS = ["Diet", "Source", "Donor", "CollectionMet", "Sex"]

#: entries at or below this abundance are treated as replaced (structural) zeros
ZERO_REPLACEMENT_CEILING = 1e-9


@dataclass
class MicrobiomeData:
    """Abundance table plus aligned sample metadata."""

    abundances: np.ndarray          # (n_samples, n_taxa) relative abundances
    taxa: list[str]
    metadata: pd.DataFrame          # (n_samples, n_factors)

    @property
    def n_samples(self) -> int:
        return self.abundances.shape[0]

    @property
    def n_taxa(self) -> int:
        return self.abundances.shape[1]

    def group(self, factor: str) -> np.ndarray:
        return self.metadata[factor].to_numpy()


def load_microbiome(
    abundance_csv: str | Path,
    metadata_csv: str | Path | None = None,
    metadata_columns: list[str] | None = None,
) -> MicrobiomeData:
    """
    Read the OTU table and (optionally) the metadata-bearing twin file.

    Raises if the two files disagree on the number of samples or if their
    shared numeric block does not match to the precision of the rounded copy -
    that check is what licenses us to pair full-precision abundances with
    labels taken from the other file.
    """
    abundance_csv = Path(abundance_csv)
    table = pd.read_csv(abundance_csv)
    meta_cols = metadata_columns or METADATA_COLUMNS

    present = [c for c in meta_cols if c in table.columns]
    if present:                                   # single self-contained file
        metadata = table[present].copy()
        abundance = table.drop(columns=present)
    else:
        metadata = None
        abundance = table

    if metadata_csv is not None:
        other = pd.read_csv(metadata_csv)
        missing = [c for c in meta_cols if c not in other.columns]
        if missing:
            raise ValueError(f"metadata file is missing columns: {missing}")
        if len(other) != len(abundance):
            raise ValueError(
                f"sample-count mismatch: {len(abundance)} abundance rows vs "
                f"{len(other)} metadata rows"
            )
        metadata = other[meta_cols].copy()

        # consistency check between the two copies of the numeric block
        shared = [c for c in abundance.columns if c in other.columns]
        if shared:
            lhs = abundance[shared].to_numpy()
            rhs = other[shared].to_numpy()
            if not np.allclose(lhs, rhs, rtol=5e-3, atol=1e-12):
                raise ValueError(
                    "the abundance blocks of the two CSV files do not agree; "
                    "the row order may differ between files"
                )

    if metadata is None:
        metadata = pd.DataFrame(index=range(len(abundance)))

    metadata = metadata.reset_index(drop=True)
    return MicrobiomeData(
        abundances=abundance.to_numpy(dtype=np.float64),
        taxa=list(abundance.columns),
        metadata=metadata,
    )


def validate_table(X: np.ndarray, closure_tol: float = 1e-4) -> dict:
    """
    Structural audit of the abundance matrix.  Returns a dictionary of
    diagnostics rather than raising, so the notebook can display it as a table
    and the reader can see *why* each preprocessing choice was made.
    """
    row_sums = X.sum(axis=1)
    tiny = X <= ZERO_REPLACEMENT_CEILING
    prevalence = (~tiny).sum(axis=0)
    return {
        "n_samples": int(X.shape[0]),
        "n_taxa": int(X.shape[1]),
        "n_missing": int(np.isnan(X).sum()),
        "n_negative": int((X < 0).sum()),
        "n_exact_zeros": int((X == 0).sum()),
        "min_value": float(X.min()),
        "max_value": float(X.max()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "is_closed_composition": bool(np.allclose(row_sums, 1.0, atol=closure_tol)),
        "fraction_at_zero_floor": float(tiny.mean()),
        "n_taxa_never_observed": int((prevalence == 0).sum()),
        "median_taxon_prevalence": float(np.median(prevalence)),
        "max_taxon_prevalence": int(prevalence.max()),
    }


def preprocess(
    X: np.ndarray,
    taxa: list[str] | None = None,
    min_prevalence: int = 0,
    min_abundance: float = 0.0,
    unzero: bool = False,
    renormalise: bool = True,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Filter and (optionally) re-close the abundance table.

    Parameters
    ----------
    min_prevalence : int
        Keep a taxon only if it is genuinely observed (above the
        zero-replacement floor) in at least this many samples.  Rare taxa
        contribute noise to Bray-Curtis and dominate log-ratio metrics.
    min_abundance : float
        Keep a taxon only if its maximum relative abundance reaches this value.
    unzero : bool
        Overwrite the zero-replacement noise with exact zeros.  Do this for
        presence/absence and abundance-difference metrics; do **not** do it
        before a log-ratio transform, which needs strictly positive values.
    renormalise : bool
        Re-close the retained rows to sum to 1 after filtering, so that
        Bray-Curtis stays comparable across samples.

    Returns
    -------
    (X_processed, taxa_kept, report)
    """
    X = np.asarray(X, dtype=np.float64)
    taxa = list(taxa) if taxa is not None else [f"OTU{i}" for i in range(X.shape[1])]
    observed = X > ZERO_REPLACEMENT_CEILING

    keep = np.ones(X.shape[1], dtype=bool)
    if min_prevalence > 0:
        keep &= observed.sum(axis=0) >= min_prevalence
    if min_abundance > 0:
        keep &= X.max(axis=0) >= min_abundance
    if keep.sum() == 0:
        raise ValueError("filters removed every taxon; relax min_prevalence/min_abundance")

    Xf = X[:, keep]
    taxa_kept = [t for t, k in zip(taxa, keep) if k]

    if unzero:
        Xf = np.where(Xf > ZERO_REPLACEMENT_CEILING, Xf, 0.0)

    retained_mass = Xf.sum(axis=1)
    if renormalise:
        safe = np.where(retained_mass > 0, retained_mass, 1.0)
        Xf = Xf / safe[:, None]

    report = {
        "n_taxa_in": int(X.shape[1]),
        "n_taxa_kept": int(keep.sum()),
        "fraction_taxa_kept": float(keep.mean()),
        "min_retained_mass": float(retained_mass.min()),
        "mean_retained_mass": float(retained_mass.mean()),
        "unzeroed": bool(unzero),
        "renormalised": bool(renormalise),
    }
    return Xf, taxa_kept, report


def dummy_code(metadata: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """
    One-hot encode categorical design factors, dropping one level each, for use
    as the covariate matrix in ``pcoa.core.apcoa``.
    """
    return pd.get_dummies(
        metadata[columns].astype("category"), drop_first=True
    ).to_numpy(dtype=np.float64)
