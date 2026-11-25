# Principal Coordinates Analysis — EC9600 Practical Assignment 1

**Group 10 · 
2022/E/048
2022/E/092
2022/E/141

A from-scratch implementation of Principal Coordinates Analysis (PCoA / classical metric
multidimensional scaling), applied to the 675-sample microbiome dataset from
Shi *et al.* (2020), *aPCoA: covariate adjusted principal coordinates analysis*,
Bioinformatics 36(13):4099–4101 ([doi:10.1093/bioinformatics/btaa276](https://doi.org/10.1093/bioinformatics/btaa276)).

---

## Quick start

```bash
# 1. dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. put the two CSVs in ./data/
#    data/MicrobiomeNoMetadata.csv
#    data/MicrobiomeWithMetadata.csv

# 3. run the analysis
jupyter lab PCoA_EC9600_Practical1.ipynb     # then Kernel > Restart & Run All

# 4. run the correctness tests
pytest -q tests/
```

Run-all takes about **2 minutes** on a single CPU core and regenerates every figure in
`results/figures/`, every table in `results/tables/`, and `results/results.json`, which holds
every number quoted in the report. All random operations are seeded (`RANDOM_STATE = 20250101`),
so re-running reproduces the report exactly.

Alternatively, run headless:

```bash
jupyter nbconvert --to notebook --execute --inplace PCoA_EC9600_Practical1.ipynb
```

---

## Contents

```
GX_EC9600_P1/
├── PCoA_EC9600_Practical1.ipynb    main deliverable — 11 sections, outputs included
├── pcoa/                           the implementation
│   ├── core.py                     Gower centring, eigendecomposition, corrections, aPCoA
│   ├── distances.py                six dissimilarity measures, memory-bounded
│   ├── evaluation.py               PERMANOVA, ANOSIM, Mantel, stress, Procrustes, silhouette
│   ├── data.py                     loading, auditing, preprocessing
│   └── viz.py                      scree, ordination, Shepard, biplot, heatmap
├── tests/test_pcoa.py              36 correctness tests
├── report/
│   ├── GX_EC9600_P1_Report.docx    4-page report
│   └── GX_EC9600_P1_Report.pdf
├── results/
│   ├── figures/                    12 figures (PNG, 200 dpi)
│   ├── tables/                     every table as CSV
│   └── results.json                every headline number
├── requirements.txt
└── README.md
```

---

## The algorithm

Given an `N x N` dissimilarity matrix `D` (Gower 1966; restated in Shi *et al.* 2020):

| Step | Operation |
|---|---|
| 1 | `A = -0.5 * D^2` |
| 2 | `G = (I - 11'/N) A (I - 11'/N)` — Gower's doubly-centred matrix |
| 3 | `G = U L U'` — symmetric eigendecomposition |
| 4 | `Y = U_k L_k^(1/2)` — project onto the `k` leading axes |

Minimal use:

```python
from pcoa import load_microbiome, pairwise_distances, pcoa, preprocess

data = load_microbiome("data/MicrobiomeNoMetadata.csv",
                       "data/MicrobiomeWithMetadata.csv")
X, taxa, _ = preprocess(data.abundances, data.taxa, min_prevalence=5, unzero=True)
D = pairwise_distances(X, metric="braycurtis")
result = pcoa(D, n_components=10)
print(result.summary())
```

---

## Headline results

| | |
|---|---|
| Dataset after filtering | 675 samples x 2,385 taxa (prevalence >= 5) |
| Variation on PCo1 + PCo2 | 28.0% (37.7% on three axes) |
| Negative eigenvalues | 366, carrying 11.0% of the absolute trace |
| Broken-stick rule retains | 20 axes (74.7% of positive variation) |
| Dominant factor | **Donor**, R² = 0.287, ANOSIM R = 0.494 |
| Secondary factor | **Diet**, R² = 0.140 (survives sensitivity analysis) |
| Stress-1 at 2 / 20 / 50 axes | 0.514 / 0.102 / 0.036 |
| aPCoA adjusted for Donor | Diet R² in leading plane 0.237 → 0.428 |

Three findings are worth flagging in particular:

1. **The table contains no exact zeros**, yet 97.6% of entries lie below 1e-9, drawn from
   uniform(0, 1e-10) — the authors zero-replaced the data. Left untreated, Jaccard returns a
   matrix of zeros. The pipeline builds two views of the table accordingly.
2. **`Sex` is a false positive.** PERMANOVA reports p = 0.001; rank-based ANOSIM reports
   R = -0.060, p = 0.916. It is a dispersion artefact of the 621-vs-54 split. Running only one
   test would have produced a wrong finding.
3. **Stress is non-monotonic in the number of axes** — minimised at ~50 axes, then rising.
   Impossible for Euclidean input; a direct consequence of the negative eigenvalues.

---

## Validation

Correctness is established against cases whose answers are known independently of this code:

* a unit square recovered from its distance matrix alone (Procrustes m² < 1e-16);
* the algebraic identity PCoA(Euclidean) == PCA, matching scikit-learn to 1e-12;
* eigenvalues and coordinates matching the **scikit-bio** reference PCoA to 1e-12;
* PERMANOVA R² from the trace formulation matching a hand-coded sum-of-squares partition;
* metric implementations matching `scipy.spatial.distance` to 1e-9.

`pytest -q tests/` — 36 passed.

---

## Notes

* **scikit-bio is optional.** It is used only as an independent cross-check; the notebook and
  tests skip those checks cleanly if it is absent.
* **Renaming for submission:** replace `GX` with your group number in the folder name, the
  report filename, and the two `[member names]` placeholders (README and report title block).
* The CSV data files are **not** included in this archive; place them in `data/` as above.

---

## References

1. Gower, J.C. (1966) *Biometrika* **53**, 325–338.
2. Shi, Y. *et al.* (2020) *Bioinformatics* **36**(13), 4099–4101. [dataset source]
3. Bray, J.R. & Curtis, J.T. (1957) *Ecol. Monogr.* **27**, 325–349.
4. Anderson, M.J. (2001) *Austral Ecology* **26**, 32–46.
5. Clarke, K.R. (1993) *Austral Ecology* **18**, 117–143.
6. Legendre, P. & Legendre, L. (2012) *Numerical Ecology*, 3rd ed., Ch. 9.
7. Cailliez, F. (1983) *Psychometrika* **48**, 305–308.
8. Lingoes, J.C. (1971) *Psychometrika* **36**, 195–203.
9. Aitchison, J. (1986) *The Statistical Analysis of Compositional Data*. Chapman & Hall.
