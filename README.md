<div align="center">

# CRISP

**C**ompositional **R**atio-based **I**mputation with **S**implex **P**rojection

*Constraint-guaranteed imputation for compositional data with structural zeros.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![tests](https://github.com/wenyu2026/crisp-imputer/actions/workflows/tests.yml/badge.svg)](https://github.com/wenyu2026/crisp-imputer/actions/workflows/tests.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%3E%3D1.20-013243.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/manuscript-under%20revision-orange.svg)](#-manuscript-status)

</div>

---

## TL;DR

Materials formulation tables — alloys, glasses, cements, electrolytes — are **compositional**: every row sums to a fixed total (usually 100), and a component that was deliberately *not added* appears as a hard `0`. These tables contain **two different kinds of missing information**:

- **structural zeros** — "this component was not added". Real information. Must be preserved.
- **missing values** (`NaN`) — "this measurement was lost". Must be imputed.

Standard tools handle neither combination well:

| Approach | Behaviour on structural zeros | Simplex constraint |
|---|---|---|
| Log-ratio family (`impKNNa`, `impCoda`, `lrEM`, `lrSVD`) | `log(0)` → needs a pseudo-count, turning *"absent"* into *"present in trace amount"* | not guaranteed |
| Naive / regression / low-rank (KNN, mean, MICE, SoftImpute) | treats `0` as an ordinary number | **violated** — outputs sum to ≠ 100 |
| **CRISP** | preserved exactly, by construction | **satisfied exactly, by construction** |

`crisp()` fills missing entries proportionally to a profile estimated from complete rows, leaves structural zeros untouched, and **provably cannot** produce an infeasible composition — no post-hoc projection needed.

---

## 📦 Installation

```bash
git clone https://github.com/wenyu2026/crisp-imputer.git
cd crisp-imputer
pip install -r requirements.txt
```

CRISP itself is a single module with no build step — just copy [`crisp.py`](crisp.py) into your project if you prefer:

```bash
pip install numpy pandas scikit-learn
cp crisp.py /your/project/
```

Or install as a package:

```bash
pip install .
```

---

## 🚀 Quick start

```python
import numpy as np
from crisp import crisp, lcrisp, auto_crisp, check_compositional_validity

# Three alloy rows; 0 = component not added, NaN = record lost
X = np.array([
    [70.0, 15.0, 15.0],    # Fe, C, Mn  — complete
    [np.nan, 20.0, 10.0],  # Fe missing
    [65.0, np.nan, 20.0],  # C missing
])

X_filled = crisp(X, comp_idx=[0, 1, 2], total=100.0)
print(X_filled)
# [[70.  15.  15.]
#  [70.  20.  10.]   <- Fe allocated from the profile
#  [65.  15.  20.]]  <- C allocated from the profile

print(check_compositional_validity(X_filled, comp_idx=[0, 1, 2], total=100.0)["sum_dev_mean"])
# 0.0   (machine precision)
```

### Variants

| Variant | Profile | Use when |
|---|---|---|
| `crisp` | global (median of normalized complete rows) | default; low missing rate |
| `lcrisp` | local k-NN profile | higher missing rate, heterogeneous data |
| `auto_crisp` | switches between the two at a threshold | you don't want to choose |

> ⚠️ Only the **base `crisp` variant is benchmarked** in the current results. `lcrisp` / `auto_crisp` are implemented but remain un-evaluated — see [Known issues](#-known-issues).

---

## 📖 API

```python
crisp(X, comp_idx, non_comp_idx=None, total=100.0)
```
Global-profile imputation.
- `X` — `(n, d)` array (or DataFrame); structural zeros as `0`, missing as `np.nan`
- `comp_idx` — column indices that form the composition (columns outside this set pass through untouched)
- `non_comp_idx` — non-compositional columns, used only for k-NN distance in `lcrisp`
- `total` — the closure constant (default `100.0`)

```python
lcrisp(X, comp_idx, non_comp_idx=None, n_neighbors=5, total=100.0)
```
Local k-NN profile — the profile is estimated from the nearest complete rows instead of globally.

```python
auto_crisp(X, comp_idx, non_comp_idx=None, total=100.0, threshold=0.15)
```
Picks `crisp` when the overall missing rate is below `threshold`, `lcrisp` above it.

The same three are available as scikit-learn-style estimators: `CRISPImputer`, `LCRISPImputer`, `AutoCRISPImputer` (each with `fit` / `transform` / `fit_transform`).

```python
project_to_simplex(x, total=100.0)
```
Post-hoc projection helper, provided for baselines and comparisons. **CRISP does not need it** — its output is feasible by construction.

```python
check_compositional_validity(X, comp_idx, total=100.0)
# -> {'sum_dev_mean': float, 'sum_dev_max': float, 'min_value': float, 'valid': bool, ...}
```
Audit helper: reports simplex violation and whether any component went negative.

---

## 🧮 How it works

### Notation

Let `S = { x ∈ R₊ᵈ : Σⱼ xⱼ = 100 }` be the composition simplex. For sample `i`, let `K` be the observed components, `M` the missing ones, and `R = 100 − Σ_{k∈K} x_k` the remainder. Let `C` be the complete rows, `M_count = |C|`.

### Algorithm

1. Estimate a **profile** `p̂` = column-wise median of the normalized complete rows.
2. Allocate the remainder among the missing components proportionally:

```
x̂_j = ( p̂_j / Σ_{m∈M} p̂_m ) · R ,    ∀ j ∈ M
```

3. Never touch structural zeros.

Because the allocation is proportional to `R` and `R` is exactly what's left to 100, every imputed row sums to `total` **by construction**. Complexity is `O(d·M_count)` for the profile and `O(d)` per imputed row — a single pass, no iteration, no hyper-parameters.

### Why the allocation is principled

The proportionally-allocated profile is the **plug-in empirical-Bayes estimate of the Dirichlet conditional mean** `E[x_j | x_K]`: under a Dirichlet prior, the conditional expectation of the missing part, given the observed part and the closure, is exactly a proportional split of the remainder by the prior's mean composition. CRISP replaces the unknown prior mean with a plug-in profile estimate.

The error decomposes into a **shrinkable** profile term `O(M_count^(−1/2))` and an **irreducible** conditional-uncertainty term.

> **Precision matters here.** The Dirichlet-conditional-mean reading holds exactly for the **mean** and **geometric-mean** profiles. The **median** profile (the default) converges to the population median, which equals `α/Σα` only under a symmetry condition `Dirichlet` marginals satisfy only in `d = 2`. So for the default median profile the plug-in reading is **approximate**, and there is a small non-vanishing bias floor. See [`THEORY.md`](THEORY.md) and [`PROOFS.md`](PROOFS.md) for the full statements, and [`HONEST_CRISP_2026-10-08.md`](HONEST_CRISP_2026-10-08.md) for the current status of each claim.

---

## 📊 Benchmark results

All numbers below are **regenerated from the released artifacts**, 20 missing-instance seeds per configuration, paired Wilcoxon signed-rank tests. The full table (100 configurations × 5 methods × 3 variants) is in [`HONEST_results.csv`](HONEST_results.csv); per-configuration detail is in [`honest_summary.txt`](honest_summary.txt).

Two comparison arms deserve naming up front, because they change the story:

- **`+closure`** — renormalize the baseline's output rows to 100 (post-hoc projection).
- **`+backfill`** — for rows with **exactly one** missing entry, solve it exactly from the closure (`x̂ = 100 − Σ known`), then renormalize.
  *This is the fair comparison.* Regression imputers are never told the row sum is 100; giving them that one line of arithmetic is the minimum courtesy.

### 1. The n<d blind spot in the standard tooling ⭐

`zCompositions::lrEMplus` is the official R function for **"zeros and missing data simultaneously"** — the exact problem CRISP targets. We ran it on every configuration.

**Result: it ran on 3 of 140 configurations.**

| Failure | Configurations |
|---|---|
| `lrEMplus works on regular data sets (no. rows > no. columns)` | **109** |
| `lrEM based on alr requires at least one complete column` | **27** |
| configuration had no missing values | 1 |

`lrEMplus` requires `n > d`, and it is built on an alr transform that needs at least one column free of both zeros and missing values. In sparse formulation data (54–86% structural zeros) **neither condition holds** — so the official tool for this exact problem is structurally unavailable in the regime that matters most.

Where it *did* run (all `glass`, low missing rate), CRISP was far more accurate:

| Configuration | `lrEMplus` | `lrEMplus` +backfill | CRISP | CRISP advantage |
|---|---|---|---|---|
| `glass` MCAR 10% | 1.196 | 0.400 | **0.073** | **5.4×** |
| `glass` MCAR 20% | 880.22 | 2.283 | **0.129** | **17.7×** |
| `glass` MNAR 10% | 0.236 | 0.082 | **0.017** | **4.8×** |

Reproduce with [`run_lremplus_baseline.R`](run_lremplus_baseline.R) → [`lremplus_baseline.csv`](lremplus_baseline.csv).

### 2. Synthetic n<d — where CRISP wins

`n < d` synthetic compositions, MCAR 10% missing, 20 seeds. "Best fair baseline" = the best of {MICE, KNN, mean, SoftImpute} across raw / +closure / **+backfill**.

| Configuration | CRISP | best raw baseline | **best +backfill** | CRISP advantage |
|---|---|---|---|---|
| `synA1` (25 × 50) | **3.459** | MICE 8.391 | mean 5.434 | **1.57×** |
| `synA2` (30 × 40) | **2.969** | MICE 6.670 | mean 3.899 | **1.31×** |
| `synA3` (20 × 40) | **3.048** | MICE 7.738 | mean 4.163 | **1.37×** |
| `synA4` (25 × 60) | **3.138** | MICE 6.439 | KNN 4.098 | **1.31×** |
| `ge_nd` (20 × 30, effective d = 9) | 4.274 | MICE 3.872 | KNN 1.991 | **CRISP loses 2.15×** |

**Read this honestly:** the headline advantage is **1.3–1.6×** against a fair baseline, not the 2–2.5× you get by comparing against un-projected regression output. And the one "real" high-dimensional table in this group is not actually high-dimensional — 21 of its 30 columns are identically zero (see [Limitations](#-limitations)).

### 3. Low-missingness data — CRISP is *not* the accuracy winner

MCAR 10%:

| Dataset | CRISP | best raw baseline | **best +backfill** | Verdict |
|---|---|---|---|---|
| `ge` (55% zeros) | 2.435 | MICE 1.646 | MICE 1.644 | CRISP loses |
| `glass` (21%) | 0.073 | MICE 0.075 | KNN 0.062 | CRISP loses |
| `synthetic` (42%) | 4.286 | MICE 3.736 | MICE 3.704 | CRISP loses |
| `steel` (17%) | 1.513 | MICE 1.548 | KNN 1.145 | CRISP loses |
| `cement` (20%) | 1.074 | MICE 0.566 | KNN 0.336 | CRISP loses |

On dense, low-missingness data CRISP is competitive but **not** the most accurate. Its differentiators there are the **strict constraint guarantee**, **structural-zero semantics** and **zero tuning** — not accuracy.

### 4. Real-material n<d subsets — the accuracy advantage disappears

| Subset | single-missing rows | CRISP | best raw baseline | **+backfill** |
|---|---|---|---|---|
| `glass` (n=6, d=8) | **100%** | 0.000 | MICE 0.136 | **all 0.000 — tie** |
| `cement` (n=6, d=7) | **100%** | 0.000 | MICE 1.807 | **all 0.000 — tie** |
| `ge` (n=8, d=9) | **100%** | 0.000 | MICE 10.272 | **all 0.000 — tie** |
| `steel` (n=10, d=13) | 75% | 3.695 | MICE 5.711 | **mean 2.769 — CRISP loses** |

When every missing row has a single missing entry, `x̂ = 100 − Σ known` is an arithmetic identity, so **any** method given the closure solves those rows exactly. On the one subset with a meaningful share of multi-missing rows, a closure-aware mean beats CRISP. We report this rather than hiding it.

### 5. Overall win rate

Across all 100 configurations, against the strongest fair (+backfill) baseline, **CRISP is the most accurate in 20**.

| Group | CRISP best in |
|---|---|
| synthetic/synthetic-style n<d, 10% missing | **14 / 30** |
| low-missingness, 20% missing | 1 / 15 |
| n<d, 30% missing | 5 / 15 |
| real-material n<d | 0 / 4 |

### 6. Constraint satisfaction

| | CRISP | MICE | KNN | SoftImpute |
|---|---|---|---|---|
| low-missingness | 0.0000 | 0.0004–0.0151 | 0.078–4.26 | 0.24–20.1 |
| synthetic n<d | **0.0014–0.0029** | 6.86–8.58 | 5.70–7.97 | 7.85–9.70 |
| real n<d | 0.0000 | 2.95 | 7.48 | 9.13 |

*(mean absolute row-sum deviation from 100)*

CRISP's violation is **three orders of magnitude smaller** than the baselines' on n<d data, and it needs no post-hoc projection. It is *not* exactly zero on synthetic data: rows with no missing values are passed through untouched, which is why you see 0.0014–0.0029 rather than machine precision.

---

## ⛔ Limitations

We would rather state these than have a reviewer find them.

1. **The accuracy advantage is narrow.** It holds for **synthetic `n < d` data at ~10% missingness** (~1.3–1.6×). It does **not** hold on low-missingness data (5/5 datasets lost), at 20–30% missingness, or on real-material `n < d` subsets.
2. **Real high-dimensional validation is missing.** The only "real high-dimensional" table in the benchmark (`ge_nd`, nominal 30 columns) has **21 identically-zero columns** — effective `d = 9` with `n = 20`, so it is *not* `n < d`. Genuine real high-dimensional `n < d` compositional data remains to be tested.
3. **The estimator is simple and not new in kind.** It is a plug-in form of a known conditional mean; the contribution is the structural-zero handling plus the constructive guarantee, not a new estimator class.
4. **The Dirichlet reading is exact only for the mean / geometric-mean profiles.** The default median profile has an approximate reading and a small non-vanishing bias.
5. **No prospective or experimental validation.** All experiments are retrospective.
6. **Nominal vs actual missingness differ.** Missingness is injected only at non-zero entries, and 54–86% of the matrices are structural zeros, so a nominal 10% becomes 0.6–8.5% of cells. Per-configuration actual counts are in [`HONEST_results.csv`](HONEST_results.csv).
7. **`SoftImpute` here is a local implementation** (hard-thresholded SVD, no centering), not a reference implementation of Mazumder et al. — treat that comparison with caution.
8. **`lcrisp` / `auto_crisp` are un-evaluated.**
9. **`crisp.py` imports `sklearn.neighbors.NearestNeighbors` without using it.** The core algorithm is pure NumPy, but the import makes scikit-learn a hard requirement.

---

## 🔁 Reproducing the results

Everything in this repository regenerates from the scripts with fixed seeds. Requires Python ≥ 3.9 and, for the R baselines, R with `robCompositions` and `zCompositions`.

```bash
# datasets are bundled in data/ — no download needed
python gen_missing.py          # build tmp/      low-missingness draws
python gen_missing_nd.py       # build tmp_nd/   n<d draws
python gen_real_nd.py          # build tmp_realnd/  real n<d subsets

# Python baselines + CRISP
python final_compare_v3.py     # -> final_results_v3.csv, wilcoxon_v3.csv
python final_nd_v3.py          # -> final_results_nd_v3.csv, wilcoxon_nd_v3.csv
python final_real_nd.py        # -> final_results_real_nd.csv

# R baselines (optional but recommended)
Rscript run_r_baseline.R impKNNa tmp
Rscript run_r_baseline.R missForest tmp
Rscript run_lremplus_baseline.R        # -> lremplus_baseline.csv

# honest re-run: adds +closure / +backfill fair-comparison arms
python honest_rerun.py         # -> HONEST_results.csv, HONEST_wilcoxon.csv
python summarize_honest.py     # -> honest_summary.txt
python make_figures.py         # -> figures/*.pdf
```

`X_true.csv` / `X_missing.csv` for every configuration are committed so the R baselines can be reproduced against the **exact same draws** the released tables used. The bulky per-configuration `impKNNa.csv` / `missForest.csv` outputs are gitignored — regenerate them with `run_r_baseline.R`.

---

## ✅ Testing

```bash
pip install pytest
pytest -q
```

**56 tests, ~1 s.** `tests/test_crisp.py` pins the invariants the method is advertised on:

| Group | What is checked |
|---|---|
| Core invariants | closure of every imputed row; non-negativity; structural zeros preserved exactly; observed entries untouched; single-missing exact recovery; hand-computed multi-missing allocation |
| API | determinism; function ↔ estimator class agreement; `fit`/`transform` ↔ `fit_transform`; fitted profile reused across calls; DataFrame input; non-compositional columns untouched |
| Variants | `lcrisp` invariants; `n_neighbors` larger than `n`; `auto_crisp` dispatch below/above threshold; closure under `auto_crisp` |
| Helpers | `check_compositional_validity` (pass and fail cases); `project_to_simplex` (1-D, 2-D with feature indices, zero row); `compositional_mae` |
| Edge cases | no-missing identity; single row; fully-missing row; observed part exceeding `total`; custom `total`; entirely-missing column; fallback profile with <3 usable rows |
| Regression guards | full-observation rows are **not** re-projected (documents the non-zero SumDev); nearly-complete rows do enter the profile; NaN in a non-compositional column becomes a column mean |

The regression guards exist so that behaviour the released benchmarks depend on — including behaviour the manuscript describes differently — cannot drift silently.

CI runs the suite on Python 3.9 / 3.11 / 3.12 via [`.github/workflows/tests.yml`](.github/workflows/tests.yml). The R baselines are not covered by CI (they need `robCompositions` / `zCompositions`) and are run manually.

---

## 🗂 Repository layout

```
crisp-imputer/
├── crisp.py                     # the method (single module)
├── example.py                   # minimal usage example
├── app_demo.py                  # interactive demo
│
├── benchmark_zeros.py           # structural-zero benchmark, make_missing, data loaders
├── final_compare_v3.py          # low-missingness comparison
├── final_nd_v3.py               # n<d comparison
├── final_real_nd.py             # real-material n<d subsets
├── honest_rerun.py              # +closure / +backfill fair-comparison re-run
├── summarize_honest.py          # honest tables
├── gen_missing*.py              # missingness generators
├── make_figures.py              # figures
├── run_r_baseline.R             # impKNNa / missForest baselines
├── run_lremplus_baseline.R      # lrEMplus baseline + fair arm
├── verify_crisp_theory.py       # theory sanity checks
│
├── tests/test_crisp.py          # 56 unit tests (pytest)
├── conftest.py                  # makes the repo root importable under pytest
├── .github/workflows/tests.yml  # CI: pytest on Python 3.9 / 3.11 / 3.12
│
├── data/                        # bundled datasets (steel, glass, ge) + provenance
├── tmp/  tmp_nd/  tmp_realnd/   # missingness draws (X_true / X_missing committed)
├── figures/                     # generated figures
│
├── *.csv                        # result tables (see below)
├── THEORY.md                    # theory summary
├── PROOFS.md                    # statements and derivations
├── CONTRIBUTION.md              # positioning vs existing methods
├── HONEST_CRISP_2026-10-08.md   # ★ regenerated results + claim-by-claim status
├── VERIFY_lrEMplus_2026-10-08.md# lrEMplus availability audit
├── paper_draft.md               # manuscript draft (under revision)
└── CRISP_PAPER_CN.pdf           # Chinese version of the draft
```

### Key result files

| File | Contents |
|---|---|
| `final_results_v3.csv` / `wilcoxon_v3.csv` | low-missingness: MAE, SumDev, paired tests |
| `final_results_nd_v3.csv` / `wilcoxon_nd_v3.csv` | n<d: MAE, SumDev, paired tests |
| `final_results_real_nd.csv` | real-material n<d subsets |
| `lremplus_baseline.csv` | `lrEMplus` availability + accuracy, 140 configurations |
| **`HONEST_results.csv`** | **100 configs × 5 methods × 3 variants (raw / +closure / +backfill)** |
| **`HONEST_wilcoxon.csv`** | **768 paired tests including the fair arms** |

---

## 📄 Manuscript status

The manuscript draft in this repository (`paper_draft.md`, `CRISP_PAPER_CN.pdf`) is **v5 and is under revision**. An independent audit of the released artifacts against the draft's claims found several statements that the data does not support — most importantly a claim that the standard compositional package cannot handle this data at all (it *can*, on 3 configurations; the accurate and stronger statement is the `n > d` requirement documented above), and a real-material accuracy claim that disappears under a fair closure-aware comparison.

[`HONEST_CRISP_2026-10-08.md`](HONEST_CRISP_2026-10-08.md) records, claim by claim, what holds, what does not, and what the corrected numbers are. **Prefer that document over the draft's §3 until the revision is published.**

---

## ⚠️ Known issues

**Resolved**

- [x] **Unit tests** — 56 tests covering the closure guarantee, non-negativity, structural-zero preservation, single-missing exact recovery, API behaviour, variants, helpers, edge cases and golden cases → [`tests/`](tests/)
- [x] **CI** — pytest on Python 3.9 / 3.11 / 3.12 → [`.github/workflows/tests.yml`](.github/workflows/tests.yml)
- [x] **Self-contained data** — the three external datasets are bundled in `data/`; `find_data()` resolves them locally

**Open**

- [ ] `lcrisp` / `auto_crisp` have invariant tests but are still **not benchmarked** against the baselines
- [ ] `lrEMplus` should be integrated as a first-class baseline in the main comparison tables (currently a separate script)
- [ ] `ge_nd` needs replacing with a genuinely high-dimensional real compositional dataset — 21 of its 30 columns are identically zero, so it is not actually `n < d`
- [ ] The `SoftImpute` baseline should use a reference implementation of Mazumder et al.
- [ ] R baselines are single-seed (`seed=42`) while Python methods use 20 seeds — extend them to multiple seeds so they can enter the paired tests
- [ ] `crisp.py` imports `pandas` and `sklearn.neighbors.NearestNeighbors` without using either; the core algorithm is pure NumPy, so both imports could be dropped (and scikit-learn would stop being a hard requirement)
- [ ] NaN in a **non-compositional** column is silently replaced by that column's mean — `_column_mean_init` fills *every* column. Compositional columns are then overwritten by the allocation, so the benchmarks are unaffected, but it is surprising library behaviour
- [ ] An entirely-NaN column emits NumPy `RuntimeWarning`s (`Mean of empty slice`, `All-NaN slice encountered`). The output is still correct, but the degenerate case should be handled explicitly

Contributions and corrections are welcome — please open an issue.

---

## 📚 Citation

```bibtex
@software{crisp_imputer,
  title  = {CRISP: Compositional Ratio-based Imputation with Simplex Projection},
  author = {wenyu2026},
  year   = {2026},
  url    = {https://github.com/wenyu2026/crisp-imputer}
}
```

If you use the bundled datasets, please also cite their originals — see [`data/README.md`](data/README.md).

---

## 📜 License

[MIT](LICENSE).

Developed as part of the [smallmatprep](https://github.com/wenyu2026/smallmatprep) toolkit for small-sample materials data preprocessing.
