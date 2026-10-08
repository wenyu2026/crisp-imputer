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

`lcrisp()` is the same construction with a **local** (k-NN) profile per sample. On this benchmark the local variant is the stronger of the two on real data (steel 2.4×, cement 1.9×), so the honest unit of comparison below is the **CRISP family**, not `crisp()` alone.

> **The honest bottom line**, stated up front so nobody has to dig for it: against a *closure-aware* baseline the family wins **34 of 80** configurations, mostly on synthetic `n < d` at ~10 % missingness (1.3–1.6×) and on `steel` (1.4–2.3×). It does **not** win on `ge`, on `cement`, at 20–30 % missingness, or on the original real `n < d` subsets. The "constraint satisfied exactly" claim holds **for imputed rows**; complete rows are passed through untouched, so the reported row-sum deviation is 0.0014–0.0029 on synthetic data rather than machine precision. Details and every correction are in [Benchmark results](#-benchmark-results) and [`HONEST_CRISP_2026-10-08.md`](HONEST_CRISP_2026-10-08.md).

---

## 🆕 What's new in 1.1.0

**Library**

| Change | Why |
|---|---|
| `crisp.py` is now **pure NumPy** — the unused `pandas` and `sklearn.neighbors` imports are gone | `import crisp` no longer drags in scikit-learn; the wheel depends on NumPy only |
| NaN in a **non-compositional** column is **left untouched** (was: silently replaced by a column mean) | CRISP imputes the composition. Filling columns outside `comp_idx` was undocumented and surprising |
| Degenerate all-NaN slices are handled explicitly — no more NumPy `Mean of empty slice` / `All-NaN slice` warnings | Cleaner output; the degenerate case is now intentional rather than incidental |

**Benchmark harness**

| Change | Why |
|---|---|
| `soft_impute` replaced with a **reference implementation** (soft-thresholded singular values + observed column centering) | The 1.0 version was hard rank truncation, which is a different algorithm. It **overstated SoftImpute's error by 1.4–3.6×**; see `softimpute_sensitivity.csv` |
| `lcrisp` / `auto_crisp` are now benchmarked | Previously implemented but never evaluated |
| New **real high-ratio `n < d`** subsets (`tmp_realhr/`, d/n 1.50–1.75, 20 %/40 % missing) | The old real subsets had d/n 1.13–1.33 with 75–100 % single-missing rows; `ge_nd` turned out not to be `n < d` at all |
| R baselines run **multi-seed** (`run_r_baseline_multiseed.R` + `stage_r_multiseed.py`) | They were single-seed and therefore excluded from the paired tests |
| `residual_backfill` now restores observed cells before closing | It was rescaling observed entries for baselines that do not preserve them, breaking the exact single-missing recovery |

**Tests**: 89 unit tests (`tests/test_crisp.py`, `tests/test_baselines.py`), CI on Python 3.9 / 3.11 / 3.12.

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

### 2. Synthetic n<d — where the CRISP family wins

`n < d` synthetic compositions, MCAR 10 % missing, 20 seeds. "Strongest fair baseline" = the best of {MICE, KNN, mean, SoftImpute} across raw / +closure / **+backfill** (the fair arm solves single-missing rows exactly from the closure).

| Configuration | CRISP | LCRISP | family best | strongest raw baseline | **strongest +backfill** | family advantage |
|---|---|---|---|---|---|---|
| `synA1` (25 × 50) | **3.459** | 5.351 | **3.459** | MICE 8.391 | mean 5.434 | **1.57×** |
| `synA2` (30 × 40) | **2.969** | 4.149 | **2.969** | MICE 6.670 | mean 3.899 | **1.31×** |
| `synA3` (20 × 40) | **3.048** | 4.434 | **3.048** | MICE 7.738 | mean 4.163 | **1.37×** |
| `synA4` (25 × 60) | **3.138** | 4.583 | **3.138** | MICE 6.439 | KNN 4.098 | **1.31×** |

Here the local profile is *worse* than the global one: 60–70 % of rows are usable but each carries several missing entries, so k-NN neighbours are too noisy.

**Read this honestly:** the advantage against a closure-aware baseline is **1.3–1.6×**, not the 2–2.5× you get by comparing against un-projected regression output. And these are synthetic pools.

### 3. Low-missingness data — `CRISP` loses, `LCRISP` wins

MCAR 10 %, 20 seeds:

| Dataset | CRISP | **LCRISP** | strongest raw baseline | **strongest +backfill** | family verdict |
|---|---|---|---|---|---|
| `ge` (86 × 9) | 2.435 | 2.062 | MICE 1.646 | MICE 1.644 | loses 1.25× |
| `glass` (40 × 8) | 0.073 | **0.061** | MICE 0.075 | KNN 0.062 | tie |
| `synthetic` (300 × 8) | **4.286** | 5.343 | MICE 3.736 | MICE 3.704 | loses 1.16× |
| `steel` (300 × 13) | 1.513 | **0.636** | MICE 1.548 | KNN 1.145 | **wins 1.80×** |
| `cement` (1030 × 7) | 1.074 | **0.551** | MICE 0.566 | KNN 0.336 | loses 1.64× |

Two things to take away:

- **`LCRISP` is consistently better than `CRISP` on the larger real datasets** (steel 2.4×, cement 1.9×) even though the missing rate is low. The global profile is the weaker variant there.
- **`auto_crisp` never selects it.** The router switches on the missing *rate* (`threshold=0.15`), and at an ~8 % rate it always picks `crisp` — exactly the wrong arm on steel and cement. Every `AutoCRISP` row in the result tables is bit-identical to the `CRISP` row, so "adaptive routing" is a no-op on this benchmark.

### 4. Real high-ratio n<d subsets — new in 1.1.0

The old "real n<d" subsets had `d/n = 1.13–1.33`, and **75–100 % of their missing entries came from single-missing rows** — where `x̂ = 100 − Σ known` is an arithmetic identity that any closure-aware method reproduces. They carried almost no information.

`gen_real_hr.py` builds better ones from the real data: `d/n = 1.50–1.75`, missingness 20 %/40 % (still only at non-zero entries), so **multi-missing rows dominate** (only 25–34 % of missing entries come from single-missing rows on glass/ge/cement, 6 % on steel).

| Subset group | n × d | d/n | family best | strongest +backfill | family verdict |
|---|---|---|---|---|---|
| `steel_MCAR_20` | 8 × 13 | 1.62 | 2.898 | MICE 4.180 | **wins 1.44×** |
| `steel_MNAR_20` | 8 × 13 | 1.62 | 1.806 | MICE 2.615 | **wins 1.45×** |
| `steel_MNAR_40` | 8 × 13 | 1.62 | 2.490 | MICE 4.444 | **wins 1.79×** |
| `glass_MNAR_40` | 5 × 8 | 1.60 | 0.061 | SoftImpute 0.089 | **wins 1.46×** |
| `cement_*` (4 groups) | 4 × 7 | 1.75 | 0.49–2.76 | KNN / MICE / SoftImpute | loses 1.05–1.60× |
| `ge_*` (4 groups) | 6 × 9 | 1.50 | 2.94–13.43 | KNN / mean | loses 1.2–1.9× |

The real-data advantage is **dataset-specific**: clear on `steel` (all mechanisms) and on `glass` at high missingness, absent on `ge` and `cement`.

### 5. `ge_nd` never belonged in the n<d claims

`ge_nd` was described as the one *real* high-dimensional `n < d` dataset. It is not: it is the first 20 rows of `ge` padded with **21 identically-zero columns**, so `d_eff = 9` with `n = 20` → an effective `d/n` of **0.45**, i.e. `n > d`. It is retained in the result tables for continuity, labelled as such, and excluded from every `n < d` claim.

### 6. Overall win rate

Across all 80 configurations, against the strongest closure-aware baseline:

| Missing-rate group | CRISP alone wins | **family (best of CRISP/LCRISP/AutoCRISP)** | family median advantage |
|---|---|---|---|
| 10 % | 12 / 30 | **17 / 30** | **1.285×** |
| 20 % | 4 / 23 | 8 / 23 | 0.850× |
| 30 % | 5 / 15 | 5 / 15 | 0.940× |
| 40 % | 4 / 8 | 4 / 8 | 1.040× |
| real n<d (single draw) | 0 / 4 | 0 / 4 | 0.860× |
| **total** | **25 / 80** | **34 / 80** | — |

The family is the honest unit of comparison — `LCRISP` is part of the method — and it wins **34 of 80**, with the advantage concentrated at 10 % missingness and on `steel`.

### 7. The official R baselines, on matched seeds

`run_r_baseline.R` runs each configuration **once** (`seed=42`), so the R baselines could not enter a paired test. `stage_r_multiseed.py` + `run_r_baseline_multiseed.R` now cover **1199 draws over 60 configurations**, and `compare_with_r.py` pairs them seed-by-seed against the Python implementations (896 paired combinations, paired Wilcoxon):

| R baseline | pairs | Python more accurate | Python sig. better | R sig. better | median ratio (R / Python) |
|---|---|---|---|---|---|
| `impKNNa` (robCompositions) | 420 | 231 (55 %) | 201 | 85 | 1.18× |
| `missForest` | 420 | 251 (60 %) | 221 | 132 | **1.03×** |
| `lrEMplus` (zCompositions) | 56 | **56 (100 %)** | 49 | **0** | **72×** |

`missForest` is a genuinely competitive baseline — within ~3 % of the Python implementations on the median configuration. When `lrEMplus` runs at all it is far worse: on `glass` MNAR 20 % its MAE is 22 246 against CRISP's 0.049.

### 8. Constraint satisfaction

| | CRISP / LCRISP | MICE | KNN | SoftImpute |
|---|---|---|---|---|
| low-missingness | **0.0000** | 0.0004–0.0151 | 0.078–4.26 | 0.05–3.93 |
| synthetic n<d | **0.0014–0.0029** | 6.86–8.58 | 5.70–7.97 | 6.79–8.60 |
| real high-ratio n<d | **0.0000** | 0.07–2.95 | 0.68–7.48 | 0.65–4.51 |

*(mean absolute row-sum deviation from 100)*

The violation is **three orders of magnitude smaller** on `n < d` data, with no post-hoc projection. It is not exactly zero on synthetic data: rows with no missing values are passed through untouched.

### 9. Downstream usability — corrected in 1.1.0

The manuscript claims that only CRISP-imputed matrices preserve a positive downstream predictive R² (+0.11) while MICE/KNN/mean collapse to negative values (−0.13 … −0.31). **That claim is not reproduced.** No script generating it existed; the figure plotted four hard-coded literals.

`downstream_eval.py` runs the experiment properly (20 % missing, 5 seeds, gradient-boosted surrogate, out-of-fold R² on the full datasets):

| Dataset (reference R², no missingness) | CRISP | MICE | KNN | mean | SoftImpute |
|---|---|---|---|---|---|
| `steel` (0.815) | +0.699 | +0.727 | **+0.757** | +0.701 | +0.560 |
| `cement` (0.467) | +0.439 | **+0.445** | +0.437 | +0.419 | +0.430 |
| `ge` (0.399) | +0.372 | +0.369 | +0.352 | +0.374 | **+0.440** |
| `glass` (0.675) | **+0.439** | +0.274 | +0.187 | −0.015 | −0.239 |

**Every method keeps a positive R² on the larger datasets.** CRISP is the best imputer downstream only on `glass` (n = 40). The "only CRISP survives" claim must be deleted; the defensible statement is the weaker "all imputers degrade downstream R² relative to complete data, and the ranking is dataset-dependent".

### 10. SoftImpute — a corrected baseline

The 1.0 `soft_impute` was **hard rank truncation**, not SoftImpute (no soft-thresholding, no centering). Against the reference implementation the old code overstated SoftImpute's error by **1.4–3.6×**:

| Configuration | 1.0 (hard rank truncation) | reference (best λ) | overstatement |
|---|---|---|---|
| `ge` low-miss MCAR 10 % | 29.34 | 8.24 | 3.56× |
| `cement` low-miss MCAR 10 % | 3.45 | 1.15 | 3.01× |
| `ge_nd` MCAR 10 % | 24.78 | 8.44 | 2.94× |
| `steel` low-miss MCAR 10 % | 5.88 | 2.72 | 2.16× |
| `synA1` n<d MCAR 10 % | 13.12 | 8.07 | 1.63× |
| `steel` realhr MCAR 20 % | 5.50 | 3.92 | 1.40× |

Crucially the conclusion survives: no λ makes SoftImpute competitive on synthetic `n < d` — the whole sweep spans 8.07–9.12 against CRISP's 3.46. See `softimpute_sensitivity.csv`.

---

## ⛔ Limitations

We would rather state these than have a reviewer find them.

1. **The accuracy advantage is narrow.** The CRISP family wins 34 of 80 configurations. It is concentrated on synthetic `n < d` at ~10 % missingness (1.3–1.6×) and on `steel` (1.4–2.3×). It does **not** hold on `ge` or `cement`, at 20–30 % missingness, or on the old real `n < d` subsets.
2. **Real high-dimensional validation is still missing.** The best real high-ratio subsets reach `d/n = 1.75`; the synthetic pools reach 2.4. A genuinely high-dimensional *real* compositional dataset with `n < d` could not be obtained, so the strongest `n < d` claim still rests on synthetic data.
3. **The router is wrong.** `auto_crisp` switches on the missing rate, so on every low-missingness dataset in the benchmark it reproduces `crisp` exactly — even where `lcrisp` is 1.9–2.4× better. Any "adaptive routing" claim is unsupported by these results.
4. **The estimator is simple and not new in kind.** It is a plug-in form of a known conditional mean; the contribution is structural-zero handling plus the constructive guarantee, not a new estimator class.
5. **The Dirichlet reading is exact only for the mean / geometric-mean profiles.** The default median profile converges to the population median and carries a non-vanishing bias.
6. **The downstream claim in the draft is not reproduced** — see §9.
7. **Nominal and actual missingness differ.** Missingness is injected only at non-zero entries, so a nominal 10 % becomes 1.4–8.3 % of cells; per-configuration values are in `HONEST_results.csv`.
8. **The R baselines have gaps.** `lrEMplus` succeeds on 116 of 1199 draws (it needs `n > d`); `impKNNa` and `missForest` succeed on all 1199.
9. **`lcrisp` is O(n²·d).** As of 1.1.0 the inner loops are vectorised (~110× faster, bit-identical output), but it still forms an `n × d` distance row per sample.
10. **One missing draw per subset** in `tmp_realnd/` / `tmp_realhr/`; the subset is the resampling unit, so a paired test across subsets resamples overlapping rows.

## 🔁 Reproducing the results

Everything in this repository regenerates from the scripts with fixed seeds. Requires Python ≥ 3.9 and, for the R baselines, R with `robCompositions` and `zCompositions`.

```bash
# datasets are bundled in data/ — no download needed
python gen_missing.py          # build tmp/       low-missingness draws
python gen_missing_nd.py       # build tmp_nd/    synthetic n<d draws
python gen_real_nd.py          # build tmp_realnd/  real n<d subsets
python gen_real_hr.py          # build tmp_realhr/  real HIGH-RATIO n<d subsets

# Python baselines + CRISP
python final_compare_v3.py     # -> final_results_v3.csv, wilcoxon_v3.csv
python final_nd_v3.py          # -> final_results_nd_v3.csv, wilcoxon_nd_v3.csv
python final_real_nd.py        # -> final_results_real_nd.csv

# honest re-run: adds +closure / +backfill fair-comparison arms,
# LCRISP / AutoCRISP arms, and the high-ratio real subsets
python honest_rerun.py         # -> HONEST_results.csv, HONEST_wilcoxon.csv
python summarize_honest.py     # -> honest_summary.txt
python softimpute_sensitivity.py   # -> softimpute_sensitivity.csv

# R baselines (optional but recommended)
Rscript run_r_baseline.R impKNNa tmp
Rscript run_r_baseline.R missForest tmp
Rscript run_lremplus_baseline.R    # -> lremplus_baseline.csv  (availability + accuracy)

# R baselines, multi-seed (so they can enter the paired tests) — slow, run in background
python stage_r_multiseed.py        # -> tmp_rms/ (1199 per-seed missing matrices)
Rscript run_r_baseline_multiseed.R # -> final_results_r_multiseed.csv

python make_figures.py         # -> figures/*.pdf
```

`X_true.csv` / `X_missing.csv` for every configuration are committed so the R baselines can be reproduced against the **exact same draws** the released tables used. The bulky per-configuration `impKNNa.csv` / `missForest.csv` outputs are gitignored — regenerate them with `run_r_baseline.R`.

---

## ✅ Testing

```bash
pip install pytest
pytest -q
```

**89 tests, ~2 s.** Two files:

`tests/test_crisp.py` pins the invariants the method is advertised on:

| Group | What is checked |
|---|---|
| Core invariants | closure of every imputed row; non-negativity; structural zeros preserved exactly; observed entries untouched; single-missing exact recovery; hand-computed multi-missing allocation |
| API | determinism; function ↔ estimator class agreement; `fit`/`transform` ↔ `fit_transform`; fitted profile reused across calls; DataFrame input; non-compositional columns untouched |
| Variants | `lcrisp` invariants; `n_neighbors` larger than `n`; `auto_crisp` dispatch below/above threshold; closure under `auto_crisp` |
| Helpers | `check_compositional_validity` (pass and fail cases); `project_to_simplex` (1-D, 2-D with feature indices, zero row); `compositional_mae` |
| Edge cases | no-missing identity; single row; fully-missing row; observed part exceeding `total`; custom `total`; entirely-missing column; fallback profile with <3 usable rows |
| Regression guards | full-observation rows are **not** re-projected (documents the non-zero SumDev); nearly-complete rows do enter the profile; NaN in a non-compositional column becomes a column mean |

The regression guards exist so that behaviour the released benchmarks depend on — including behaviour the manuscript describes differently — cannot drift silently.

`tests/test_baselines.py` covers the harness the results rest on:

| Group | What is checked |
|---|---|
| `make_missing` | never marks a structural zero (all mechanisms × rates); deterministic; actual rate matches nominal on non-zero entries; rejects unknown mechanisms |
| `find_data` | resolves the datasets bundled in `data/`; returns `None` for unknown files |
| `soft_impute` | shape and finiteness; determinism; beats column means on a genuinely low-rank matrix; `lam_ratio=0` provably degenerates to column-mean imputation; the centering flag has an effect; survives a fully-missing column; sweep helper returns the right keys |
| fair-comparison arms | `closure_project` closes every row and leaves zero rows at zero; `residual_backfill` solves single-missing rows exactly and leaves multi-missing rows to the baseline; `variants_of` exposes exactly the three documented arms |

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
├── final_compare_v3.py          # low-missingness comparison; reference soft_impute
├── final_nd_v3.py               # synthetic n<d comparison
├── final_real_nd.py             # real-material n<d subsets
├── honest_rerun.py              # +closure / +backfill arms, LCRISP / AutoCRISP, high-ratio subsets
├── summarize_honest.py          # honest tables
├── softimpute_sensitivity.py    # SoftImpute regularisation sweep vs the 1.0 implementation
├── gen_missing.py               # low-missingness missingness generator
├── gen_missing_nd.py            # synthetic n<d generator
├── gen_real_nd.py               # real n<d subset generator
├── gen_real_hr.py               # real HIGH-RATIO n<d subset generator
├── stage_r_multiseed.py         # stage per-seed missing matrices for the R baselines
├── make_figures.py              # figures
├── verify_crisp_theory.py       # theory sanity checks
├── run_r_baseline.R             # impKNNa / missForest baselines (single seed)
├── run_r_baseline_multiseed.R   # impKNNa / missForest / lrEMplus, multi-seed
├── run_lremplus_baseline.R      # lrEMplus availability + accuracy audit
│
├── tests/test_crisp.py          # library invariants (pytest)
├── tests/test_baselines.py      # harness: make_missing, find_data, soft_impute, fair arms
├── conftest.py                  # makes the repo root importable under pytest
├── .github/workflows/tests.yml  # CI: pytest on Python 3.9 / 3.11 / 3.12
│
├── data/                        # bundled datasets (steel, glass, ge) + provenance
├── tmp/  tmp_nd/                # 20-seed missingness draws (X_true / X_missing committed)
├── tmp_realnd/  tmp_realhr/     # real n<d and real high-ratio n<d subsets
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

The manuscript draft in this repository (`paper_draft.md`, `CRISP_PAPER_CN.pdf`) is **v5 and is under revision**. An independent audit of the released artifacts against the draft's claims found several statements that the data does not support:

| Draft claim | Status |
|---|---|
| "the standard compositional package cannot handle this data at all" | **Partly wrong.** `lrEMplus` *can* run — on 3 of 140 configurations. The accurate and much stronger statement is the `n > d` requirement: 109 configurations fail because `lrEMplus` needs more rows than columns. |
| "CRISP is 2–2.5× more accurate than the baselines" | **Overstated.** Against a closure-aware baseline the advantage is 1.3–1.6× on synthetic `n < d` at 10 % missingness, and absent elsewhere. The family wins 34 of 80 configurations. |
| "only CRISP-imputed matrices keep a positive downstream R²; MICE/KNN/mean collapse to negative" | **Not reproduced.** No generating script existed. Every method keeps a positive R² on the three larger datasets; CRISP is best only on `glass`. |
| "SumDev = 0 to machine precision" | **Overstated.** 0.0000 on low-missingness and real subsets; **0.0014–0.0029** on synthetic `n < d`, because complete rows are not re-projected. |
| "a genuinely high-dimensional real `n < d` dataset (ge_nd)" | **False as stated.** `ge_nd` is `ge` padded with 21 zero columns — effective `d/n = 0.45`. Replaced by the real high-ratio subsets (`d/n = 1.50–1.75`). |
| "SoftImpute is 3–12× worse than CRISP" | **Baseline was mis-implemented.** The 1.0 `soft_impute` was hard rank truncation; it overstated SoftImpute's error by 1.4–3.6×. The corrected conclusion still holds, but by a smaller margin. |
| "adaptive routing between global and local profiles" | **Unsupported.** `auto_crisp` routes on the missing rate alone, so it reproduces `crisp` on every low-missingness dataset — even where `lcrisp` is 1.9–2.4× better. |

[`HONEST_CRISP_2026-10-08.md`](HONEST_CRISP_2026-10-08.md) records, claim by claim, what holds, what does not, and what the corrected numbers are. **Prefer that document over the draft's §3 until the revision is published.**

---

## ⚠️ Known issues

**Resolved in 1.1.0**

- [x] **Unit tests** — 89 tests across `tests/test_crisp.py` and `tests/test_baselines.py`
- [x] **CI** — pytest on Python 3.9 / 3.11 / 3.12
- [x] **Self-contained data** — the three external datasets are bundled in `data/`
- [x] **`lcrisp` / `auto_crisp` benchmarked** — both now appear in `HONEST_results.csv`
- [x] **`lrEMplus` integrated** — availability and accuracy audited over all 140 configurations (`lremplus_baseline.csv`), summarised in the honest tables
- [x] **`ge_nd` replaced** — real high-ratio `n < d` subsets (`tmp_realhr/`) now carry the real-data claim; `ge_nd` is kept but explicitly labelled as *not* `n < d`
- [x] **`SoftImpute` reference implementation** — soft-thresholded singular values + centering; the 1.0 implementation (hard rank truncation) overstated its error by 1.4–3.6×
- [x] **R baselines multi-seed** — `run_r_baseline_multiseed.R` covers 1199 draws over 60 configurations, so they can enter the paired tests
- [x] **Unused imports removed** — `crisp.py` is pure NumPy; scikit-learn is no longer a runtime dependency
- [x] **Non-compositional NaN left untouched**
- [x] **All-NaN slices handled without RuntimeWarnings**

**Open**

- [ ] **`make_figures.py` does not regenerate from `HONEST_results.csv`.** Figure 3's SumDev panel and Figure 5 are hard-coded literals (`make_figures.py` around lines 59 and 98) rather than read from the result CSVs, and they still carry v1.0 numbers. Figures must be regenerated from the tables before any submission.
- [ ] **`lcrisp` is O(n²·d) in a Python loop.** Benchmarking it on the large datasets (n = 1030) dominates the honest re-run's runtime. Vectorise it (or use a spatial index) before recommending it for large pools.
- [ ] **The real high-ratio subsets reach d/n = 1.50–1.75, not the d/n > 2 of the synthetic pools.** A genuinely high-dimensional *real* compositional dataset with `n < d` could not be obtained, so the strongest `n < d` claim still rests on synthetic data.
- [ ] One missing draw per subset in `tmp_realnd/` / `tmp_realhr/`; the subset is the resampling unit, so a paired test over subsets resamples overlapping rows.
- [ ] No packaged release on PyPI — `pip install crisp-imputer` does not work yet.

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
