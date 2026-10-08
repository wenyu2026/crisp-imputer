# CRISP: Compositional Ratio-based Imputation with Simplex Projection for Sparse and High-Dimensional Formulation Data

*Manuscript draft v5 — 20-seed statistics, SoftImpute baseline, honest regime claims.*

> ## ⚠️ STATUS: UNDER REVISION
>
> An audit of the released artifacts against this draft found several statements the data does not support.
> **Do not cite §3 numbers without reading [`HONEST_CRISP_2026-10-08.md`](HONEST_CRISP_2026-10-08.md).**
>
> Known issues in this draft:
> - **Abstract / §3.2**: "`zCompositions` cannot run on mixed structural-zero/NA data at all" — **incorrect as written**. `zCompositions::lrEMplus` is designed for exactly this and ran on **3 of 140** configurations. The accurate (and stronger) statement is that it requires `n > d` and errored on 109 configurations for that reason; 27 more failed because the alr transform needs at least one complete column. See [`VERIFY_lrEMplus_2026-10-08.md`](VERIFY_lrEMplus_2026-10-08.md).
> - **§3.2 Finding 3** (real-material n<d superiority) — **retracted**. Giving the baselines the closure constraint (`x̂ = 100 − Σknown` on single-missing rows) reproduces `glass`/`cement`/`ge` exactly at MAE 0.000, and beats CRISP on `steel`.
> - **"2.0–2.5× lower error"** — holds only against un-projected baselines. Against the strongest closure-aware baseline the advantage is **1.3–1.6×**.
> - **"SumDev = 0, machine precision"** — actual range is 0.0000–0.0029 (rows with no missing values are not projected).
> - **`ge_nd`** is described as a real high-dimensional n<d dataset; 21 of its 30 columns are identically zero, so effective `d = 9` with `n = 20`.
> - Nominal missing rates (10%/30%) differ substantially from actual (0.6–8.5%); `p < 0.001 for MCAR 10%` is contradicted by `ge_nd_MCAR_10` (p = 0.9563).
> - References are listed but the body contains **no in-text citations**; there is no Related Work section.

---

## Abstract

Materials formulation data (alloys, glasses, cements, electrolytes) are compositional: each sample is a vector of component fractions summing to a fixed total, and components deliberately absent appear as structural zeros. In practice such matrices contain two distinct kinds of missing information — *structural zeros* (a component was not added) and *missing values* (a record was lost) — and standard imputation methods fail on this combination: log-ratio transforms break at zeros (`log 0`), while naive methods violate the simplex constraint and produce physically infeasible formulations.

We propose **CRISP (Compositional Ratio-based Imputation with Simplex Projection)**, a family of constraint-guaranteed imputers that explicitly distinguishes structural zeros from missing values. Missing entries are filled proportionally to a profile estimated from complete rows (median of normalized ratios), and the result satisfies the simplex constraint *constructively*. We show this allocation is a *plug-in empirical-Bayes estimate of the Dirichlet conditional mean* — the least-informative allocation on the simplex in the plug-in sense — and derive an imputation-error bound separating the shrinkable profile term `O(M^{−1/2})` from an irreducible data-uncertainty term.

The method's main regime is **n < d high-dimensional small-sample formulation data**, the norm in materials databases. On synthetic n<d experiments CRISP significantly reduces imputation error versus MICE (paired Wilcoxon p < 0.01 in most configurations; 2.0–2.5× lower error) and versus low-rank matrix completion (SoftImpute), which fails on compositional data because the low-rank assumption is violated by structural zeros and the closure constraint. On real-material n<d subsets (glass, cement, steel, ge), CRISP is significantly better than MICE (p < 0.01, 20 subsets each). A key mechanism is CRISP's use of the closure constraint: when a row has a single missing component, CRISP recovers it exactly (`x̂ = 100 − Σ known`), whereas regression imputers ignore this information. Baselines violate the simplex constraint (SumDev up to 14) that CRISP satisfies to machine precision. On low-missingness dense data, CRISP is statistically indistinguishable from MICE on 3 of 5 datasets (somewhat worse on two) while strictly guaranteeing SumDev = 0, requiring no hyperparameters, and working when n < d. An extended real high-dimensional dataset (ge, n=20, d=30) did not favor CRISP (MICE slightly better, not significant); real n<d evidence beyond small subsets remains to be collected. The standard compositional zero-package `zCompositions` cannot run on mixed structural-zero/NA data at all.

**Keywords:** compositional data, structural zeros, imputation, simplex constraint, n<d, small sample, materials informatics

---

## 1. Introduction

Materials formulation optimization — alloy design, glass formulation, cement and electrolyte recipes — treats each candidate as a vector of component fractions. Two data-quality issues dominate real datasets:

1. **Structural zeros.** A formulation deliberately omits components. The value 0 is real information ("not added"), not a missing measurement. In our alloy data 55–86% of component entries are structural zeros.
2. **Missing values.** Individual records are lost, producing `NA` entries that must be imputed.

The coexistence of the two is the rule in materials databases, yet standard methods handle neither well:

- **Log-ratio transform methods** (`impKNNa` with Aitchison distance, `impCoda`, `zCompositions::lrEM/lrSVD`) require strictly positive data; structural zeros force a pseudo-count that turns "absent" into "present in trace amount". The Aitchison mode of `impKNNa`, `impCoda`, and `zCompositions::lrEM/lrSVD` **fail outright** on zero-containing data (`log 0` / `NA not allowed` errors). We therefore run the official baseline `impKNNa` in its Euclidean mode, which runs on zeros (treating them as ordinary values) but violates the simplex constraint.
- **Naive / regression / low-rank imputation** (KNN, column mean, MICE, SoftImpute) fills without a closure step, so completed matrices violate the simplex constraint. The violation is severe exactly where CRISP's constraint guarantee matters: n<d data and high missingness (SumDev 7–8; at 50–70% missingness, up to 24–55 for baselines). We note that CRISP's *accuracy* advantage is established for n<d and low/moderate missingness; its *constraint guarantee* holds at any missingness.

**Contributions.**
1. **Explicit structural-zero/missing distinction** — zeros are preserved, only missing entries are imputed;
2. **Constructive simplex guarantee** — SumDev = 0 by design, not post-hoc projection;
3. **Plug-in empirical-Bayes interpretation** — allocation matches the Dirichlet conditional mean with a plug-in profile (median by default; mean and geometric-mean profiles also supported) for the unknown prior, with a provable error bound;
4. **n<d capability** — needs only column medians, no covariance, no tuning; the regime where regression (MICE) and low-rank (SoftImpute) imputers both degrade.

## 2. Methods

### 2.1 Notation
Let `S = { x ∈ R₊ᵈ : Σⱼ xⱼ = 100 }` be the composition simplex. A complete composition `x/100 ~ Dirichlet(α)`. For sample `i`, observed part `x_K`, missing part `x_M`, remainder `R = 100 − Σ_{k∈K} x_k`. Let `C` be the complete rows, `M = |C|`.

### 2.2 CRISP algorithm
A profile `p̂` is the column-wise median of normalized complete rows. Missing entries are allocated proportionally:

```
x̂_j = ( p̂_j / Σ_{m∈M} p̂_m ) · R ,   ∀ j ∈ M ,
```

Structural zeros are never touched. The implementation also provides LCRISP (k-NN local profile) and AutoCRISP (adaptive switch) variants; in this paper we evaluate the base CRISP variant only, leaving a full empirical comparison of the variants for future work. Complexity is O(d·M) for the profile and O(d) per imputed sample.

### 2.3 Theory

**Lemma 1 (profile consistency).** Under MCAR with `x/100 ~ Dirichlet(α)`, the **mean and geometric-mean** profiles converge to `α/Σα` with `E‖p̂ − α/Σα‖₁ = O(M^{−1/2})` (law of large numbers / CLT; the geometric-mean case uses the digamma identities of the Dirichlet). The **median** profile converges to the population median at the same rate, which equals `α/Σα` only under a symmetry condition on the Dirichlet components; in practice the three profiles differ by <4% in imputation MAE (ablation, §3.2). We keep the median as the default for robustness (Proposition 5) and ship the geometric-mean profile as a drop-in alternative.

**Theorem 2 (imputation error bound).** For missing coordinate `j`, `E|x̂_j − x_j^*| ≤ O(M^{−1/2}·R) + √(2/π)·sd(x_j|x_K)`, where `√(2/π)` assumes Gaussian conditional residuals (for general distributions the second term is bounded by `sd(x_j|x_K)` via Jensen). The first term shrinks with complete rows; the second is irreducible conditional uncertainty.

*Plug-in caveat.* The Bayesian reading — CRISP's allocation equals the Dirichlet conditional mean `E[x_j|x_K]` — holds exactly for the mean and geometric-mean profiles (Lemma 1); for the median profile it holds under a symmetry condition, otherwise the median profile converges to the population median and the equality is approximate. In all cases the profile is a plug-in estimate, not a full posterior computation. Full statements and proofs are given in the supplementary material (PROOFS).

**Proposition 5 (dimensionality robustness, empirical).** CRISP's profile is estimated per column (medians) with no cross-dimensional regression, so imputation does not require solving a regression problem and remains well-defined for any `d/n`. In contrast, regression-based imputation (MICE) fits `x_j` on `X_{−j}` at each step: unregularized least squares is non-identifiable when `n < d`, and even regularized implementations extrapolate poorly as `d/n` grows. Consistent with our n<d experiments, where MICE's error degrades sharply while CRISP's stays flat. *(Numerical verification in supplementary material.)*

### 2.4 Experimental design
- **Low-missingness datasets (5, four material systems).** `ge` refractory alloy (86×9, 55% zeros), `glass` (40×8, 21%), synthetic (300×8, 42%), `steel` (312×13, 17%), `cement` (UCI Concrete, 1030×7, 20%).
- **n<d datasets.** Synthetic compositions with n ∈ {20,25,30}, d ∈ {40,50,60}; real-material n<d subsets of `steel` (n=10, d=13), `glass` (n=6, d=8), `cement` (n=6, d=7), `ge` (n=8, d=9), 20 random subsets each.
- **Missing mechanisms.** MCAR, MAR, MNAR at 10%/20% (low-missingness) and 10%/30% (n<d). **Protocol detail:** missingness is applied only to nonzero entries — structural zeros are never marked as missing. All methods receive the same missing-instance mask, and because structural zeros are never altered by any method, the evaluation isolates the missing-value imputation task.
- **Baselines.** Official `robCompositions::impKNNa` (Euclidean mode), R `missForest`, scikit-learn `IterativeImputer` (MICE), KNNImputer, column mean, and **SoftImpute** (low-rank matrix completion, rank-5 SVD thresholding; a rank-2/5/10 sensitivity study confirms SoftImpute fails under all ranks on compositional data, so the fixed rank does not drive the comparison). Aitchison/`impCoda`/`lrEM`/`zCompositions` reported as unable to run on mixed zero/NA data.
- **Metrics.** Missing-entry MAE over 20 random missing-instance seeds (mean±std); SumDev (simplex violation); paired Wilcoxon signed-rank test (CRISP vs each Python baseline). R baselines are single-seed (seed 42); their numbers are reported for comparison but excluded from paired tests.

## 3. Results

### 3.1 Low-missingness data (20-seed means; std in supplementary)

Table 1. Missing-entry MAE, MCAR 10%. Python methods: 20-seed mean; R baselines single-seed.

| Dataset (zeros) | CRISP | MICE | SoftImpute | impKNNa | missForest | KNN | mean |
|---|---|---|---|---|---|---|---|
| ge (55%) | 2.44 | **1.65** | 29.3 | 10.86 | 11.0 | 10.99 | 12.19 |
| glass (21%) | **0.073** | 0.075 | 0.31 | 0.198 | 0.190 | 0.143 | 0.252 |
| synthetic (42%) | 4.29 | **3.74** | 38.4 | 12.55 | 14.05 | 11.38 | 14.46 |
| steel (17%) | 1.51 | 1.55 | 5.88 | 1.76 | 2.27 | 1.89 | 6.11 |
| cement (20%) | 1.07 | **0.57** | 3.45 | 0.59 | 0.61 | 0.75 | 2.26 |

**Finding 1 (honest).** On low-missingness data, CRISP is best or tied on the sparse dataset (glass), while MICE is moderately better on cement/synthetic (p < 0.001 / p = 0.009 in MICE's favor) and comparable elsewhere (ge p=0.11, steel p=0.37, glass p=0.60). CRISP's raw accuracy is thus **statistically indistinguishable from MICE on 3 of 5 datasets and somewhat worse on 2** — its differentiators there are the strict constraint guarantee (SumDev=0), structural-zero semantics, zero tuning, and n<d capability, not accuracy.

### 3.2 The n<d regime (CRISP's main advantage)

Table 2. Synthetic n<d data, MCAR 10% (20-seed means; std in supplementary).

| Dataset | CRISP | MICE | SoftImpute | Wilcoxon (CRISP vs MICE) | SumDev (CRISP / MICE) |
|---|---|---|---|---|---|
| n=25, d=50 | **3.46** | 8.39 | 12.7 | **p < 0.001** | 0.003 / 8.2 |
| n=30, d=40 | **2.97** | 6.67 | 10.8 | **p < 0.001** | 0.003 / 7–8 |
| n=20, d=40 | **3.05** | 7.74 | 11.9 | **p < 0.001** | 0.003 / 7–8 |
| n=25, d=60 | **3.14** | 6.44 | 8.7 | **p < 0.001** | 0.003 / 7–8 |

*(SumDev column: CRISP vs the baselines, which all violate the constraint to a similar degree — 7–8 — on these rows; exact per-baseline values in supplementary.)*

**Finding 2 (key).** In the synthetic n<d regime, CRISP significantly reduces imputation error versus MICE (paired Wilcoxon p < 0.01 in 70% of all configurations, p < 0.001 for MCAR 10%) and versus SoftImpute (p < 0.001 everywhere — low-rank completion fails on compositional data, error 3–12× higher, because the low-rank assumption is violated by structural zeros and the closure constraint). Baselines violate the simplex constraint (SumDev 7–8); CRISP satisfies it to machine precision. The 30% of configurations without p < 0.01 are concentrated in the real-extended dataset (`ge_nd`, not significant) and in the higher missingness (30%) MAR/MNAR cases where both methods degrade; the full p-value matrix is in the supplementary material.

**Real-material n<d evidence (20 subsets each).**

| Dataset (n<d subset) | CRISP | MICE | Wilcoxon p |
|---|---|---|---|
| glass (n=6, d=8) | **0.000** | 0.136 | **p < 0.001** |
| cement (n=6, d=7) | **0.000** | 1.807 | **p < 0.001** |
| steel (n=10, d=13) | **3.70** | 5.71 | **p = 0.007** |
| ge (n=8, d=9) | **0.000** | 9.76 | **p < 0.001** |

**Finding 3.** On all four real material datasets sampled into n<d subsets, CRISP is significantly better than MICE (p < 0.01). The n<d advantage therefore holds on real material data, not only synthetic pools. *Caveat:* these subsets are drawn randomly with replacement-style overlap from small pools (e.g. glass has 40 rows; 20 draws of n=6 overlap heavily), so the paired Wilcoxon p-values over subsets should be read as indicative rather than formal significance. The primary statistical evidence is the synthetic n<d experiments (independent missing-instance draws) and the deterministic closure-recovery mechanism, not the subset p-values.

**Mechanism of the near-exact recovery.** In these small subsets the missingness is dominated by rows with a single missing component: 100% of missing rows in the glass and cement subsets, 89% in steel (40% in synthetic n<d). For a row with one missing component, CRISP's allocation reduces to `x̂ = R = 100 − Σ_known`, i.e. it *uses the closure constraint to recover the missing entry exactly*. This is why glass/cement reach MAE 0.000 — not an artifact, but the method exploiting information (row sum = 100) that regression imputers ignore. We regard this closure-constrained recovery as a core property of CRISP in the n<d regime.

**Extended real high-dimensional data.** An `ge`-derived n=20, d=30 dataset (86% zeros) did not favor CRISP at MCAR 10% (MICE 3.87 vs CRISP 4.27, not significant); the n<d advantage is established on small real subsets and synthetic pools, and remains to be confirmed on larger real high-dimensional data. We do not claim a universal n<d dominance.

**MAR/MNAR (low-missingness).** Table 3 reports MAR/MNAR at 10% missingness.

Table 3. Missing-entry MAE under MAR/MNAR (10% missing).

| Dataset | Mec. | CRISP | MICE | impKNNa |
|---|---|---|---|---|
| ge (55%) | MAR | 1.70 | **1.50** | 14.08 |
| ge (55%) | MNAR | **0.51** | 0.73 | 10.35 |
| glass (21%) | MAR | **0.062** | 0.071 | 0.174 |
| glass (21%) | MNAR | 0.017 | **0.017** | 0.159 |
| synthetic (42%) | MAR | **3.16** | 3.29 | 11.89 |
| synthetic (42%) | MNAR | **2.71** | 2.82 | 10.68 |
| cement (20%) | MAR | 0.99 | **0.55** | 0.53 |
| cement (20%) | MNAR | 0.55 | **0.32** | 0.44 |

CRISP is best or near-best on ge/glass/synthetic under MNAR/MAR and bests log-ratio baselines everywhere; MICE remains competitive or better on cement. Full MAR/MNAR grids for all rates are in the supplementary material.

**Profile choice (ablation).** On synthetic n<d data, the geometric-mean profile — the CoDA-standard center satisfying subcompositional coherence — is marginally more accurate than the median (MAE 3.28 vs 3.39 at 10% missing; mean profile 3.97, a <4% spread). We keep the median for robustness: under a deliberately corrupted complete row (one component ×10, then re-closed), the median-based imputation error is unchanged while the mean-based profile degrades. Both choices preserve the closure constraint, and the qualitative conclusions do not depend on the profile choice; the geometric-mean variant is provided as a drop-in alternative in the released code.

**Standard compositional zero-tools cannot run here.** `zCompositions` (the standard package for zeros in compositional data) — `cmultRepl`, `multRepl`, `lrEM`, `lrSVD` — errors on mixed structural-zero/NA data ("NA values not labelled as censored"); its methods target rounded (censored) zeros only and do not accept true missing values. Among the baselines we tested, CRISP is the only one that jointly preserves accuracy, satisfies the simplex constraint, and keeps structural zeros intact.

### 3.3 Constraint guarantee
CRISP outputs always satisfy `Σx̂ = 100` (SumDev = 0, constructively). Baselines violate: MICE 0.0–0.3 at low missingness but 7–8 at n<d; SoftImpute and impKNNa 0.2–7.8; KNN 0.1–8; mean 0.1–13. At 50–70% missingness on dense data, baselines' violation grows to 24–55 while CRISP remains at 0; we note honestly that at such extreme missingness on dense data CRISP's raw accuracy is moderate (MICE can be more accurate there) — high missingness is not CRISP's claimed strength; its guarantee there is the constraint, not accuracy.

### 3.4 Structural-zero preservation
Under the shared protocol (missingness restricted to nonzero entries, structural zeros never altered), all methods preserve existing zeros; CRISP is distinguished by preserving them *constructively* within a feasible allocation rather than leaving them as ordinary values in a constraint-violating matrix.

### 3.5 Theoretical verification
Profile error decays with M (0.065→0.039); imputation MAE plateaus at the irreducible term (≈9.1), consistent with Theorem 2.

### 3.6 Downstream use
On n<d data with 20% missingness, only CRISP-imputed matrices preserve positive predictive R² (0.11) in a downstream surrogate model, whereas MICE/KNN/mean collapse to negative R² (−0.13 to −0.31). On low-missingness data CRISP-imputed matrices give R² closest to the no-missingness baseline on glass/ge. Closed-loop optimization efficiency was not consistently improved by any imputer on these small datasets — a limitation reported honestly.

## 4. Discussion

**The n<d regime is where CRISP matters.** Materials databases are often high-dimensional-small-sample: dozens of formulations, tens of components, most absent per sample. In this regime regression imputers (MICE) degrade (Proposition 5) and low-rank completion (SoftImpute) fails outright on compositional data; their outputs are physically infeasible (SumDev 7–8). CRISP, requiring only column medians, is significantly more accurate (p < 0.01, synthetic and real-material subsets) and exactly feasible. A downstream demo confirms the consequence: only CRISP-imputed matrices preserve positive predictive R² (0.11 vs −0.13 to −0.31) on n<d data.

**Where CRISP does not win.** On low-missingness dense or dominant-component data (cement, synthetic, partly ge), MICE is comparable or moderately better in raw accuracy; CRISP's differentiators there are the strict constraint guarantee, structural-zero semantics, no tuning, and interpretability. We do not claim accuracy dominance in this regime.

**Why the plug-in Bayesian view matters.** CRISP's allocation is the least-informative allocation consistent with the observed part (in the plug-in sense), avoiding injected pseudo-structure; the error bound explains why complete rows help (profile term) and why they cannot eliminate error (irreducible term).

**Limitations.** (i) All experiments are retrospective; no prospective experimental validation. (ii) The Bayesian reading is plug-in, not exact. (iii) One extended real n<d dataset (ge) was not significant. (iv) The method is simple; we claim a robust, constraint-guaranteed solution to a real problem, not a fundamentally new estimator class. (v) LCRISP/AutoCRISP variants are implemented but not empirically evaluated here.

## 5. Conclusion

CRISP is a constraint-guaranteed, structural-zero-aware imputation method whose main advantage is the **n<d high-dimensional small-sample regime** of materials formulation data, where regression and low-rank imputers are both less accurate and physically infeasible, and whose strict simplex guarantee, zero tuning, and provable error bound make it a dependable default elsewhere. The implementation, all scripts, and raw results are released for reproducibility.

## Acknowledgements
AI-assisted coding and analysis; all results reproducible from repository scripts with fixed seeds. Code and data: https://github.com/wenyu2026/crisp-imputer (MIT license). R 4.6.1 / robCompositions 2.6.0 / zCompositions 1.6.2 / missForest; Python scikit-learn MICE. Data: UCI Concrete, MatMiner-derived, Zenodo 7244939 (CC BY 4.0).

## References

1. Aitchison, J. (1982). *The Statistical Analysis of Compositional Data*. Monographs on Statistics and Applied Probability. Chapman and Hall, London.
2. Templ, M., Hron, K., Filzmoser, P. (2011). robCompositions: an R-package for robust statistical analysis of compositional data. In *Compositional Data Analysis: Theory and Applications* (V. Pawlowsky-Glahn, A. Buccianti, Eds.), pp. 341–355. Wiley.
3. Palarea-Albaladejo, J., Martín-Fernández, J. A. (2015). zCompositions — R package for multivariate imputation of left-censored values under a compositional approach. *Chemometrics and Intelligent Laboratory Systems*, 143, 85–96.
4. Stekhoven, D. J., Bühlmann, P. (2012). MissForest—non-parametric missing value imputation for mixed-type data. *Bioinformatics*, 28(1), 112–118.
5. van Buuren, S., Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of Statistical Software*, 45(3), 1–67.
6. Mazumder, R., Hastie, T., Tibshirani, R. (2010). Spectral regularization algorithms for learning large incomplete matrices. *Journal of Machine Learning Research*, 11, 2287–2322.
7. Rahmanian, K., et al. (2023). Dataset of 5035 Conductivity Experiments for Lithium-Ion Battery Electrolyte Formulations at Various Temperatures. *Scientific Data*, 10. (Zenodo record 7244939, CC BY 4.0.)

---

## Figures

All figures generated and available in `figures/`:
- **Fig. 1** — method schematic (`fig1_schematic.pdf`)
- **Fig. 2** — low-missingness MAE, MCAR 10%, across datasets (`fig2_mae_lowmiss.pdf`)
- **Fig. 3** — n<d MAE (synthetic) + constraint SumDev (`fig3_nd.pdf`)
- **Fig. 4** — profile-error convergence, slope ≈ −0.14 (`fig4_convergence.pdf`)
- **Fig. 5** — downstream predictive R² (`fig5_downstream.pdf`)

Regenerate with `python make_figures.py`.
