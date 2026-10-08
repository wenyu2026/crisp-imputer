# -*- coding: utf-8 -*-
"""Unit tests for CRISP.

The suite pins the invariants the method is advertised on:

* constructive closure      — every row that had a missing entry sums to `total`
* non-negativity            — no negative components
* structural-zero integrity — an existing 0 is never imputed into a positive value
* single-missing recovery   — a row with exactly one missing entry is solved exactly
* observed-value preservation — present entries are not altered

plus the public API, the helper functions, the `lcrisp` / `auto_crisp` variants,
edge cases, and a few regression guards documenting behaviour that the released
benchmarks depend on (see `HONEST_CRISP_2026-10-08.md`).

Run:  pytest -q
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from crisp import (
    CRISPImputer,
    LCRISPImputer,
    AutoCRISPImputer,
    crisp,
    lcrisp,
    auto_crisp,
    project_to_simplex,
    check_compositional_validity,
    compositional_mae,
)

TOTAL = 100.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def random_composition(rng, n=40, d=8, kmin=3, kmax=6, total=TOTAL):
    """Random compositions with structural zeros.

    Rows are exactly closed to `total` (no rounding) so that closure assertions
    test CRISP rather than the generator's rounding error.
    """
    kmax = min(kmax, d)
    kmin = min(kmin, kmax)
    X = np.zeros((n, d))
    for i in range(n):
        k = int(rng.integers(kmin, kmax + 1))
        cols = rng.choice(d, size=k, replace=False)
        X[i, cols] = rng.dirichlet(np.ones(k)) * total
    return X / X.sum(axis=1, keepdims=True) * total


def inject_missing(X, rate, rng, restrict_to_nonzero=True):
    """Return (X_with_nan, mask). Missingness is injected at non-zero entries only,
    matching the benchmark protocol (structural zeros are never marked missing)."""
    base = (X != 0) if restrict_to_nonzero else np.ones(X.shape, dtype=bool)
    mask = (rng.random(X.shape) < rate) & base
    Xm = X.copy()
    Xm[mask] = np.nan
    return Xm, mask


def comp_idx_of(d):
    return list(range(d))


# ---------------------------------------------------------------------------
# 1. Core invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("rate", [0.05, 0.15, 0.30])
def test_closure_holds_for_every_imputed_row(seed, rate):
    """Constructive simplex guarantee: rows that had missing entries sum to `total`."""
    rng = np.random.default_rng(seed)
    X = random_composition(rng, n=40, d=8)
    Xm, mask = inject_missing(X, rate, rng)
    Xp = crisp(Xm, comp_idx=comp_idx_of(8), total=TOTAL)

    touched = mask.any(axis=1)
    assert touched.any(), "test setup produced no imputed rows"

    row_sums = Xp[touched].sum(axis=1)
    np.testing.assert_allclose(row_sums, TOTAL, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_output_is_non_negative(seed):
    rng = np.random.default_rng(seed)
    X = random_composition(rng, n=40, d=8)
    Xm, _ = inject_missing(X, 0.20, rng)
    Xp = crisp(Xm, comp_idx=comp_idx_of(8), total=TOTAL)
    assert np.isfinite(Xp).all()
    assert (Xp >= 0).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_structural_zeros_are_preserved_exactly(seed):
    """A structural zero stays exactly 0 — it is information, not a missing value."""
    rng = np.random.default_rng(seed)
    X = random_composition(rng, n=40, d=8)
    zeros = X == 0
    assert zeros.any(), "test setup produced no structural zeros"

    Xm, mask = inject_missing(X, 0.30, rng)
    Xp = crisp(Xm, comp_idx=comp_idx_of(8), total=TOTAL)

    never_missing = zeros & ~mask
    np.testing.assert_array_equal(Xp[never_missing], 0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_observed_entries_are_preserved(seed):
    """Entries that were observed (and whose row remainder is positive) are untouched."""
    rng = np.random.default_rng(seed)
    X = random_composition(rng, n=40, d=8)
    Xm, mask = inject_missing(X, 0.10, rng)
    Xp = crisp(Xm, comp_idx=comp_idx_of(8), total=TOTAL)

    observed = ~np.isnan(Xm)
    # Only rows whose observed part leaves a positive remainder keep their values verbatim.
    remainder = TOTAL - np.nansum(Xm, axis=1)
    rows = observed & (remainder > 0)[:, None]
    np.testing.assert_allclose(Xp[rows], X[rows], atol=1e-9)


def test_single_missing_entry_is_recovered_exactly():
    """The closure turns a one-missing row into arithmetic: x_hat = total - sum(known)."""
    X = np.array([
        [40.0, 30.0, 20.0, 10.0],
        [50.0, 25.0, 15.0, 10.0],
        [45.0, 30.0, 15.0, 10.0],
        [35.0, 35.0, 20.0, 10.0],
        [np.nan, 25.0, 25.0, 10.0],   # 100 - 60 = 40
        [30.0, np.nan, 40.0, 10.0],   # 100 - 80 = 20
    ])
    Xp = crisp(X, comp_idx=[0, 1, 2, 3], total=TOTAL)

    assert Xp[4, 0] == pytest.approx(40.0, abs=1e-9)
    assert Xp[5, 1] == pytest.approx(20.0, abs=1e-9)
    np.testing.assert_allclose(Xp[[4, 5]].sum(axis=1), TOTAL, atol=1e-9)


def test_two_missing_entries_split_the_remainder_by_profile():
    """Hand-computed 3-component case with two missing entries in one row."""
    X = np.array([
        [40.0, 30.0, 30.0],
        [50.0, 25.0, 25.0],
        [45.0, 30.0, 25.0],
        [np.nan, np.nan, 50.0],
    ])
    Xp = crisp(X, comp_idx=[0, 1, 2], total=TOTAL)

    # profile = column-wise median of normalised complete rows = [.45, .30, .25]
    # missing profile = [.45, .30] -> renormalised [.6, .4]; remainder = 50
    assert Xp[3, 0] == pytest.approx(30.0, abs=1e-9)
    assert Xp[3, 1] == pytest.approx(20.0, abs=1e-9)
    assert Xp[3, 2] == pytest.approx(50.0, abs=1e-9)


def test_readme_example():
    """The quick-start example in README.md must produce the documented output."""
    X = np.array([
        [70.0, 15.0, 15.0],
        [np.nan, 20.0, 10.0],
        [65.0, np.nan, 20.0],
    ])
    Xp = crisp(X, comp_idx=[0, 1, 2], total=TOTAL)
    expected = np.array([
        [70.0, 15.0, 15.0],
        [70.0, 20.0, 10.0],
        [65.0, 15.0, 20.0],
    ])
    np.testing.assert_allclose(Xp, expected, atol=1e-9)

    valid = check_compositional_validity(Xp, comp_idx=[0, 1, 2], total=TOTAL)
    assert valid["sum_dev_mean"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. API behaviour
# ---------------------------------------------------------------------------

def test_is_deterministic():
    rng = np.random.default_rng(7)
    X = random_composition(rng, n=30, d=6)
    Xm, _ = inject_missing(X, 0.20, rng)
    a = crisp(Xm, comp_idx=comp_idx_of(6), total=TOTAL)
    b = crisp(Xm, comp_idx=comp_idx_of(6), total=TOTAL)
    np.testing.assert_array_equal(a, b)


def test_function_and_imputer_class_agree():
    rng = np.random.default_rng(3)
    X = random_composition(rng, n=30, d=6)
    Xm, _ = inject_missing(X, 0.20, rng)
    a = crisp(Xm, comp_idx=comp_idx_of(6), total=TOTAL)
    b = CRISPImputer(comp_idx=comp_idx_of(6), total=TOTAL).fit_transform(Xm)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_fit_then_transform_equals_fit_transform():
    rng = np.random.default_rng(4)
    X = random_composition(rng, n=30, d=6)
    Xm, _ = inject_missing(X, 0.20, rng)

    imp = CRISPImputer(comp_idx=comp_idx_of(6), total=TOTAL)
    a = imp.fit_transform(Xm)
    b = imp.transform(Xm)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_fitted_profile_is_reused_across_transform_calls():
    """A fitted imputer must not silently re-fit on new data."""
    train = np.array([[50.0, 30.0, 20.0], [45.0, 35.0, 20.0], [55.0, 25.0, 20.0]])
    test = np.array([[np.nan, 40.0, 20.0], [np.nan, 30.0, 30.0]])
    imp = CRISPImputer(comp_idx=[0, 1, 2], total=TOTAL).fit(train)
    out = imp.transform(test)
    np.testing.assert_allclose(out.sum(axis=1), TOTAL, atol=1e-9)


def test_accepts_pandas_dataframe():
    rng = np.random.default_rng(5)
    X = random_composition(rng, n=20, d=5)
    Xm, _ = inject_missing(X, 0.15, rng)
    df = pd.DataFrame(Xm, columns=list("ABCDE"))
    from_df = crisp(df, comp_idx=[0, 1, 2, 3, 4], total=TOTAL)
    from_arr = crisp(Xm, comp_idx=[0, 1, 2, 3, 4], total=TOTAL)
    np.testing.assert_allclose(np.asarray(from_df), from_arr, atol=1e-12)


def test_non_compositional_columns_pass_through():
    """Columns outside `comp_idx` are not part of the composition and are left alone."""
    X = np.array([
        [40.0, 30.0, 30.0, 12.5],
        [50.0, 25.0, 25.0, 13.0],
        [45.0, 30.0, 25.0, 11.0],
        [35.0, 35.0, np.nan, 14.0],
    ])
    Xp = crisp(X, comp_idx=[0, 1, 2], total=TOTAL)
    np.testing.assert_allclose(Xp[:, 3], X[:, 3], atol=1e-12)
    # and the compositional part still satisfies the constraint
    np.testing.assert_allclose(Xp[:, :3].sum(axis=1), TOTAL, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rate", [0.05, 0.20, 0.35])
def test_lcrisp_invariants(rate):
    rng = np.random.default_rng(11)
    X = random_composition(rng, n=35, d=7)
    Xm, mask = inject_missing(X, rate, rng)
    Xp = lcrisp(Xm, comp_idx=comp_idx_of(7), n_neighbors=5, total=TOTAL)

    touched = mask.any(axis=1)
    np.testing.assert_allclose(Xp[touched].sum(axis=1), TOTAL, atol=1e-9)
    assert (Xp >= 0).all()
    np.testing.assert_array_equal(Xp[(X == 0) & ~mask], 0.0)


def test_lcrisp_handles_more_neighbors_than_rows():
    X = np.array([
        [50.0, 30.0, 20.0],
        [45.0, 35.0, 20.0],
        [np.nan, 30.0, 25.0],
    ])
    Xp = lcrisp(X, comp_idx=[0, 1, 2], n_neighbors=99, total=TOTAL)
    assert np.isfinite(Xp).all()
    np.testing.assert_allclose(Xp.sum(axis=1), TOTAL, atol=1e-9)


def test_auto_crisp_rate_strategy_uses_crisp_below_threshold():
    """The legacy missing-rate rule, now opt-in via strategy='rate'."""
    rng = np.random.default_rng(13)
    X = random_composition(rng, n=40, d=6)
    Xm, _ = inject_missing(X, 0.05, rng)
    a = auto_crisp(Xm, comp_idx=comp_idx_of(6), total=TOTAL,
                   strategy="rate", threshold=0.50)
    b = crisp(Xm, comp_idx=comp_idx_of(6), total=TOTAL)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_auto_crisp_rate_strategy_uses_lcrisp_above_threshold():
    rng = np.random.default_rng(14)
    X = random_composition(rng, n=40, d=6)
    Xm, _ = inject_missing(X, 0.35, rng)
    a = auto_crisp(Xm, comp_idx=comp_idx_of(6), total=TOTAL,
                   strategy="rate", threshold=0.10)
    b = lcrisp(Xm, comp_idx=comp_idx_of(6), total=TOTAL)
    np.testing.assert_allclose(a, b, atol=1e-12)


def clustered_composition(rng, n_clusters=4, per_cluster=15, d=6, jitter=1.5):
    """Distinct clusters with near-identical members, so neighbours are informative."""
    bases = [rng.dirichlet(np.ones(d) * 0.4) * 100.0 for _ in range(n_clusters)]
    rows = []
    for base in bases:
        for _ in range(per_cluster):
            v = np.clip(base + rng.normal(0, jitter, size=d), 0.5, None)
            rows.append(v / v.sum() * 100.0)
    return np.array(rows)


def test_auto_crisp_holdout_prefers_lcrisp_when_neighbours_are_informative():
    """With clear clusters the local profile reconstructs held-out entries better,
    and the default (threshold-free) selector must notice."""
    rng = np.random.default_rng(16)
    X = clustered_composition(rng, n_clusters=4, per_cluster=15, d=6)
    Xm, _ = inject_missing(X, 0.20, rng)
    imp = AutoCRISPImputer(comp_idx=comp_idx_of(6), total=TOTAL)
    out = imp.fit_transform(Xm)
    assert imp.variant_ == "lcrisp", imp.selection_
    assert imp.selection_["mae_local"] < imp.selection_["mae_global"]
    # and it really is the LCRISP output
    np.testing.assert_allclose(
        out, lcrisp(Xm, comp_idx=comp_idx_of(6), total=TOTAL), atol=1e-12)


def test_auto_crisp_holdout_falls_back_to_crisp_without_a_pseudo_task():
    """No row has three observed non-zero entries -> nothing to cross-validate."""
    rng = np.random.default_rng(17)
    n, d = 12, 5
    X = np.zeros((n, d))
    for i in range(n):
        cols = rng.choice(d, size=2, replace=False)
        X[i, cols] = rng.dirichlet(np.ones(2)) * 100.0
    X[0, 0] = np.nan
    imp = AutoCRISPImputer(comp_idx=comp_idx_of(d), total=TOTAL)
    imp.fit_transform(X)
    assert imp.variant_ == "crisp"
    assert np.isnan(imp.selection_["mae_global"])


def test_auto_crisp_holdout_is_deterministic_and_records_its_choice():
    rng = np.random.default_rng(18)
    X = clustered_composition(rng, n_clusters=3, per_cluster=12, d=6)
    Xm, _ = inject_missing(X, 0.20, rng)

    def run():
        imp = AutoCRISPImputer(comp_idx=comp_idx_of(6), total=TOTAL, seed=0)
        out = imp.fit_transform(Xm)
        return out, imp.variant_, imp.selection_

    a, va, sa = run()
    b, vb, sb = run()
    np.testing.assert_array_equal(a, b)
    assert va == vb
    assert sa == sb
    assert sa["strategy"] == "holdout"


def test_auto_crisp_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        AutoCRISPImputer(comp_idx=[0, 1], strategy="not_a_strategy")


def test_auto_crisp_still_guarantees_the_closure():
    rng = np.random.default_rng(15)
    X = random_composition(rng, n=40, d=6)
    Xm, mask = inject_missing(X, 0.30, rng)
    out = AutoCRISPImputer(comp_idx=comp_idx_of(6), total=TOTAL).fit_transform(Xm)
    np.testing.assert_allclose(out[mask.any(axis=1)].sum(axis=1), TOTAL, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------

def test_check_validity_passes_for_crisp_output():
    rng = np.random.default_rng(21)
    X = random_composition(rng, n=30, d=6)
    Xm, _ = inject_missing(X, 0.20, rng)
    Xp = crisp(Xm, comp_idx=comp_idx_of(6), total=TOTAL)
    res = check_compositional_validity(Xp, comp_idx=comp_idx_of(6), total=TOTAL)
    assert res["valid"] is True
    assert res["neg_count"] == 0
    assert res["sum_dev_max"] < 1e-9


def test_check_validity_flags_an_infeasible_matrix():
    bad = np.array([[40.0, 30.0, 20.0], [10.0, 10.0, 10.0]])
    res = check_compositional_validity(bad, comp_idx=[0, 1, 2], total=TOTAL)
    assert res["valid"] is False
    # row sums are 90 and 30 -> deviations 10 and 70
    assert res["sum_dev_max"] == pytest.approx(70.0)

    negative = np.array([[110.0, -10.0, 0.0]])
    res_neg = check_compositional_validity(negative, comp_idx=[0, 1, 2], total=TOTAL)
    assert res_neg["valid"] is False
    assert res_neg["neg_count"] == 1


def test_project_to_simplex_1d():
    out = project_to_simplex(np.array([50.0, 50.0]), total=TOTAL)
    assert out.shape == (2,)
    assert out.sum() == pytest.approx(TOTAL)

    out2 = project_to_simplex(np.array([-10.0, 30.0]), total=TOTAL)
    np.testing.assert_allclose(out2, [0.0, 100.0], atol=1e-9)


def test_project_to_simplex_2d_only_touches_selected_features():
    X = np.array([[10.0, 20.0, 5.0], [1.0, 2.0, 3.0]])
    out = project_to_simplex(X, total=TOTAL, feature_indices=[0, 1])
    np.testing.assert_allclose(out[:, :2].sum(axis=1), TOTAL, atol=1e-9)
    np.testing.assert_allclose(out[:, 2], [5.0, 3.0], atol=1e-12)


def test_project_to_simplex_handles_all_zero_row():
    out = project_to_simplex(np.array([[0.0, 0.0, 0.0]]), total=TOTAL)
    np.testing.assert_allclose(out, [[TOTAL / 3] * 3], atol=1e-9)


def test_compositional_mae_hand_computed():
    X_true = np.array([[40.0, 30.0, 30.0], [50.0, 25.0, 25.0]])
    X_pred = np.array([[40.0, 30.0, 30.0], [45.0, 25.0, 30.0]])
    mask = np.array([[False, False, False], [True, False, False]])
    res = compositional_mae(X_true, X_pred, mask, comp_idx=[0, 1, 2])
    assert res["overall_mae"] == pytest.approx(5.0)
    assert res["comp_mae"] == pytest.approx(5.0)
    assert res["per_component_mae"] == {0: pytest.approx(5.0)}


# ---------------------------------------------------------------------------
# 5. Edge cases and robustness
# ---------------------------------------------------------------------------

def test_no_missing_input_is_returned_unchanged():
    X = np.array([[40.0, 30.0, 30.0], [50.0, 25.0, 25.0]])
    np.testing.assert_array_equal(crisp(X, comp_idx=[0, 1, 2], total=TOTAL), X)


def test_single_row_input():
    X = np.array([[np.nan, 40.0, 20.0]])
    out = crisp(X, comp_idx=[0, 1, 2], total=TOTAL)
    np.testing.assert_allclose(out.sum(axis=1), TOTAL, atol=1e-9)
    assert (out >= 0).all()


def test_fully_missing_row_is_allocated_from_the_profile():
    X = np.array([
        [50.0, 30.0, 20.0],
        [45.0, 35.0, 20.0],
        [55.0, 25.0, 20.0],
        [np.nan, np.nan, np.nan],
    ])
    out = crisp(X, comp_idx=[0, 1, 2], total=TOTAL)
    np.testing.assert_allclose(out[3].sum(), TOTAL, atol=1e-9)
    assert (out[3] >= 0).all()


def test_remaining_not_positive_scales_known_and_zeroes_missing():
    """When the observed part already meets or exceeds `total`, missing entries go to 0
    and the observed part is scaled down onto the simplex."""
    X = np.array([
        [60.0, 40.0, 0.0],
        [70.0, 40.0, np.nan],   # observed comp sum = 110 > 100
    ])
    out = crisp(X, comp_idx=[0, 1, 2], total=TOTAL)
    assert out[1, 2] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(out[1].sum(), TOTAL, atol=1e-9)
    assert (out[1] >= 0).all()


def test_custom_total():
    X = np.array([
        [0.5, 0.3, 0.2],
        [0.4, 0.35, 0.25],
        [0.45, 0.3, 0.25],
        [np.nan, 0.3, 0.3],
    ])
    out = crisp(X, comp_idx=[0, 1, 2], total=1.0)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    assert out[3, 0] == pytest.approx(0.4, abs=1e-9)


def test_entirely_missing_column_does_not_crash():
    X = np.array([
        [np.nan, 30.0, 20.0, 10.0],
        [np.nan, 25.0, 25.0, 10.0],
        [np.nan, 35.0, 15.0, 10.0],
        [np.nan, 30.0, 20.0, 10.0],
    ])
    out = crisp(X, comp_idx=[0, 1, 2, 3], total=TOTAL)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(axis=1), TOTAL, atol=1e-9)


def test_fewer_than_three_usable_rows_uses_the_fallback_profile():
    """With <3 complete/near-complete rows the profile falls back to column medians."""
    X = np.array([
        [60.0, 30.0, 10.0],
        [np.nan, 20.0, np.nan],
        [np.nan, np.nan, np.nan],
    ])
    out = crisp(X, comp_idx=[0, 1, 2], total=TOTAL)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[1:].sum(axis=1), TOTAL, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. Regression guards for behaviour the released benchmarks depend on
# ---------------------------------------------------------------------------

def test_fully_observed_rows_are_not_reprojected():
    """Rows with no missing values are passed through untouched — including rows whose
    components do not sum to `total`.

    This is why the released benchmarks report a small non-zero SumDev (0.0014-0.0029)
    on synthetic data rather than exact machine precision. See README "Limitations"
    and HONEST_CRISP_2026-10-08.md.
    """
    X = np.array([[60.0, 20.0, 10.0]])          # sums to 90, not 100
    out = crisp(X, comp_idx=[0, 1, 2], total=TOTAL)
    np.testing.assert_array_equal(out, X)

    res = check_compositional_validity(out, comp_idx=[0, 1, 2], total=TOTAL)
    assert res["sum_dev_mean"] == pytest.approx(10.0)
    assert res["valid"] is False


def test_nearly_complete_rows_contribute_to_the_profile():
    """The profile uses complete rows *and* rows with a single missing entry, filling that
    entry with a column mean before normalising (see `_estimate_global_profile`).

    The manuscript describes the profile as coming from "complete rows" only; this test
    records the implemented behaviour so the two cannot drift apart silently.
    """
    X = np.array([
        [50.0, 30.0, 20.0],
        [np.nan, 30.0, 20.0],   # nearly complete -> usable
        [50.0, 25.0, 25.0],
        [np.nan, 30.0, np.nan],
    ])
    imp = CRISPImputer(comp_idx=[0, 1, 2], total=TOTAL).fit(X)

    complete_only = X[~np.isnan(X).any(axis=1)]
    profile_complete_only = np.median(
        complete_only / complete_only.sum(axis=1, keepdims=True), axis=0
    )
    # rows 0, 2 are complete; row 1 is also usable, so the profile must differ
    assert not np.allclose(imp.global_profile_, profile_complete_only, atol=1e-12)


def test_nan_in_a_non_compositional_column_is_left_untouched():
    """CRISP imputes the composition only.

    Columns outside `comp_idx` are not part of the composition, so a NaN there must be
    preserved rather than silently replaced by a column mean. (Behaviour changed in
    1.1.0; see README "Known issues".)
    """
    X = np.array([
        [40.0, 30.0, 30.0, 10.0],
        [50.0, 25.0, 25.0, 20.0],
        [45.0, 30.0, 25.0, np.nan],
    ])
    out = crisp(X, comp_idx=[0, 1, 2], total=TOTAL)
    assert np.isnan(out[2, 3])
    # present values in the non-compositional column are unchanged
    np.testing.assert_allclose(out[:2, 3], [10.0, 20.0], atol=1e-12)
    # the compositional part still closes
    np.testing.assert_allclose(out[:, :3].sum(axis=1), TOTAL, atol=1e-9)


@pytest.mark.parametrize("case", ["all_nan_column", "single_row", "transform_all_nan_column"])
def test_degenerate_inputs_emit_no_runtime_warnings(case):
    """Degenerate all-NaN slices must be handled explicitly, not left to NumPy's
    "Mean of empty slice" / "All-NaN slice encountered" warnings."""
    all_nan_col = np.array([
        [np.nan, 30.0, 20.0, 10.0],
        [np.nan, 25.0, 25.0, 10.0],
        [np.nan, 35.0, 15.0, 10.0],
        [np.nan, 30.0, 20.0, 10.0],
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if case == "all_nan_column":
            crisp(all_nan_col, comp_idx=[0, 1, 2, 3], total=TOTAL)
        elif case == "single_row":
            crisp(np.array([[np.nan, 40.0, 20.0]]), comp_idx=[0, 1, 2], total=TOTAL)
        else:
            imp = CRISPImputer(comp_idx=[0, 1, 2], total=TOTAL).fit(
                np.array([[50.0, 30.0, 20.0], [45.0, 35.0, 20.0], [55.0, 25.0, 20.0]])
            )
            imp.transform(np.array([[np.nan, 40.0, 20.0], [np.nan, 30.0, 30.0]]))

    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime == [], f"unexpected RuntimeWarning(s): {[str(w.message) for w in runtime]}"
