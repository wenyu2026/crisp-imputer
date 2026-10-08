# -*- coding: utf-8 -*-
"""Unit tests for the benchmark harness.

Covers the pieces the released results depend on but that are not part of the
`crisp` library itself:

* `make_missing`            — missingness protocol (never touches structural zeros)
* `find_data`               — resolves the datasets bundled in `data/`
* `soft_impute`             — the reference SoftImpute implementation
* `soft_impute_lambda_sweep`— sweep helper
* honest_rerun helpers      — `closure_project`, `residual_backfill`

Run:  pytest -q
"""
import os
import sys

import numpy as np
import pytest

import honest_rerun as H
from benchmark_zeros import find_data, make_missing
from final_compare_v3 import soft_impute, soft_impute_lambda_sweep

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sparse_composition(rng, n=30, d=8, total=100.0):
    X = np.zeros((n, d))
    for i in range(n):
        k = int(rng.integers(3, min(6, d) + 1))
        cols = rng.choice(d, size=k, replace=False)
        X[i, cols] = rng.dirichlet(np.ones(k)) * total
    return X / X.sum(axis=1, keepdims=True) * total


# ---------------------------------------------------------------------------
# make_missing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mech", ["MCAR", "MAR", "MNAR"])
@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5])
def test_make_missing_never_marks_structural_zeros(mech, rate):
    """The whole benchmark protocol rests on this: a structural zero is never
    treated as a missing measurement."""
    rng = np.random.default_rng(0)
    X = sparse_composition(rng, n=40, d=8)
    miss = make_missing(X, rate, mech, seed=0)
    assert (miss.shape == X.shape)
    assert not (miss & (X == 0)).any()


@pytest.mark.parametrize("mech", ["MCAR", "MAR", "MNAR"])
def test_make_missing_is_deterministic(mech):
    rng = np.random.default_rng(1)
    X = sparse_composition(rng, n=30, d=6)
    a = make_missing(X, 0.3, mech, seed=7)
    b = make_missing(X, 0.3, mech, seed=7)
    np.testing.assert_array_equal(a, b)


def test_make_missing_rate_is_applied_to_nonzero_entries():
    rng = np.random.default_rng(2)
    X = sparse_composition(rng, n=200, d=8)
    nonzero = int((X != 0).sum())
    for rate in (0.1, 0.3):
        got = make_missing(X, rate, "MCAR", seed=0).sum()
        assert abs(got / nonzero - rate) < 0.05, (rate, got / nonzero)


def test_make_missing_rejects_unknown_mechanism():
    X = np.ones((4, 3))
    with pytest.raises(ValueError):
        make_missing(X, 0.1, "NOT_A_MECHANISM", seed=0)


# ---------------------------------------------------------------------------
# find_data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", ["steel_strength.csv", "glass_40samples.csv",
                                   "ge_refractory_alloy.csv"])
def test_find_data_resolves_bundled_datasets(fname):
    p = find_data(fname)
    assert p is not None, f"{fname} not found — the repo should be self-contained"
    assert os.path.exists(p)
    assert os.path.normpath(p).startswith(os.path.normpath(BASE))


def test_find_data_returns_none_for_unknown_file():
    assert find_data("definitely_not_a_dataset_xyz.csv") is None


# ---------------------------------------------------------------------------
# soft_impute (reference implementation)
# ---------------------------------------------------------------------------

def test_soft_impute_preserves_shape_and_finiteness():
    rng = np.random.default_rng(3)
    X = sparse_composition(rng, n=25, d=50)
    miss = make_missing(X, 0.2, "MCAR", seed=0)
    Xm = X.copy()
    Xm[miss] = np.nan
    Xp = soft_impute(Xm)
    assert Xp.shape == Xm.shape
    assert np.isfinite(Xp).all()


def test_soft_impute_is_deterministic():
    rng = np.random.default_rng(4)
    X = sparse_composition(rng, n=30, d=10)
    miss = make_missing(X, 0.2, "MCAR", seed=0)
    Xm = X.copy()
    Xm[miss] = np.nan
    np.testing.assert_array_equal(soft_impute(Xm), soft_impute(Xm))


def test_soft_impute_recovers_a_low_rank_matrix_better_than_column_means():
    """Soft-imputation on a genuinely low-rank matrix must beat mean imputation."""
    rng = np.random.default_rng(5)
    u = rng.normal(size=(60, 1))
    v = rng.normal(size=(1, 15))
    M = u @ v + 50.0
    mask = rng.random(M.shape) < 0.15
    Mobs = M.copy()
    Mobs[mask] = np.nan

    Xp = soft_impute(Mobs, lam_ratio=0.01)
    mae = float(np.mean(np.abs(M[mask] - Xp[mask])))
    col_means = np.nanmean(Mobs, axis=0)
    mean_mae = float(np.mean(np.abs(M[mask] - col_means[np.where(mask)[1]])))
    assert mae < mean_mae


def test_soft_impute_with_zero_lambda_degenerates_to_column_mean():
    """lam_ratio=0 disables the soft threshold, so the unobserved entries stay at
    their column-mean initialisation. Documented behaviour, not a bug."""
    rng = np.random.default_rng(6)
    X = sparse_composition(rng, n=30, d=8)
    miss = make_missing(X, 0.2, "MCAR", seed=0)
    Xm = X.copy()
    Xm[miss] = np.nan

    Xp = soft_impute(Xm, lam_ratio=0.0)
    col_means = np.nanmean(Xm, axis=0)
    np.testing.assert_allclose(Xp[miss], col_means[np.where(miss)[1]], atol=1e-8)


def test_soft_impute_honours_centering_flag():
    rng = np.random.default_rng(7)
    X = sparse_composition(rng, n=30, d=8) + 1000.0     # heavily shifted columns
    miss = make_missing(X - 1000.0, 0.2, "MCAR", seed=0)
    Xm = X.copy()
    Xm[miss] = np.nan
    centered = soft_impute(Xm, lam_ratio=0.02, center=True)
    raw = soft_impute(Xm, lam_ratio=0.02, center=False)
    assert not np.allclose(centered, raw)


def test_soft_impute_handles_a_fully_missing_column():
    X = np.array([
        [np.nan, 30.0, 20.0, 10.0],
        [np.nan, 25.0, 25.0, 10.0],
        [np.nan, 35.0, 15.0, 10.0],
        [np.nan, 30.0, 20.0, 10.0],
    ])
    Xp = soft_impute(X)
    assert np.isfinite(Xp).all()


def test_soft_impute_lambda_sweep_keys_and_shapes():
    rng = np.random.default_rng(8)
    X = sparse_composition(rng, n=20, d=6)
    miss = make_missing(X, 0.2, "MCAR", seed=0)
    Xm = X.copy()
    Xm[miss] = np.nan
    ratios = (0.0, 0.05, 0.5)
    out = soft_impute_lambda_sweep(Xm, ratios=ratios)
    assert set(out.keys()) == set(float(r) for r in ratios)
    for Xp in out.values():
        assert Xp.shape == Xm.shape and np.isfinite(Xp).all()


# ---------------------------------------------------------------------------
# honest_rerun fair-comparison helpers
# ---------------------------------------------------------------------------

def test_closure_project_makes_every_row_sum_to_total():
    X = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    out = H.closure_project(X)
    np.testing.assert_allclose(out.sum(axis=1), 100.0, atol=1e-9)
    assert (out >= 0).all()


def test_closure_project_leaves_an_all_zero_row_at_zero():
    """A zero row has no mass to renormalise; it cannot be made to sum to 100
    without inventing observations, so it stays zero."""
    out = H.closure_project(np.array([[0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.0, 0.0, 0.0]], atol=1e-12)


def test_residual_backfill_solves_single_missing_rows_exactly():
    Xm = np.array([
        [10.0, 20.0, np.nan],
        [np.nan, 5.0, 5.0],
        [1.0, 2.0, 3.0],
    ])
    base = np.full_like(Xm, 99.0)          # a deliberately wrong baseline prediction
    out = H.residual_backfill(base, Xm)
    assert out[0, 2] == pytest.approx(70.0, abs=1e-9)
    assert out[1, 0] == pytest.approx(90.0, abs=1e-9)
    np.testing.assert_allclose(out.sum(axis=1), 100.0, atol=1e-9)


def test_residual_backfill_leaves_multi_missing_rows_to_the_baseline():
    Xm = np.array([[np.nan, np.nan, 50.0],
                   [10.0, 20.0, 30.0]])
    base = np.array([[3.0, 6.0, 50.0],
                     [10.0, 20.0, 30.0]])
    out = H.residual_backfill(base, Xm)
    # row 0 has two missing entries, so it is untouched apart from being closed:
    # the baseline's [3, 6, 50] is scaled by 100/59.
    np.testing.assert_allclose(out[0], np.array([3.0, 6.0, 50.0]) / 59.0 * 100.0, atol=1e-9)


def test_variants_of_returns_the_three_documented_arms():
    Xm = np.array([[10.0, 20.0, np.nan], [1.0, 2.0, 3.0]])
    base = np.full_like(Xm, 5.0)
    out = H.variants_of(base, Xm)
    assert set(out.keys()) == {"raw", "+closure", "+backfill"}
