# -*- coding: utf-8 -*-
"""supplement.py — 补充实验（审稿第 1/6/7 条）

① 单缺失组分占比（解释 n<d 子集 MAE=0.000 的机制）
⑥ profile 消融：median / geometric-mean / mean（CoDA 子组合一致性）
⑦ 高缺失率 50%/70%：CRISP vs MICE 精度
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, impute_crisp, impute_mean
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

BASE = os.path.dirname(os.path.abspath(__file__))


def impute_mice(Xm):
    return IterativeImputer(max_iter=20, random_state=0).fit_transform(Xm)


def profile_impute(Xm, profile_fn):
    """通用 profile 插补（与 CRISP 相同分配，但轮廓用不同聚合）。"""
    X = np.asarray(Xm, dtype=float)
    comp_idx = list(range(X.shape[1]))
    complete = ~np.isnan(X).any(axis=1)
    if complete.sum() < 1:
        return np.nan_to_num(X, nan=0.0)
    norm = X[complete][:, comp_idx]
    norm = np.atleast_2d(norm) / np.atleast_2d(norm).sum(axis=1, keepdims=True)
    profile = profile_fn(norm, axis=0)
    profile = np.maximum(np.atleast_1d(profile), 1e-6)  # 最小正数保护（同 CRISP）
    profile = profile / profile.sum()
    Xf = X.copy()
    for i in range(X.shape[0]):
        missing = np.isnan(X[i, comp_idx])
        if not missing.any():
            continue
        known = ~missing
        known_sum = np.nansum(X[i, comp_idx][known])
        R = 100.0 - known_sum
        miss_prof = profile[missing]
        if miss_prof.sum() > 0:
            alloc = (miss_prof / miss_prof.sum()) * R
            Xf[i, np.where(missing)[0]] = alloc
    return Xf


def geo_mean(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.exp(np.mean(np.log(np.maximum(x, 1e-9)), axis=axis))


# ---------- ① 单缺失组分占比（n<d 子集） ----------
def single_missing_frac():
    print("=" * 50)
    print("① 单缺失组分占比（解释 MAE=0.000）")
    for dname in ["glass", "cement", "steel"]:
        folder = os.path.join(BASE, "tmp_realnd", f"{dname}_s0")
        Xm = pd.read_csv(os.path.join(folder, "X_missing.csv")).values.astype(float)
        miss = np.isnan(Xm)
        row_miss = miss.sum(axis=1)
        n_miss_rows = (row_miss >= 1).sum()
        n_single = (row_miss == 1).sum()
        print(f"{dname}: 缺失行 {n_miss_rows}/{len(Xm)}, 单缺失组分行 {n_single}"
              f" ({n_single/max(n_miss_rows,1)*100:.0f}% of missing rows)")
    # 合成 n<d
    X = _gen_sparse(25, 50, (8, 15), 0)
    miss = make_missing(X, 0.1, "MCAR", seed=42)
    Xm = X.copy(); Xm[miss] = np.nan
    rm = np.isnan(Xm).sum(axis=1)
    print(f"synA1(MCAR10%): 单缺失组分行占缺失行 {(rm==1).sum()/max((rm>=1).sum(),1)*100:.0f}%")


def _gen_sparse(n, d, nc, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    for i in range(n):
        k = int(rng.integers(*nc))
        cols = rng.choice(d, size=k, replace=False)
        X[i, cols] = rng.dirichlet(np.ones(k)) * 100.0
    return X / X.sum(axis=1, keepdims=True) * 100.0


# ---------- ⑥ profile 消融 ----------
def ablation():
    print("\n" + "=" * 50)
    print("⑥ profile 消融：median / geometric-mean / mean（合成 n<d）")
    X = _gen_sparse(25, 50, (8, 15), 0)
    for rate in [0.1, 0.3]:
        maes = {"median": [], "geo_mean": [], "mean": []}
        for s in range(10):
            miss = make_missing(X, rate, "MCAR", seed=s)
            Xm = X.copy(); Xm[miss] = np.nan
            for name, fn in [("median", np.median), ("geo_mean", geo_mean), ("mean", np.mean)]:
                Xp = profile_impute(Xm, fn)
                maes[name].append(float(np.mean(np.abs(X[miss] - Xp[miss]))))
        print(f"缺失率 {rate:.0%}: " + " | ".join(f"{k}={np.mean(v):.3f}" for k, v in maes.items()))


# ---------- ⑦ 高缺失率 ----------
def high_missing():
    print("\n" + "=" * 50)
    print("⑦ 高缺失率 50%/70%：CRISP vs MICE（合成 n=150 d=12 + ge）")
    X = _gen_sparse(150, 12, (4, 8), 0)
    for rate in [0.5, 0.7]:
        for s in range(5):
            miss = make_missing(X, rate, "MCAR", seed=s)
            Xm = X.copy(); Xm[miss] = np.nan
            c = impute_crisp(Xm); m = impute_mice(Xm)
            mc = float(np.mean(np.abs(X[miss] - c[miss])))
            mm = float(np.mean(np.abs(X[miss] - m[miss])))
            print(f"syn {rate:.0%} s{s}: CRISP {mc:.3f} vs MICE {mm:.3f}")


if __name__ == "__main__":
    single_missing_frac()
    ablation()
    high_missing()
