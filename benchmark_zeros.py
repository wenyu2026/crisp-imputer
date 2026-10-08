# -*- coding: utf-8 -*-
"""benchmark_zeros.py — 结构零差异化实验（条件①）

对比 CRISP 与 ilr-KNN / KNN直接 / 均值填充，在"结构零 + 缺失"混合场景：
  - 缺失位置插补 MAE（插补质量）
  - 结构零保留（原 0 位置被插成多少）
  - 约束 SumDev
数据集：合成成分数据 + 真实 steel / glass / ge（合金）
输出：results_zeros.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crisp import crisp
from sklearn.impute import KNNImputer, SimpleImputer

RESULTS_ALL = []


# ============ ilr 变换（Helmert 正交基，可逆） ============
def helmert_ilr_basis(d):
    H = np.zeros((d, d))
    H[:, 0] = 1.0 / np.sqrt(d)
    for i in range(1, d):
        H[:i, i] = 1.0 / np.sqrt(i * (i + 1))
        H[i, i] = -np.sqrt(i / (i + 1.0))
    return H[:, 1:]

def ilr_transform(X, pseudocount=1e-6):
    X = np.maximum(np.asarray(X, dtype=float), pseudocount)
    B = helmert_ilr_basis(X.shape[1])
    logX = np.log(X)
    gm = np.exp(logX.mean(axis=1, keepdims=True))
    clr = logX - np.log(gm)
    return clr @ B

def ilr_inv_transform(Z, d):
    B = helmert_ilr_basis(d)
    x = np.exp(Z @ B.T)
    return x / x.sum(axis=1, keepdims=True)


# ============ 插补方法 ============
def impute_crisp(Xm):
    return crisp(Xm, comp_idx=list(range(Xm.shape[1])), total=100.0)

def impute_ilr_knn(Xm, n_neighbors=5):
    d = Xm.shape[1]
    X0 = Xm.copy()
    for j in range(d):
        med = np.nanmedian(X0[:, j])
        if np.isnan(med):
            med = 1.0
        X0[np.isnan(X0[:, j]), j] = med
    X0 = np.maximum(X0, 1e-6)
    Z = ilr_transform(X0, pseudocount=1e-6)
    complete = ~np.isnan(Xm).any(axis=1)
    full_idx = np.where(complete)[0]
    out_Z = Z.copy()
    for i in np.where(~complete)[0]:
        if len(full_idx) < 3:
            break
        known = ~np.isnan(Xm[i])
        dist = np.linalg.norm(Xm[complete][:, known] - Xm[i][known], axis=1)
        nb = full_idx[np.argsort(dist)[:n_neighbors]]
        out_Z[i] = np.mean(Z[nb], axis=0)
    Xrec = ilr_inv_transform(out_Z, d)
    return Xrec * 100.0

def impute_knn_direct(Xm, n_neighbors=5):
    return KNNImputer(n_neighbors=n_neighbors).fit_transform(Xm)

def impute_mean(Xm):
    return SimpleImputer(strategy="mean").fit_transform(Xm)

# 主基线方法（论文用）：CRISP vs 官方 impKNNa(R robCompositions) vs KNN直接 vs 均值填充。
# 官方 impKNNa 结果由 run_r_baseline.R 生成，在 final_compare.py 中合并。
# 注意：下方 impute_ilr_knn 为早期 Python 简化实现，已降级、不作为主基线，
# 避免"弱化对手"的不公平对比（论文若提及 ilr 类方法，必须以官方 robCompositions 为准）。
METHODS = [("CRISP", impute_crisp), ("KNN直接", impute_knn_direct), ("均值填充", impute_mean)]


def evaluate(X_true, X_pred, miss_mask, dataset, rate):
    missing_mae = float(np.mean(np.abs(X_true[miss_mask] - X_pred[miss_mask])))
    zero_mask = (X_true == 0) & ~miss_mask
    zero_err = float(np.mean(np.abs(X_pred[zero_mask]))) if zero_mask.any() else 0.0
    sum_dev = float(np.mean(np.abs(X_pred.sum(axis=1) - 100.0)))
    RESULTS_ALL.append({"dataset": dataset, "missing": rate, "method": "?",
                        "missing_mae": round(missing_mae, 4), "zero_err": round(zero_err, 4),
                        "sum_dev": round(sum_dev, 6)})


def make_missing(X, rate, mechanism, seed):
    """按缺失机制生成缺失掩码（只在非零/结构有值位置缺失）。

    MCAR: 完全随机；MAR: 缺失概率依赖其他观测组分（行内其他组分含量高 → 更易缺失）；
    MNAR: 缺失概率依赖缺失值本身（组分含量越低 → 越易缺失）。
    """
    rng = np.random.default_rng(seed)
    nonzero = X != 0
    n, d = X.shape
    miss = np.zeros(X.shape, dtype=bool)
    if mechanism == "MCAR":
        miss = (rng.random(X.shape) < rate) & nonzero
    elif mechanism == "MAR":
        for j in range(d):
            others = X.sum(axis=1) - X[:, j]
            p = rate * (0.4 + 0.6 * others / (others.max() + 1e-9))
            miss[:, j] = (rng.random(n) < p) & nonzero[:, j]
    elif mechanism == "MNAR":
        colmax = X.max(axis=0)
        for j in range(d):
            p = rate * (0.2 + 0.8 * (1.0 - X[:, j] / (colmax[j] + 1e-9)))
            miss[:, j] = (rng.random(n) < p) & nonzero[:, j]
    else:
        raise ValueError(mechanism)
    return miss


def run_compare(dataset, X, rate, mechanism, seed):
    """X: 真值成分矩阵（含结构零）。注入缺失后对比各方法。"""
    Xm = X.copy()
    miss = make_missing(X, rate, mechanism, seed)
    Xm[miss] = np.nan
    rows = []
    for name, fn in METHODS:
        Xp = fn(Xm)
        missing_mae = float(np.mean(np.abs(X[miss] - Xp[miss])))
        zero_mask = (X == 0) & ~miss
        zero_err = float(np.mean(np.abs(Xp[zero_mask]))) if zero_mask.any() else 0.0
        sum_dev = float(np.mean(np.abs(Xp.sum(axis=1) - 100.0)))
        RESULTS_ALL.append({"dataset": dataset, "mechanism": mechanism, "missing": rate,
                            "method": name, "missing_mae": round(missing_mae, 4),
                            "zero_err": round(zero_err, 4), "sum_dev": round(sum_dev, 6)})
        rows.append({"method": name, "missing_mae": round(missing_mae, 4),
                     "zero_err": round(zero_err, 4), "sum_dev": round(sum_dev, 6)})
    print(pd.DataFrame(rows).to_string(index=False))


def run_synthetic():
    print("=" * 60)
    print("合成成分数据（含结构零）：n=300, d=8, 每样本 3-6 组分")
    rng = np.random.default_rng(0)
    n, d = 300, 8
    X = np.zeros((n, d))
    for i in range(n):
        ncomp = int(rng.integers(3, 7))
        cols = rng.choice(d, size=ncomp, replace=False)
        props = rng.dirichlet(np.ones(ncomp))
        X[i, cols] = props * 100.0
    X = np.round(X / X.sum(axis=1, keepdims=True) * 100.0, 2)
    print(f"结构零占比: {(X == 0).mean()*100:.0f}%")
    for mech in ["MCAR", "MAR", "MNAR"]:
        for rate in [0.1, 0.2]:
            print(f"\n--- {mech} 缺失率 {rate:.0%} ---")
            run_compare("synthetic", X, rate, mech, seed=0)


def find_data(fname):
    """定位数据集。

    本仓库自带 data/ 目录（steel / glass / ge），优先使用，保证 clone 后即可复现。
    其余候选路径是为了兼容早期在上级目录组织数据的开发环境。
    """
    base = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(base, "data", fname),
        os.path.join(base, "..", "..", "项目一", "smallmatprep", "data", fname),
        os.path.join(base, "..", "项目一", "smallmatprep", "data", fname),
        os.path.join(base, "..", "..", "smallmatprep", "data", fname),
        os.path.join(base, "..", "data", fname),
    ]
    for c in cands:
        if os.path.exists(c):
            return os.path.normpath(c)
    return None


def run_real():
    datasets = [
        ("steel", "steel_strength.csv", ["c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"], "tensile strength"),
        ("glass", "glass_40samples.csv", ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"], "RI"),
        ("ge", "ge_refractory_alloy.csv", None, "Hardness (GPa)"),
    ]
    for dname, fname, comps, target in datasets:
        path = find_data(fname)
        if path is None:
            print(f"\n!! 未找到 {fname}，跳过")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if comps is None:
            comps = [c for c in df.columns if c.endswith("(at%)")]
        df = df.dropna(subset=[target]).reset_index(drop=True)
        X = df[comps].values.astype(float)
        X = X / X.sum(axis=1, keepdims=True) * 100.0
        X = X[:300]
        print(f"\n{'='*60}\n真实数据 {dname}: n={len(X)}, d={len(comps)}, 结构零={(X==0).mean()*100:.0f}%")
        for mech in ["MCAR", "MAR", "MNAR"]:
            for rate in [0.1, 0.2]:
                print(f"\n--- {mech} 缺失率 {rate:.0%} ---")
                run_compare(dname, X, rate, mech, seed=42)


if __name__ == "__main__":
    run_synthetic()
    run_real()
    df = pd.DataFrame(RESULTS_ALL)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_zeros.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n\n已保存 {out}  ({len(df)} 行)")
