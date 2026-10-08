# -*- coding: utf-8 -*-
"""final_compare.py — 最终对比表（学术诚信版本）

主基线（论文用）：
  CRISP（本项目） vs impKNNa（官方 R robCompositions） vs KNN直接 vs 均值填充
早期 Python 简化 ilr-KNN 已降级、不纳入主表（避免不公平对比）。

输出：final_results.csv（4 数据集 × 3 机制 × 2 缺失率 × 4 方法）
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import (make_missing, impute_crisp,
                             impute_knn_direct, impute_mean, find_data)

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
MECHANISMS = ["MCAR", "MAR", "MNAR"]
RATES = [0.1, 0.2]
METHODS = ["CRISP", "impKNNa", "missForest", "lrEM", "KNN直接", "均值填充"]


def gen_synthetic():
    rng = np.random.default_rng(0)
    n, d = 300, 8
    X = np.zeros((n, d))
    for i in range(n):
        ncomp = int(rng.integers(3, 7))
        cols = rng.choice(d, size=ncomp, replace=False)
        props = rng.dirichlet(np.ones(ncomp))
        X[i, cols] = props * 100.0
    return np.round(X / X.sum(axis=1, keepdims=True) * 100.0, 2)


def load_real(name):
    specs = {
        "steel": ("steel_strength.csv", ["c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"], "tensile strength"),
        "glass": ("glass_40samples.csv", ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"], "RI"),
        "ge": ("ge_refractory_alloy.csv", None, "Hardness (GPa)"),
    }
    fname, comps, target = specs[name]
    path = find_data(fname)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if comps is None:
        comps = [c for c in df.columns if c.endswith("(at%)")]
    df = df.dropna(subset=[target]).reset_index(drop=True)
    X = df[comps].values.astype(float)
    return X[:300] / X[:300].sum(axis=1, keepdims=True) * 100.0


def evaluate(X_true, Xp, miss):
    minc = min(Xp.shape[1], X_true.shape[1])
    missing_mae = float(np.mean(np.abs(X_true[:, :minc][miss[:, :minc]] - Xp[:, :minc][miss[:, :minc]])))
    zero_mask = (X_true == 0) & ~miss
    zero_err = float(np.mean(np.abs(Xp[:, :minc][zero_mask[:, :minc]]))) if zero_mask.any() else 0.0
    sum_dev = float(np.mean(np.abs(Xp.sum(axis=1) - 100.0)))
    return missing_mae, zero_err, sum_dev


def load_cement():
    """UCI Concrete Compressive Strength：7 个配方组分，归一化到总和 100。"""
    df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cement_concrete.xls"))
    X = df.iloc[:, :7].values.astype(float)
    return X / X.sum(axis=1, keepdims=True) * 100.0


def main():
    datasets = {"synthetic": gen_synthetic(), "steel": load_real("steel"),
                "glass": load_real("glass"), "ge": load_real("ge"),
                "cement": load_cement()}
    rows = []
    for dname, X in datasets.items():
        zero_frac = (X == 0).mean()
        print(f"=== {dname} (结构零 {zero_frac*100:.0f}%) ===")
        for mech in MECHANISMS:
            for rate in RATES:
                miss = make_missing(X, rate, mech, seed=42)
                Xm = X.copy(); Xm[miss] = np.nan
                # Python 方法
                results = {
                    "CRISP": impute_crisp(Xm),
                    "KNN直接": impute_knn_direct(Xm),
                    "均值填充": impute_mean(Xm),
                }
                # R 基线结果（impKNNa / missForest / lrEM）
                for rmethod in ["impKNNa", "missForest", "lrEM"]:
                    fp = os.path.join(TMP, f"{dname}_{mech}_{int(rate*100)}", f"{rmethod}.csv")
                    if os.path.exists(fp):
                        results[rmethod] = pd.read_csv(fp).values.astype(float)
                for method in METHODS:
                    if method not in results:
                        continue
                    mae, zerr, sdev = evaluate(X, results[method], miss)
                    rows.append({"dataset": dname, "mechanism": mech, "missing": rate,
                                 "method": method, "missing_mae": round(mae, 4),
                                 "zero_err": round(zerr, 4), "sum_dev": round(sdev, 6)})
        # 打印该数据集摘要（MCAR 10% 行）
        sub = pd.DataFrame(rows)
        sub = sub[(sub.dataset == dname) & (sub.mechanism == "MCAR") & (sub.missing == 0.1)]
        if len(sub):
            print(sub[["method", "missing_mae", "zero_err", "sum_dev"]].to_string(index=False))
            print()
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_results.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已保存 {out} ({len(df)} 行)")

    # 汇总：CRISP vs impKNNa 的优势倍数
    print("\n=== CRISP vs 官方 impKNNa：插补 MAE 倍数 ===")
    for dname in datasets:
        for mech in ["MCAR", "MAR", "MNAR"]:
            for rate in [0.1, 0.2]:
                c = df[(df.dataset == dname) & (df.mechanism == mech) & (df.missing == rate) & (df.method == "CRISP")]
                k = df[(df.dataset == dname) & (df.mechanism == mech) & (df.missing == rate) & (df.method == "impKNNa")]
                if len(c) and len(k):
                    print(f"{dname:10s} {mech:5s} {rate:.0%}: CRISP {c.iloc[0].missing_mae:.3f} vs impKNNa {k.iloc[0].missing_mae:.3f} -> {k.iloc[0].missing_mae/max(c.iloc[0].missing_mae,1e-9):.1f}x")


if __name__ == "__main__":
    main()
