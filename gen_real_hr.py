# -*- coding: utf-8 -*-
"""gen_real_hr.py — 真实材料数据的**高比值** n<d 子集

背景：原有 "真实 n<d" 子集（tmp_realnd/）的 d/n 只有 1.125–1.333，而 ge_nd 那个
所谓"真实高维 n<d"数据集实际是 21 列恒零填充（有效 d=9、n=20，真实 d/n=0.45，
根本不是 n<d）。本脚本用**真实数据**构造比值明显更高的 n<d 子集：

    steel  d=13, n=8  (d/n=1.63)
    ge     d=9,  n=6  (d/n=1.50)
    glass  d=8,  n=5  (d/n=1.60)
    cement d=7,  n=4  (d/n=1.75)

缺失率取 20% / 40%（只注入非零位），使**多缺失行**占主导——否则单缺失行会被
闭合约束一行算术直接解出，无法区分方法。

输出：tmp_realhr/{ds}_{mech}_{rate}_s{seed}/{X_true.csv, X_missing.csv}
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, find_data  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(BASE, "tmp_realhr")
MECHANISMS = ["MCAR", "MNAR"]
RATES = [0.2, 0.4]
N_SEED = 20


def load(name):
    specs = {
        "steel": ("steel_strength.csv",
                  ["c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"],
                  "tensile strength"),
        "glass": ("glass_40samples.csv", ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"], "RI"),
        "ge": ("ge_refractory_alloy.csv", None, "Hardness (GPa)"),
    }
    fname, comps, target = specs[name]
    df = pd.read_csv(find_data(fname), encoding="utf-8-sig")
    if comps is None:
        comps = [c for c in df.columns if c.endswith("(at%)")]
    df = df.dropna(subset=[target]).reset_index(drop=True)
    X = df[comps].values.astype(float)
    return X / X.sum(axis=1, keepdims=True) * 100.0


def load_cement():
    df = pd.read_excel(os.path.join(BASE, "cement_concrete.xls"))
    X = df.iloc[:, :7].values.astype(float)
    return X / X.sum(axis=1, keepdims=True) * 100.0


def main():
    datasets = {
        "steel": (load("steel"), 8),
        "ge": (load("ge"), 6),
        "glass": (load("glass"), 5),
        "cement": (load_cement(), 4),
    }
    summary = []
    for dname, (X, n_sub) in datasets.items():
        d = X.shape[1]
        print(f"{dname:<8} 原 n={len(X):<4} d={d:<3} -> 子集 n={n_sub}  d/n={d/n_sub:.2f}  结构零={(X==0).mean()*100:.0f}%")
        for mech in MECHANISMS:
            for rate in RATES:
                for s in range(N_SEED):
                    rng = np.random.default_rng(100 + s)
                    idx = rng.choice(len(X), size=n_sub, replace=False)
                    Xs = X[idx]
                    miss = make_missing(Xs, rate, mech, seed=1000 + s)
                    # 保证每列至少保留一个观测值：整列缺失是病态配置，而 sklearn 的
                    # KNNImputer / SimpleImputer 会直接删除该列，导致输出列数少于输入、
                    # 后续按掩码取值时索引错位。这里把该列缺失数最少的一格还原为观测。
                    for j in np.where(miss.all(axis=0))[0]:
                        cand = np.where(Xs[:, j] != 0)[0]
                        if cand.size:
                            miss[cand[0], j] = False
                    if miss.sum() == 0:
                        continue
                    Xm = Xs.copy()
                    Xm[miss] = np.nan
                    folder = os.path.join(TMP, f"{dname}_{mech}_{int(rate*100)}_s{s}")
                    os.makedirs(folder, exist_ok=True)
                    pd.DataFrame(Xs).to_csv(os.path.join(folder, "X_true.csv"), index=False)
                    pd.DataFrame(Xm).to_csv(os.path.join(folder, "X_missing.csv"), index=False)
                    per_row = miss.sum(axis=1)
                    summary.append({
                        "dataset": dname, "n": n_sub, "d": d, "ratio": d / n_sub,
                        "mechanism": mech, "rate": rate, "seed": s,
                        "n_missing": int(miss.sum()),
                        "single_missing_rows": int((per_row == 1).sum()),
                        "multi_missing_rows": int((per_row > 1).sum()),
                        "complete_rows": int((per_row == 0).sum()),
                    })
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(BASE, "realhr_design.csv"), index=False)
    print(f"\n共生成 {len(sm)} 个子集 -> {TMP}")
    print("\n每个数据集的规模与缺失结构（均值）：")
    agg = sm.groupby("dataset").agg(
        n=("n", "first"), d=("d", "first"), d_over_n=("ratio", "first"),
        缺失格=("n_missing", "mean"), 单缺失行=("single_missing_rows", "mean"),
        多缺失行=("multi_missing_rows", "mean"), 完整行=("complete_rows", "mean"))
    print(agg.round(2).to_string())
    print("\n其中单缺失行占比（越低越好，越低说明越不依赖闭合恒等式）：")
    agg["单缺失行占比"] = (agg.单缺失行 / agg.缺失格 * 100).round(1)
    print(agg[["单缺失行占比"]].to_string())


if __name__ == "__main__":
    main()
