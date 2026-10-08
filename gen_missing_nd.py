# -*- coding: utf-8 -*-
"""gen_missing_nd.py — 为 n<d 场景生成缺失矩阵 CSV（供 R 官方基线跑）。

数据集：
  A1: 合成 n=25 d=50（结构零 71%）   A2: n=30 d=40
  A3: 合成 n=40 d=30               A4: n=60 d=25
  ge-nd: ge 扩维到 d=20（真实体系，n=86, n<d）
缺失：MCAR/MAR/MNAR × 10%/30%（n<d 下常规缺失率）
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, find_data

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_nd")
MECHANISMS = ["MCAR", "MAR", "MNAR"]
RATES = [0.1, 0.3]


def gen_sparse(n, d, ncomp_range, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    for i in range(n):
        ncomp = int(rng.integers(*ncomp_range))
        cols = rng.choice(d, size=ncomp, replace=False)
        props = rng.dirichlet(np.ones(ncomp))
        X[i, cols] = props * 100.0
    return np.round(X / X.sum(axis=1, keepdims=True) * 100.0, 2)


def ge_nd():
    """ge 前 20 个样本，扩维到 d=30（加 21 个零列）→ n=20 < d=30，保留真实组分。"""
    path = find_data("ge_refractory_alloy.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    comps = [c for c in df.columns if c.endswith("(at%)")]
    df = df.dropna(subset=["Hardness (GPa)"]).reset_index(drop=True)
    X = df[comps].values.astype(float)[:20]
    X = X / X.sum(axis=1, keepdims=True) * 100.0
    extra = np.zeros((len(X), 21))  # 扩维到 20+21=30
    return np.hstack([X, extra])


def main():
    datasets = {
        "synA1": gen_sparse(25, 50, (8, 15), 0),
        "synA2": gen_sparse(30, 40, (8, 15), 0),
        "synA3": gen_sparse(20, 40, (8, 15), 0),
        "synA4": gen_sparse(25, 60, (10, 20), 0),
        "ge_nd": ge_nd(),
    }
    for dname, X in datasets.items():
        n, d = X.shape
        zf = (X == 0).mean() * 100
        print(f"{dname}: n={n} d={d} n<d={n<d} 结构零={zf:.0f}%")
        for mech in MECHANISMS:
            for rate in RATES:
                miss = make_missing(X, rate, mech, seed=42)
                Xm = X.copy(); Xm[miss] = np.nan
                folder = os.path.join(TMP, f"{dname}_{mech}_{int(rate*100)}")
                os.makedirs(folder, exist_ok=True)
                pd.DataFrame(X).to_csv(os.path.join(folder, "X_true.csv"), index=False)
                pd.DataFrame(Xm).to_csv(os.path.join(folder, "X_missing.csv"), index=False)
    print(f"\n已生成到 {TMP}")


if __name__ == "__main__":
    main()
