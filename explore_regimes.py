# -*- coding: utf-8 -*-
"""explore_regimes.py — 找 CRISP 真正强于 MICE 的场景（方向 B）

场景 A: n < d 高维小样本（MICE 回归可能退化）
场景 B: 极高缺失率 50%/70%（信息少，CRISP 全局轮廓可能更稳）
对比：CRISP / MICE / KNN直接 / 均值填充（Python 方法，快速探测）
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, impute_crisp, impute_knn_direct, impute_mean
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


def impute_mice(Xm, n_iter=20):
    return IterativeImputer(max_iter=n_iter, random_state=0).fit_transform(Xm)


def gen_sparse(n, d, ncomp_range, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    for i in range(n):
        ncomp = int(rng.integers(*ncomp_range))
        cols = rng.choice(d, size=ncomp, replace=False)
        props = rng.dirichlet(np.ones(ncomp))
        X[i, cols] = props * 100.0
    return np.round(X / X.sum(axis=1, keepdims=True) * 100.0, 2)


def run_case(name, X, rate, mechanism, seed=0):
    miss = make_missing(X, rate, mechanism, seed=seed)
    Xm = X.copy(); Xm[miss] = np.nan
    rows = []
    for mname, fn in [("CRISP", impute_crisp), ("MICE", impute_mice),
                      ("KNN", impute_knn_direct), ("mean", impute_mean)]:
        Xp = fn(Xm)
        mae = float(np.mean(np.abs(X[miss] - Xp[miss])))
        sd = float(np.mean(np.abs(Xp.sum(axis=1) - 100.0)))
        rows.append({"method": mname, "mae": round(mae, 4), "sumdev": round(sd, 4)})
    print(f"--- {name} ---")
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


def main():
    print("=" * 50)
    print("场景 A: n < d 高维小样本")
    for (n, d, nc) in [(30, 40, (8, 15)), (40, 30, (8, 15)), (25, 50, (10, 20))]:
        X = gen_sparse(n, d, nc, seed=0)
        print(f"\n[n={n}, d={d}, 结构零={(X==0).mean()*100:.0f}%]")
        run_case(f"A n{d}x d{n}", X, 0.2, "MCAR", seed=3)

    print("\n" + "=" * 50)
    print("场景 B: 极高缺失率（用合成 + ge）")
    rng = np.random.default_rng(0)
    X = gen_sparse(150, 12, (4, 8), seed=0)
    print(f"\n[合成 n=150 d=12, 结构零 {(X==0).mean()*100:.0f}%]")
    for rate in [0.5, 0.7]:
        run_case(f"B rate={rate:.0%}", X, rate, "MCAR", seed=5)

    # ge 真实数据
    from benchmark_zeros import find_data
    path = find_data("ge_refractory_alloy.csv")
    if path:
        df = pd.read_csv(path, encoding="utf-8-sig")
        comps = [c for c in df.columns if c.endswith("(at%)")]
        df = df.dropna(subset=["Hardness (GPa)"]).reset_index(drop=True)
        Xg = df[comps].values.astype(float)
        Xg = Xg / Xg.sum(axis=1, keepdims=True) * 100.0
        print(f"\n[ge n={len(Xg)} d={len(comps)}, 结构零 {(Xg==0).mean()*100:.0f}%]")
        for rate in [0.5, 0.7]:
            run_case(f"B-ge rate={rate:.0%}", Xg, rate, "MCAR", seed=5)


if __name__ == "__main__":
    main()
