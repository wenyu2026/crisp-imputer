# -*- coding: utf-8 -*-
"""gen_real_nd.py — 真实材料数据的 n<d 子集（方向①）

从 4 个真实数据集抽 n<d 小样本子集（保留真实成分结构）：
  steel n=10 d=13 | ge n=8 d=9 | glass n=6 d=8 | cement n=6 d=7
每种 5 个随机子集（子集 seed 0-4），各注入 MCAR 10% 缺失（缺失 seed 42）。
输出：tmp_realnd/{ds}_s{seed}/
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, find_data

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_realnd")


def load(name):
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
    return X / X.sum(axis=1, keepdims=True) * 100.0


def load_cement():
    df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cement_concrete.xls"))
    X = df.iloc[:, :7].values.astype(float)
    return X / X.sum(axis=1, keepdims=True) * 100.0


def main():
    datasets = {"steel": (load("steel"), 10), "ge": (load("ge"), 8),
                "glass": (load("glass"), 6), "cement": (load_cement(), 6)}
    for dname, (X, n_sub) in datasets.items():
        d = X.shape[1]
        print(f"{dname}: 原 n={len(X)} d={d} -> 子集 n={n_sub} (n<d={n_sub<d})")
        for s in range(20):
            rng = np.random.default_rng(100 + s)
            idx = rng.choice(len(X), size=n_sub, replace=False)
            Xs = X[idx]
            miss = make_missing(Xs, 0.1, "MCAR", seed=42)
            Xm = Xs.copy(); Xm[miss] = np.nan
            folder = os.path.join(TMP, f"{dname}_s{s}")
            os.makedirs(folder, exist_ok=True)
            pd.DataFrame(Xs).to_csv(os.path.join(folder, "X_true.csv"), index=False)
            pd.DataFrame(Xm).to_csv(os.path.join(folder, "X_missing.csv"), index=False)
    print(f"\n已生成到 {TMP}")


if __name__ == "__main__":
    main()
