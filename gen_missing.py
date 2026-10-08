# -*- coding: utf-8 -*-
"""gen_missing.py — 为 R 基线（impKNNa/impCoda）生成缺失矩阵 CSV。

对每个 (数据集, 缺失机制, 缺失率) 导出：
  tmp/{dataset}_{mech}_{rate}/X_true.csv      真值成分矩阵
  tmp/{dataset}_{mech}_{rate}/X_missing.csv  含 NaN 的缺失矩阵
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
MECHANISMS = ["MCAR", "MAR", "MNAR"]
RATES = [0.1, 0.2]


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


def find_data(fname):
    base = os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(base, "..", "..", "项目一", "smallmatprep", "data", fname),
              os.path.join(base, "..", "项目一", "smallmatprep", "data", fname),
              os.path.join(base, "..", "..", "smallmatprep", "data", fname)]:
        if os.path.exists(c):
            return os.path.normpath(c)
    return None


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
    X = X / X.sum(axis=1, keepdims=True) * 100.0
    return X[:300]


def load_cement():
    """UCI Concrete Compressive Strength：7 个配方组分 + Age，目标强度。"""
    import pandas as pd
    df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cement_concrete.xls"))
    X = df.iloc[:, :7].values.astype(float)  # Cement..Fine Aggregate（7 组分）
    return X / X.sum(axis=1, keepdims=True) * 100.0


def main():
    datasets = {"synthetic": gen_synthetic(), "steel": load_real("steel"),
                "glass": load_real("glass"), "ge": load_real("ge"),
                "cement": load_cement()}
    for dname, X in datasets.items():
        for mech in MECHANISMS:
            for rate in RATES:
                miss = make_missing(X, rate, mech, seed=42)
                Xm = X.copy()
                Xm[miss] = np.nan
                folder = os.path.join(TMP, f"{dname}_{mech}_{int(rate*100)}")
                os.makedirs(folder, exist_ok=True)
                pd.DataFrame(X).to_csv(os.path.join(folder, "X_true.csv"), index=False)
                pd.DataFrame(Xm).to_csv(os.path.join(folder, "X_missing.csv"), index=False)
    print(f"已生成 {len(datasets)*len(MECHANISMS)*len(RATES)} 组缺失矩阵到 {TMP}")


if __name__ == "__main__":
    main()
