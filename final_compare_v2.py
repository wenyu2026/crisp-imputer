# -*- coding: utf-8 -*-
"""final_compare_v2.py — 多种子统计 + MICE + Wilcoxon（审稿严谨性版）

- Python 方法（CRISP / KNN直接 / 均值填充 / MICE）：5 随机缺失种子，报告 mean±std
- R 基线（impKNNa / missForest）：单种子（seed=42），如实标注
- Wilcoxon 符号秩检验：CRISP vs 各 Python 基线（5 种子配对）
输出：final_results_v2.csv（完整表）+ 汇总表
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, impute_crisp, impute_knn_direct, impute_mean, find_data

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
MECHANISMS = ["MCAR", "MAR", "MNAR"]
RATES = [0.1, 0.2]
N_SEED = 5
PY_METHODS = ["CRISP", "KNN直接", "均值填充", "MICE"]
R_METHODS = ["impKNNa", "missForest"]


def impute_mice(Xm, n_iter=20):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    return IterativeImputer(max_iter=n_iter, random_state=0).fit_transform(Xm)


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


def load_cement():
    df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cement_concrete.xls"))
    X = df.iloc[:, :7].values.astype(float)
    return X / X.sum(axis=1, keepdims=True) * 100.0


def evals(X_true, Xp, miss):
    minc = min(Xp.shape[1], X_true.shape[1])
    mae = float(np.mean(np.abs(X_true[:, :minc][miss[:, :minc]] - Xp[:, :minc][miss[:, :minc]])))
    zm = (X_true == 0) & ~miss
    zerr = float(np.mean(np.abs(Xp[:, :minc][zm[:, :minc]]))) if zm.any() else 0.0
    sd = float(np.mean(np.abs(Xp.sum(axis=1) - 100.0)))
    return mae, zerr, sd


def main():
    datasets = {"synthetic": gen_synthetic(), "steel": load_real("steel"),
                "glass": load_real("glass"), "ge": load_real("ge"),
                "cement": load_cement()}
    rows, summary = [], []

    for dname, X in datasets.items():
        zero_frac = (X == 0).mean()
        print(f"=== {dname} (结构零 {zero_frac*100:.0f}%) ===")
        for mech in MECHANISMS:
            for rate in RATES:
                # Python 方法：5 种子
                py_mae = {m: [] for m in PY_METHODS}
                py_sumdev = {m: [] for m in PY_METHODS}
                for s in range(N_SEED):
                    miss = make_missing(X, rate, mech, seed=s)
                    Xm = X.copy(); Xm[miss] = np.nan
                    preds = {
                        "CRISP": impute_crisp(Xm),
                        "KNN直接": impute_knn_direct(Xm),
                        "均值填充": impute_mean(Xm),
                        "MICE": impute_mice(Xm),
                    }
                    for m, Xp in preds.items():
                        mae, _, sd = evals(X, Xp, miss)
                        py_mae[m].append(mae); py_sumdev[m].append(sd)
                # R 基线：单种子（seed=42）
                r_results = {}
                for rm in R_METHODS:
                    fp = os.path.join(TMP, f"{dname}_{mech}_{int(rate*100)}", f"{rm}.csv")
                    if os.path.exists(fp):
                        miss42 = make_missing(X, rate, mech, seed=42)
                        Xm42 = X.copy(); Xm42[miss42] = np.nan
                        mae, _, sd = evals(X, pd.read_csv(fp).values.astype(float), miss42)
                        r_results[rm] = (mae, sd)

                # 汇总行
                for m in PY_METHODS:
                    a = np.array(py_mae[m])
                    rows.append({"dataset": dname, "mechanism": mech, "missing": rate, "method": m,
                                 "mae_mean": round(a.mean(), 4), "mae_std": round(a.std(), 4),
                                 "sumdev_mean": round(np.mean(py_sumdev[m]), 4), "seeds": N_SEED})
                for rm in R_METHODS:
                    if rm in r_results:
                        mae, sd = r_results[rm]
                        rows.append({"dataset": dname, "mechanism": mech, "missing": rate, "method": rm,
                                     "mae_mean": round(mae, 4), "mae_std": float("nan"),
                                     "sumdev_mean": round(sd, 4), "seeds": 1})

                # Wilcoxon：CRISP vs Python 基线
                w = {"dataset": dname, "mechanism": mech, "missing": rate, "method": "CRISP",
                     "mae_mean": round(np.mean(py_mae["CRISP"]), 4),
                     "mae_std": round(np.std(py_mae["CRISP"]), 4),
                     "sumdev_mean": round(np.mean(py_sumdev["CRISP"]), 4), "seeds": N_SEED}
                summary.append(w)
                for m in PY_METHODS[1:]:
                    try:
                        p = wilcoxon(np.array(py_mae["CRISP"]), np.array(py_mae[m])).pvalue
                    except ValueError:
                        p = float("nan")
                    summary.append({"dataset": dname, "mechanism": mech, "missing": rate,
                                    "method": f"CRISP vs {m}", "p_value": round(p, 4),
                                    "crisp_mean": round(np.mean(py_mae["CRISP"]), 4),
                                    f"{m}_mean": round(np.mean(py_mae[m]), 4)})

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_results_v2.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n完整表已保存 {out} ({len(df)} 行)")

    print("\n=== 关键行（MCAR 10%，CRISP 5 种子均值±std）===")
    sub = df[(df.method == "CRISP") & (df.mechanism == "MCAR") & (df.missing == 0.1)]
    print(sub[["dataset", "mae_mean", "mae_std"]].to_string(index=False))

    print("\n=== CRISP vs 基线优势（MCAR 10%，均值）===")
    for dname in datasets:
        c = df[(df.dataset == dname) & (df.mechanism == "MCAR") & (df.missing == 0.1) & (df.method == "CRISP")]
        k = df[(df.dataset == dname) & (df.mechanism == "MCAR") & (df.missing == 0.1) & (df.method == "impKNNa")]
        if len(c) and len(k):
            print(f"{dname:10s} CRISP {c.iloc[0].mae_mean:.3f}±{c.iloc[0].mae_std:.3f} vs impKNNa {k.iloc[0].mae_mean:.3f} -> {k.iloc[0].mae_mean/max(c.iloc[0].mae_mean,1e-9):.1f}x")


if __name__ == "__main__":
    main()
