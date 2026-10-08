# -*- coding: utf-8 -*-
"""final_compare_v3.py — 审稿严谨性版 v3

修复：
  ① 20 随机缺失种子（Wilcoxon 双侧最小 p 可达 0.00195，可判显著）
  ④ 新增 SoftImpute（低秩矩阵补全）基线
  ⑧ Python 方法统一 20 种子 mean±std；R 基线单种子（如实标注）
输出：final_results_v3.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, impute_crisp, impute_knn_direct, impute_mean, find_data
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
MECHANISMS = ["MCAR", "MAR", "MNAR"]
RATES = [0.1, 0.2]
N_SEED = 20
PY = ["CRISP", "MICE", "KNN", "mean", "SoftImpute"]
R = ["impKNNa", "missForest"]


def impute_mice(Xm):
    return IterativeImputer(max_iter=20, random_state=0).fit_transform(Xm)


def soft_impute(Xm, rank=5, max_iter=100, tol=1e-5):
    """简化 SoftImpute（proximal gradient + SVD 低秩补全）。"""
    X = Xm.copy().astype(float)
    for j in range(X.shape[1]):
        med = np.nanmedian(X[:, j])
        X[np.isnan(X[:, j]), j] = med if not np.isnan(med) else 0.0
    mask = ~np.isnan(Xm)
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(rank, len(s))
        X_new = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
        X_new[mask] = Xm[mask]  # 恢复观测
        if np.linalg.norm(X_new - X) < tol:
            X = X_new
            break
        X = X_new
    return X


def gen_synthetic():
    rng = np.random.default_rng(0)
    n, d = 300, 8
    X = np.zeros((n, d))
    for i in range(n):
        nc = int(rng.integers(3, 7))
        cols = rng.choice(d, size=nc, replace=False)
        X[i, cols] = rng.dirichlet(np.ones(nc)) * 100.0
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


def main():
    datasets = {"synthetic": gen_synthetic(), "steel": load_real("steel"),
                "glass": load_real("glass"), "ge": load_real("ge"),
                "cement": load_cement()}
    rows, wil = [], []
    for dname, X in datasets.items():
        zero_frac = (X == 0).mean()
        print(f"=== {dname} (结构零 {zero_frac*100:.0f}%) ===")
        for mech in MECHANISMS:
            for rate in RATES:
                py_mae = {m: [] for m in PY}
                py_sd = {m: [] for m in PY}
                for s in range(N_SEED):
                    miss = make_missing(X, rate, mech, seed=s)
                    Xm = X.copy(); Xm[miss] = np.nan
                    preds = {"CRISP": impute_crisp(Xm), "MICE": impute_mice(Xm),
                             "KNN": impute_knn_direct(Xm), "mean": impute_mean(Xm),
                             "SoftImpute": soft_impute(Xm)}
                    for m, Xp in preds.items():
                        py_mae[m].append(float(np.mean(np.abs(X[miss] - Xp[miss]))))
                        py_sd[m].append(float(np.mean(np.abs(Xp.sum(axis=1) - 100.0))))
                for m in PY:
                    a, s = np.array(py_mae[m]), np.array(py_sd[m])
                    rows.append({"dataset": dname, "mechanism": mech, "missing": rate, "method": m,
                                 "mae_mean": round(a.mean(), 4), "mae_std": round(a.std(), 4),
                                 "sumdev": round(s.mean(), 4), "seeds": N_SEED})
                # R 单种子
                miss42 = make_missing(X, rate, mech, seed=42)
                Xm42 = X.copy(); Xm42[miss42] = np.nan
                for rm in R:
                    fp = os.path.join(TMP, f"{dname}_{mech}_{int(rate*100)}", f"{rm}.csv")
                    if os.path.exists(fp):
                        Xp = pd.read_csv(fp).values.astype(float)
                        rows.append({"dataset": dname, "mechanism": mech, "missing": rate, "method": rm,
                                     "mae_mean": round(float(np.mean(np.abs(X[miss42]-Xp[miss42]))), 4),
                                     "mae_std": float("nan"), "sumdev": round(float(np.mean(np.abs(Xp.sum(axis=1)-100))), 4), "seeds": 1})
                # Wilcoxon（20 种子配对，仅 Python 方法）
                for m in PY[1:]:
                    try:
                        p = wilcoxon(np.array(py_mae["CRISP"]), np.array(py_mae[m])).pvalue
                    except ValueError:
                        p = float("nan")
                    wil.append({"dataset": dname, "mechanism": mech, "missing": rate, "test": f"CRISP vs {m}",
                                "p": round(p, 4), "crisp": round(np.mean(py_mae["CRISP"]), 3),
                                "other": round(np.mean(py_mae[m]), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_results_v3.csv"),
              index=False, encoding="utf-8-sig")
    wdf = pd.DataFrame(wil)
    wdf.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wilcoxon_v3.csv"),
               index=False, encoding="utf-8-sig")

    # 关键行输出（MCAR 10%）
    print("\n=== MCAR 10%（20 种子 mean±std）===")
    sub = df[(df.mechanism == "MCAR") & (df.missing == 0.1) & (df.method.isin(["CRISP", "MICE", "SoftImpute", "impKNNa"]))]
    piv = sub.pivot_table(index="dataset", columns="method", values="mae_mean")
    print(piv.round(3).to_string())
    print("\n=== Wilcoxon CRISP vs MICE / SoftImpute（20 种子，MCAR 10%）===")
    wsub = wdf[(wdf.mechanism == "MCAR") & (wdf.missing == 0.1) & (wdf.test.isin(["CRISP vs MICE", "CRISP vs SoftImpute"]))]
    print(wsub.pivot_table(index="dataset", columns="test", values="p").round(4).to_string())


if __name__ == "__main__":
    main()
