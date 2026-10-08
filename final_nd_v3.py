# -*- coding: utf-8 -*-
"""final_nd_v3.py — n<d 场景 20 种子 + SoftImpute（审稿严谨版）

输出：final_results_nd_v3.csv + wilcoxon_nd_v3.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, impute_crisp, impute_knn_direct, impute_mean
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from final_compare_v3 import soft_impute, impute_mice

BASE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(BASE, "tmp_nd")
N_SEED = 20
PY = ["CRISP", "MICE", "KNN", "mean", "SoftImpute"]
R = ["impKNNa", "missForest"]


def parse(folder):
    parts = folder.rsplit("_", 2)
    return parts[0], parts[1], float(parts[2]) / 100.0


def main():
    rows, wil = [], []
    for folder in sorted(os.listdir(TMP)):
        d = os.path.join(TMP, folder)
        if not os.path.isdir(d):
            continue
        dname, mech, rate = parse(folder)
        X = pd.read_csv(os.path.join(d, "X_true.csv")).values.astype(float)
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
            a, sd = np.array(py_mae[m]), np.array(py_sd[m])
            rows.append({"dataset": folder, "method": m, "mae": round(a.mean(), 4),
                         "mae_std": round(a.std(), 4), "sumdev": round(sd.mean(), 4), "seeds": N_SEED})
        for rm in R:
            fp = os.path.join(d, f"{rm}.csv")
            if os.path.exists(fp):
                miss42 = make_missing(X, rate, mech, seed=42)
                Xp = pd.read_csv(fp).values.astype(float)
                rows.append({"dataset": folder, "method": rm, "mae": round(float(np.mean(np.abs(X[miss42]-Xp[miss42]))), 4),
                             "mae_std": float("nan"), "sumdev": round(float(np.mean(np.abs(Xp.sum(axis=1)-100))), 4), "seeds": 1})
        for m in ["MICE", "SoftImpute"]:
            try:
                p = wilcoxon(np.array(py_mae["CRISP"]), np.array(py_mae[m])).pvalue
            except ValueError:
                p = float("nan")
            wil.append({"dataset": folder, "test": f"CRISP vs {m}", "p": round(p, 4),
                        "crisp": round(np.mean(py_mae["CRISP"]), 3), "other": round(np.mean(py_mae[m]), 3)})
        print(f"{folder}: CRISP {np.mean(py_mae['CRISP']):.3f}±{np.std(py_mae['CRISP']):.3f}"
              f" | MICE {np.mean(py_mae['MICE']):.3f} | SoftImpute {np.mean(py_mae['SoftImpute']):.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(BASE, "final_results_nd_v3.csv"), index=False, encoding="utf-8-sig")
    wdf = pd.DataFrame(wil)
    wdf.to_csv(os.path.join(BASE, "wilcoxon_nd_v3.csv"), index=False, encoding="utf-8-sig")
    print("\n=== Wilcoxon（20 种子，CRISP vs MICE / SoftImpute）===")
    print(wdf[wdf.test == "CRISP vs MICE"].to_string(index=False))
    print()
    print(wdf[wdf.test == "CRISP vs SoftImpute"].to_string(index=False))


if __name__ == "__main__":
    main()
