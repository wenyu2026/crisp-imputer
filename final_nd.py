# -*- coding: utf-8 -*-
"""final_nd.py — n<d 场景正式评估（方向 B）

Python 方法（CRISP/MICE/KNN/mean）5 种子 mean±std + Wilcoxon；
R 基线（impKNNa/missForest）单种子（seed=42）。
输出：final_results_nd.csv
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

BASE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(BASE, "tmp_nd")
N_SEED = 5
PY = ["CRISP", "MICE", "KNN", "mean"]
R = ["impKNNa", "missForest"]


def impute_mice(Xm):
    return IterativeImputer(max_iter=20, random_state=0).fit_transform(Xm)


def parse(folder):
    parts = folder.rsplit("_", 2)
    return parts[0], parts[1], float(parts[2]) / 100.0


def main():
    rows, summary = [], []
    folders = sorted(os.listdir(TMP))
    for folder in folders:
        d = os.path.join(TMP, folder)
        if not os.path.isdir(d):
            continue
        dname, mech, rate = parse(folder)
        X = pd.read_csv(os.path.join(d, "X_true.csv")).values.astype(float)
        n, dd = X.shape
        zero_frac = (X == 0).mean() * 100
        print(f"{folder}: n={n} d={dd} 结构零={zero_frac:.0f}%")

        # Python 5 种子
        py_mae = {m: [] for m in PY}
        py_sd = {m: [] for m in PY}
        for s in range(N_SEED):
            miss = make_missing(X, rate, mech, seed=s)
            Xm = X.copy(); Xm[miss] = np.nan
            preds = {"CRISP": impute_crisp(Xm), "MICE": impute_mice(Xm),
                     "KNN": impute_knn_direct(Xm), "mean": impute_mean(Xm)}
            for m, Xp in preds.items():
                mae = float(np.mean(np.abs(X[miss] - Xp[miss])))
                sd = float(np.mean(np.abs(Xp.sum(axis=1) - 100.0)))
                py_mae[m].append(mae); py_sd[m].append(sd)
        # R 单种子（seed 42）
        r_res = {}
        miss42 = make_missing(X, rate, mech, seed=42)
        for rm in R:
            fp = os.path.join(d, f"{rm}.csv")
            if os.path.exists(fp):
                Xp = pd.read_csv(fp).values.astype(float)
                mae = float(np.mean(np.abs(X[miss42] - Xp[miss42])))
                sd = float(np.mean(np.abs(Xp.sum(axis=1) - 100.0)))
                r_res[rm] = (mae, sd)
                rows.append({"dataset": folder, "method": rm, "mae": round(mae, 4),
                             "sumdev": round(sd, 4), "seeds": 1})
        # 汇总行
        for m in PY:
            a = np.array(py_mae[m])
            rows.append({"dataset": folder, "method": m, "mae": round(a.mean(), 4),
                         "mae_std": round(a.std(), 4), "sumdev": round(np.mean(py_sd[m]), 4), "seeds": N_SEED})
        # Wilcoxon
        for m in PY[1:]:
            try:
                p = wilcoxon(np.array(py_mae["CRISP"]), np.array(py_mae[m])).pvalue
            except ValueError:
                p = float("nan")
            summary.append({"dataset": folder, "test": f"CRISP vs {m}",
                            "p": round(p, 4), "crisp": round(np.mean(py_mae["CRISP"]), 3),
                            "other": round(np.mean(py_mae[m]), 3)})
        print(f"  CRISP {np.mean(py_mae['CRISP']):.3f}±{np.std(py_mae['CRISP']):.3f}"
              f" | MICE {np.mean(py_mae['MICE']):.3f}"
              + (f" | impKNNa {r_res['impKNNa'][0]:.3f}" if "impKNNa" in r_res else ""))

    df = pd.DataFrame(rows)
    out = os.path.join(BASE, "final_results_nd.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存 {out} ({len(df)} 行)")
    print("\n=== Wilcoxon（CRISP vs MICE，p<0.05 表示显著）===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
