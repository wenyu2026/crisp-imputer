# -*- coding: utf-8 -*-
"""final_real_nd.py — 真实 n<d 子集评估（方向①）

对每个真实数据集 × 5 子集：CRISP/MICE/KNN/mean（Python）+ impKNNa/missForest（R）。
聚合 5 子集 mean±std + Wilcoxon。
输出：final_results_real_nd.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import impute_crisp, impute_knn_direct, impute_mean
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

BASE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(BASE, "tmp_realnd")
PY = ["CRISP", "MICE", "KNN", "mean"]
R = ["impKNNa", "missForest"]


def impute_mice(Xm):
    return IterativeImputer(max_iter=20, random_state=0).fit_transform(Xm)


def main():
    datasets = ["steel", "ge", "glass", "cement"]
    rows, summ = [], []
    for dname in datasets:
        folders = [f for f in sorted(os.listdir(TMP)) if f.startswith(dname)]
        py_mae = {m: [] for m in PY}
        py_sd = {m: [] for m in PY}
        r_mae = {m: [] for m in R}
        r_sd = {m: [] for m in R}
        for folder in folders:
            d = os.path.join(TMP, folder)
            X = pd.read_csv(os.path.join(d, "X_true.csv")).values.astype(float)
            Xm = pd.read_csv(os.path.join(d, "X_missing.csv")).values.astype(float)
            miss = np.isnan(Xm)
            preds = {"CRISP": impute_crisp(Xm), "MICE": impute_mice(Xm),
                     "KNN": impute_knn_direct(Xm), "mean": impute_mean(Xm)}
            for m, Xp in preds.items():
                if miss.sum() == 0:
                    py_mae[m].append(0.0)
                else:
                    py_mae[m].append(float(np.mean(np.abs(X[miss] - Xp[miss]))))
                py_sd[m].append(float(np.mean(np.abs(Xp.sum(axis=1) - 100.0))))
            for rm in R:
                fp = os.path.join(d, f"{rm}.csv")
                if os.path.exists(fp):
                    Xp = pd.read_csv(fp).values.astype(float)
                    if miss.sum() == 0:
                        r_mae[rm].append(0.0)
                    else:
                        r_mae[rm].append(float(np.mean(np.abs(X[miss] - Xp[miss]))))
                    r_sd[rm].append(float(np.mean(np.abs(Xp.sum(axis=1) - 100.0))))
        # 汇总
        print(f"=== {dname} (n<d 子集 × 5) ===")
        for m in PY:
            a, s = np.array(py_mae[m]), np.array(py_sd[m])
            rows.append({"dataset": dname, "method": m, "mae_mean": round(a.mean(), 3),
                         "mae_std": round(a.std(), 3), "sumdev": round(s.mean(), 3)})
            print(f"  {m:8s} {a.mean():.3f}±{a.std():.3f} | SumDev {s.mean():.3f}")
        for rm in R:
            if r_mae[rm]:
                rows.append({"dataset": dname, "method": rm, "mae_mean": round(np.mean(r_mae[rm]), 3),
                             "mae_std": round(np.std(r_mae[rm]), 3), "sumdev": round(np.mean(r_sd[rm]), 3)})
                print(f"  {rm:8s} {np.mean(r_mae[rm]):.3f}±{np.std(r_mae[rm]):.3f} | SumDev {np.mean(r_sd[rm]):.3f}")
        # Wilcoxon CRISP vs MICE
        try:
            p = wilcoxon(np.array(py_mae["CRISP"]), np.array(py_mae["MICE"])).pvalue
        except ValueError:
            p = float("nan")
        summ.append({"dataset": dname, "test": "CRISP vs MICE", "p": round(p, 4),
                     "crisp": round(np.mean(py_mae["CRISP"]), 3), "mice": round(np.mean(py_mae["MICE"]), 3)})
        print(f"  Wilcoxon CRISP vs MICE: p={p:.3f}")
        print()

    df = pd.DataFrame(rows)
    out = os.path.join(BASE, "final_results_real_nd.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已保存 {out}")
    print("\n=== 汇总（CRISP vs MICE，真实 n<d 数据）===")
    print(pd.DataFrame(summ).to_string(index=False))


if __name__ == "__main__":
    main()
