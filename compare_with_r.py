# -*- coding: utf-8 -*-
"""compare_with_r.py — 把 R 官方基线纳入配对 Wilcoxon 检验

run_r_baseline_multiseed.R 产出的是逐种子 MAE。本脚本在同一批逐种子缺失矩阵上重算
Python 侧（CRISP 家族 + MICE/KNN/mean/SoftImpute）的逐种子 MAE，从而可以做**同种子配对**
检验——这是原来单种子 R 基线无法做到的。

输出：r_vs_python_paired.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import impute_crisp, impute_knn_direct, impute_mean  # noqa: E402
from final_compare_v3 import impute_mice, soft_impute  # noqa: E402
from crisp import lcrisp, auto_crisp  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(BASE, "tmp_rms")
PY_METHODS = ["CRISP", "LCRISP", "AutoCRISP", "MICE", "KNN", "mean", "SoftImpute"]
R_METHODS = ["impKNNa", "missForest", "lrEMplus"]


def main():
    r = pd.read_csv(os.path.join(BASE, "final_results_r_multiseed.csv"))
    r = r[r.status == "ok"]
    if r.empty:
        print("R 结果为空")
        return

    py_rows = []
    for cfg, g in r.groupby("config"):
        folder = os.path.join(STAGE, cfg)
        if not os.path.isdir(folder):
            continue
        X = pd.read_csv(os.path.join(folder, "X_true.csv")).values.astype(float)
        d = X.shape[1]
        for seed in sorted(g.seed.unique()):
            f = os.path.join(folder, f"X_missing_s{int(seed)}.csv")
            if not os.path.exists(f):
                continue
            Xm = pd.read_csv(f).values.astype(float)
            miss = np.isnan(Xm)
            if miss.sum() == 0:
                continue
            preds = {"CRISP": impute_crisp(Xm), "MICE": impute_mice(Xm),
                     "KNN": impute_knn_direct(Xm), "mean": impute_mean(Xm),
                     "SoftImpute": soft_impute(Xm),
                     "LCRISP": lcrisp(Xm, comp_idx=list(range(d)), total=100.0),
                     "AutoCRISP": auto_crisp(Xm, comp_idx=list(range(d)), total=100.0)}
            for m, Xp in preds.items():
                py_rows.append({"config": cfg, "seed": int(seed), "method": m,
                                "mae": float(np.mean(np.abs(X[miss] - Xp[miss]))),
                                "sumdev": float(np.mean(np.abs(Xp.sum(axis=1) - 100.0)))})
        print(f"[py] {cfg} done", flush=True)

    py = pd.DataFrame(py_rows)
    merged = r.merge(py, on=["config", "seed"], suffixes=("_r", "_py"))

    out = []
    for cfg, g in merged.groupby("config"):
        for rm in R_METHODS:
            gr = g[g.method_r == rm]
            if gr.empty:
                continue
            for pm in PY_METHODS:
                gp = gr[gr.method_py == pm]
                if len(gp) < 5:
                    continue
                a, b = gp.mae_py.values, gp.mae_r.values
                try:
                    p = float(wilcoxon(a, b).pvalue)
                except ValueError:
                    p = float("nan")
                out.append({
                    "config": cfg, "python": pm, "r": rm, "n_pairs": len(gp),
                    "mae_python": round(float(a.mean()), 4),
                    "mae_r": round(float(b.mean()), 4),
                    "ratio_r_over_python": round(float(b.mean() / a.mean()), 3) if a.mean() > 0 else np.nan,
                    "p": round(p, 4),
                    "winner": pm if a.mean() < b.mean() else rm,
                })
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(BASE, "r_vs_python_paired.csv"), index=False, encoding="utf-8-sig")

    print(f"\n=== R 官方基线 vs Python 实现（同种子配对 Wilcoxon）===")
    print(f"可配对组合数: {len(df)}")
    print("\n各 R 基线的胜率与显著情况：")
    for rm in R_METHODS:
        s = df[df.r == rm]
        if s.empty:
            continue
        py_win = int((s.winner != rm).sum())
        sig_py = int(((s.p < 0.05) & (s.winner != rm)).sum())
        sig_r = int(((s.p < 0.05) & (s.winner == rm)).sum())
        print(f"  {rm:<11} Python 侧胜 {py_win}/{len(s)}；"
              f"Python 显著更优 {sig_py}，R 显著更优 {sig_r}")

    print("\n逐 R 基线中位数优势比（>1 表示 Python 更准）：")
    for rm in R_METHODS:
        s = df[df.r == rm]
        if s.empty:
            continue
        print(f"  {rm:<11} 中位 ratio_r/python = {s.ratio_r_over_python.median():.3f}")

    print("\n最突出的组合（按 ratio 排序前 10）：")
    print(df.sort_values("ratio_r_over_python", ascending=False).head(10)
          [["config", "python", "r", "mae_python", "mae_r", "ratio_r_over_python", "p"]]
          .to_string(index=False))

    print("\nR 基线占优的组合（ratio < 1）计数：")
    print(df[df.ratio_r_over_python < 1].groupby("r").size().to_string())


if __name__ == "__main__":
    main()
