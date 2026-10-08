# -*- coding: utf-8 -*-
"""downstream_eval.py — 下游可用性：插补后的矩阵能否支撑一个预测模型？

论文 §3.6 声称 "只有 CRISP 插补的矩阵保留正的下游预测 R² (0.11)，而 MICE/KNN/mean 崩到
负 R² (−0.13 ~ −0.31)"。**仓库里此前没有任何生成该结论的脚本或结果文件**——图 5 的四个
数字是硬编码的字面量（见 make_figures.py:98）。

本脚本把该实验真正跑出来，产出 downstream_results.csv：

  对每个数据集 × 每种插补方法：
    1. 在非零位注入 20% 缺失；
    2. 插补得到完整矩阵；
    3. 用插补后的组分列做 5 折交叉验证的梯度提升回归，预测真实目标；
    4. 报告折外 R²。
  另给出"无缺失"的参照 R²（该任务的信息上限近似）。

注意：这里用**完整数据集**（n=40–1030）而不是 n<d 子集——n=5–10 的 R² 是噪声，
无法支撑任何结论。论文原来的 n<d 说法在此被更正为大规模设定。
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, impute_crisp, impute_knn_direct, impute_mean, find_data  # noqa
from final_compare_v3 import impute_mice, soft_impute  # noqa

BASE = os.path.dirname(os.path.abspath(__file__))
N_SEED = 5
RATE = 0.20
METHODS = ["CRISP", "MICE", "KNN", "mean", "SoftImpute"]


def load_dataset(name):
    specs = {
        "steel": ("steel_strength.csv",
                  ["c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"],
                  "tensile strength"),
        "glass": ("glass_40samples.csv", ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"], "RI"),
        "ge": ("ge_refractory_alloy.csv", None, "Hardness (GPa)"),
    }
    if name == "cement":
        df = pd.read_excel(os.path.join(BASE, "cement_concrete.xls"))
        X = df.iloc[:, :7].values.astype(float)
        y = df.iloc[:, -1].values.astype(float)
    else:
        fname, comps, target = specs[name]
        df = pd.read_csv(find_data(fname), encoding="utf-8-sig")
        if comps is None:
            comps = [c for c in df.columns if c.endswith("(at%)")]
        df = df.dropna(subset=[target]).reset_index(drop=True)
        X = df[comps].values.astype(float)
        y = df[target].values.astype(float)
    X = X / X.sum(axis=1, keepdims=True) * 100.0
    return X, y


def cv_r2(X, y, seed=0):
    """Out-of-fold R^2 of a gradient-boosted surrogate on (imputed) compositional features."""
    model = GradientBoostingRegressor(random_state=0)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    return float(np.mean(scores))


def main():
    rows = []
    for name in ["steel", "cement", "ge", "glass"]:
        X, y = load_dataset(name)
        n, d = X.shape
        ref = cv_r2(X, y)
        print(f"\n{name}: n={n} d={d}  无缺失参照 R²={ref:.4f}")
        rows.append({"dataset": name, "method": "none (reference)", "n": n, "d": d,
                     "r2_mean": round(ref, 4), "r2_std": 0.0, "seeds": 0})

        acc = {m: [] for m in METHODS}
        acc_sd = {m: [] for m in METHODS}
        for s in range(N_SEED):
            miss = make_missing(X, RATE, "MCAR", seed=s)
            if miss.sum() == 0 or miss.all(axis=0).any():
                continue
            Xm = X.copy()
            Xm[miss] = np.nan
            preds = {"CRISP": impute_crisp(Xm), "MICE": impute_mice(Xm),
                     "KNN": impute_knn_direct(Xm), "mean": impute_mean(Xm),
                     "SoftImpute": soft_impute(Xm)}
            for m, Xp in preds.items():
                acc[m].append(cv_r2(Xp, y, seed=s))
                acc_sd[m].append(float(np.mean(np.abs(Xp.sum(axis=1) - 100.0))))
        for m in METHODS:
            if not acc[m]:
                continue
            a = np.array(acc[m])
            print(f"   {m:<11} R²={a.mean():+.4f} ± {a.std():.4f}   SumDev={np.mean(acc_sd[m]):.4f}")
            rows.append({"dataset": name, "method": m, "n": n, "d": d,
                         "r2_mean": round(float(a.mean()), 4),
                         "r2_std": round(float(a.std()), 4),
                         "sumdev": round(float(np.mean(acc_sd[m])), 4),
                         "seeds": len(a)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "downstream_results.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已写出 downstream_results.csv（{len(df)} 行）")

    print("\n=== 汇总：20% 缺失下相对无缺失参照的 R² 损失 ===")
    for name in df.dataset.unique():
        sub = df[df.dataset == name]
        ref = sub[sub.method == "none (reference)"].r2_mean.iloc[0]
        print(f"\n{name} (参照 {ref:+.4f}):")
        for _, r in sub[sub.method != "none (reference)"].sort_values("r2_mean", ascending=False).iterrows():
            delta = r.r2_mean - ref
            print(f"   {r.method:<11} R²={r.r2_mean:+.4f}   相对参照 {delta:+.4f}")


if __name__ == "__main__":
    main()
