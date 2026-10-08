# -*- coding: utf-8 -*-
"""app_demo.py — 应用案例演示（任务③）：n<d 配方数据，插补 → 建模 → 预测性能。

合成 n=30, d=40 的配方成分，响应 y = 材料性能（含非线性+交互，模拟真实体系）。
缺失 20% 后用各方法插补，训练模型预测 y，5 折 CV R² 对比。
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing, impute_crisp, impute_knn_direct, impute_mean
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


def impute_mice(Xm):
    return IterativeImputer(max_iter=20, random_state=0).fit_transform(Xm)


def gen(n, d, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    for i in range(n):
        nc = int(rng.integers(8, 15))
        cols = rng.choice(d, size=nc, replace=False)
        props = rng.dirichlet(np.ones(nc))
        X[i, cols] = props * 100.0
    X = X / X.sum(axis=1, keepdims=True) * 100.0
    # 材料风格响应：线性主效应 + 关键交互 + 噪声
    beta = rng.normal(0, 1, d)
    y = X @ beta + 0.5 * X[:, 0] * X[:, 3] - 0.3 * X[:, 5] ** 2 / 100.0 + rng.normal(0, 2, n)
    return X, y


def main():
    rng = np.random.default_rng(0)
    X, y = gen(50, 60, 0)
    miss = make_missing(X, 0.2, "MCAR", seed=1)
    Xm = X.copy(); Xm[miss] = np.nan
    print(f"n<d 应用场景: n={len(y)}, d={X.shape[1]} (n<d={len(y)<X.shape[1]}), 缺失 20%")

    # 无缺失基线（Ridge：n<d 下岭回归可拟合）
    kf = KFold(5, shuffle=True, random_state=42)
    def cv_r2(Xi):
        preds, truths = [], []
        for tr, va in kf.split(Xi):
            m = Ridge(alpha=1.0)
            m.fit(Xi[tr], y[tr]); preds.append(m.predict(Xi[va])); truths.append(y[va])
        return r2_score(np.concatenate(truths), np.concatenate(preds))

    print(f"无缺失基线 R2 = {cv_r2(X):.3f}")
    print()
    print(f"{'方法':10s} {'插补MAE':>10} {'建模R2':>8} {'SumDev':>8}")
    for name, fn in [("CRISP", impute_crisp), ("MICE", impute_mice),
                     ("KNN", impute_knn_direct), ("mean", impute_mean)]:
        Xp = fn(Xm)
        mae = float(np.mean(np.abs(X[miss] - Xp[miss])))
        r2 = cv_r2(Xp)
        sd = float(np.mean(np.abs(Xp.sum(axis=1) - 100.0)))
        print(f"{name:10s} {mae:10.3f} {r2:8.3f} {sd:8.3f}")


if __name__ == "__main__":
    main()
