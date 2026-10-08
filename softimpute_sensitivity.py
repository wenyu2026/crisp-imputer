# -*- coding: utf-8 -*-
"""softimpute_sensitivity.py — SoftImpute 正则化灵敏度 + 与 v1.0 旧实现的对比

v1.0 里的 `soft_impute` 其实是**硬秩截断**（固定 rank=5、无软阈值、无中心化），
不是 Mazumder-Hastie-Tibshirani 的 SoftImpute。现版本改用核范数近端算子
（软阈值）+ 观测列均值中心化。

本脚本在若干代表性配置上：
  1. 扫描 lam_ratio（相对最大奇异值），报告 MAE / SumDev 的完整曲线——
     证明 SoftImpute 在 n<d 上的失败**不是某个正则化取值的产物**；
  2. 同时给出 v1.0 旧实现的数字，量化旧实现把 SoftImpute 的误差夸大了多少。

输出：softimpute_sensitivity.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing  # noqa: E402
from final_compare_v3 import soft_impute  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
N_SEED = 20
RATIOS = [0.0, 0.005, 0.02, 0.05, 0.15, 0.5]

CONFIGS = [
    ("tmp/ge_MCAR_10", "low-miss real (ge, 86x9)"),
    ("tmp/cement_MCAR_10", "low-miss real (cement, 1030x7)"),
    ("tmp/steel_MCAR_10", "low-miss real (steel, 300x13)"),
    ("tmp_nd/synA1_MCAR_10", "n<d synthetic (25x50)"),
    ("tmp_nd/ge_nd_MCAR_10", "n<d 'real' (20x30, effective d=9)"),
    ("tmp_realhr/steel_MCAR_20_s0", "real high-ratio n<d (8x13)"),
]


def legacy_soft_impute(Xm, rank=5, max_iter=100, tol=1e-5):
    """The v1.0 implementation, kept only for comparison.

    Hard rank truncation, no soft thresholding, no centering, unobserved entries
    initialised to column medians.
    """
    X = np.array(Xm, dtype=float, copy=True)
    for j in range(X.shape[1]):
        med = np.nanmedian(X[:, j])
        X[np.isnan(X[:, j]), j] = med if not np.isnan(med) else 0.0
    mask = ~np.isnan(Xm)
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(rank, len(s))
        X_new = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
        X_new[mask] = Xm[mask]
        if np.linalg.norm(X_new - X) < tol:
            X = X_new
            break
        X = X_new
    return X


def resolve(path):
    """Return (X_true, mech, rate) for a config folder, or for a stored subset folder."""
    p = os.path.join(BASE, path)
    X = pd.read_csv(os.path.join(p, "X_true.csv")).values.astype(float)
    if "tmp_rms" in path or "tmp_real" in path:
        return X, None, None
    parts = os.path.basename(path).rsplit("_", 2)
    return X, parts[1], float(parts[2]) / 100.0


def main():
    rows = []
    for path, label in CONFIGS:
        X, mech, rate = resolve(path)
        p = os.path.join(BASE, path)
        acc = {r: [] for r in RATIOS}
        acc_sd = {r: [] for r in RATIOS}
        legacy, legacy_sd = [], []
        seeds = range(N_SEED) if mech is not None else [None]

        for s in seeds:
            if s is None:
                Xm = pd.read_csv(os.path.join(p, "X_missing.csv")).values.astype(float)
            else:
                miss = make_missing(X, rate, mech, seed=s)
                if miss.sum() == 0:
                    continue
                Xm = X.copy()
                Xm[miss] = np.nan
            m = np.isnan(Xm)
            if m.sum() == 0:
                continue
            if m.all(axis=0).any():
                continue                      # ill-posed, same rule as honest_rerun
            for r in RATIOS:
                Xp = soft_impute(Xm, lam_ratio=r)
                acc[r].append(float(np.mean(np.abs(X[m] - Xp[m]))))
                acc_sd[r].append(float(np.mean(np.abs(Xp.sum(axis=1) - 100.0))))
            Xp = legacy_soft_impute(Xm)
            legacy.append(float(np.mean(np.abs(X[m] - Xp[m]))))
            legacy_sd.append(float(np.mean(np.abs(Xp.sum(axis=1) - 100.0))))

        if not legacy:
            print(f"{path}: 无可评配置，跳过")
            continue
        n, d = X.shape
        base = {"config": path, "label": label, "n": n, "d": d, "d_over_n": round(d / n, 2),
                "seeds": len(legacy)}
        for r in RATIOS:
            rows.append({**base, "variant": f"softimpute(lam_ratio={r})",
                         "mae": round(float(np.mean(acc[r])), 4),
                         "mae_std": round(float(np.std(acc[r])), 4),
                         "sumdev": round(float(np.mean(acc_sd[r])), 4)})
        rows.append({**base, "variant": "v1.0 legacy (hard rank-5 truncation)",
                     "mae": round(float(np.mean(legacy)), 4),
                     "mae_std": round(float(np.std(legacy)), 4),
                     "sumdev": round(float(np.mean(legacy_sd)), 4)})
        best = min(np.mean(acc[r]) for r in RATIOS)
        print(f"{path:<34} legacy={np.mean(legacy):9.4f}  best-ref={best:9.4f}  "
              f"夸大 {np.mean(legacy)/max(best,1e-9):5.2f}×")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "softimpute_sensitivity.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已写出 softimpute_sensitivity.csv（{len(df)} 行）")
    print("\n=== v1.0 旧实现 vs 参考实现最优 ===")
    piv = df.pivot_table(index="config", columns="variant", values="mae")
    for cfg in piv.index:
        leg = piv.loc[cfg, "v1.0 legacy (hard rank-5 truncation)"]
        ref = piv.loc[cfg, [c for c in piv.columns if c.startswith("softimpute(")]].min()
        print(f"  {cfg:<34} legacy {leg:9.4f} -> reference best {ref:9.4f}  ({leg/max(ref,1e-9):.2f}× 夸大)")


if __name__ == "__main__":
    main()
