# -*- coding: utf-8 -*-
"""eval_r_baseline.py — 评估 R 基线（impKNNa/impCoda）的插补结果。

读 tmp/ 下每组 X_true + {method}.csv，计算 MAE / 结构零污染 / SumDev，汇总 CSV。
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(BASE, "tmp")


def main():
    rows = []
    for folder in sorted(os.listdir(TMP)):
        d = os.path.join(TMP, folder)
        if not os.path.isdir(d):
            continue
        X_true = pd.read_csv(os.path.join(d, "X_true.csv")).values.astype(float)
        X_miss = pd.read_csv(os.path.join(d, "X_missing.csv")).values.astype(float)
        miss = np.isnan(X_miss)
        parts = folder.rsplit("_", 2)  # dataset_mech_rate -> [dataset, mech, rate]
        dname, mech, rate = parts[0], parts[1], parts[2]
        for method in ["impKNNa", "impCoda"]:
            fp = os.path.join(d, f"{method}.csv")
            if not os.path.exists(fp):
                continue
            Xp = pd.read_csv(fp).values.astype(float)
            # 若 R 输出列数不同，用最小公共列
            minc = min(Xp.shape[1], X_true.shape[1])
            missing_mae = float(np.mean(np.abs(X_true[:, :minc][miss[:, :minc]] - Xp[:, :minc][miss[:, :minc]])))
            zero_mask = (X_true == 0) & ~miss
            zero_err = float(np.mean(np.abs(Xp[:, :minc][zero_mask[:, :minc]]))) if zero_mask.any() else 0.0
            sum_dev = float(np.mean(np.abs(Xp.sum(axis=1) - 100.0)))
            rows.append({"dataset": dname, "mechanism": mech, "missing": float(rate) / 100,
                         "method": method, "missing_mae": round(missing_mae, 4),
                         "zero_err": round(zero_err, 4), "sum_dev": round(sum_dev, 6)})
            print(f"{dname} {mech} {rate}% {method}: MAE={missing_mae:.4f} zero={zero_err:.4f} sumdev={sum_dev:.4f}")
    df = pd.DataFrame(rows)
    out = os.path.join(BASE, "results_r_baseline.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存 {out} ({len(df)} 行)")


if __name__ == "__main__":
    main()
