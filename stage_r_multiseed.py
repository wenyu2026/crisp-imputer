# -*- coding: utf-8 -*-
"""stage_r_multiseed.py — 为 R 基线的**多种子**运行准备缺失矩阵

原 R 基线只跑了单个缺失抽取（seed=42）写进 tmp/*/X_missing.csv，因此无法与 Python 的
20 种子做配对检验。本脚本按 Python 侧完全相同的协议（同一 make_missing、同一 20 个种子）
把每个配置的逐种子缺失矩阵落盘，供 run_r_baseline_multiseed.R 在**单个 R 进程**内循环处理。

输出：tmp_rms/{tmp|tmp_nd}__{配置}/X_true.csv + X_missing_s{0..19}.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_zeros import make_missing  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "tmp_rms")
N_SEED = 20


def main():
    os.makedirs(OUT, exist_ok=True)
    staged = 0
    skipped = 0
    for root in ["tmp", "tmp_nd"]:
        src = os.path.join(BASE, root)
        if not os.path.isdir(src):
            continue
        for folder in sorted(os.listdir(src)):
            d = os.path.join(src, folder)
            if not os.path.isdir(d):
                continue
            parts = folder.rsplit("_", 2)
            if len(parts) != 3:
                continue
            mech, rate = parts[1], float(parts[2]) / 100.0
            X = pd.read_csv(os.path.join(d, "X_true.csv")).values.astype(float)

            dst = os.path.join(OUT, f"{root}__{folder}")
            os.makedirs(dst, exist_ok=True)
            pd.DataFrame(X).to_csv(os.path.join(dst, "X_true.csv"), index=False)

            n_ok = 0
            for s in range(N_SEED):
                miss = make_missing(X, rate, mech, seed=s)
                if miss.sum() == 0:
                    continue
                if miss.all(axis=0).any():
                    # 整列全缺失：sklearn 基线会删列；R 基线则无法公平比较，跳过并计数
                    skipped += 1
                    continue
                Xm = X.copy()
                Xm[miss] = np.nan
                pd.DataFrame(Xm).to_csv(os.path.join(dst, f"X_missing_s{s}.csv"), index=False)
                n_ok += 1
            staged += n_ok
            print(f"{root}/{folder}: {n_ok} seeds", flush=True)

    print(f"\n共落盘 {staged} 个缺失矩阵（跳过病态 {skipped} 个）-> {OUT}")


if __name__ == "__main__":
    main()
