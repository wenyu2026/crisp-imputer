# -*- coding: utf-8 -*-
"""honest_rerun.py — CRISP 诚实版重算

目的：把论文里的数字用产物重新算一遍，并补上审计指出的两处公平对照。
    1) +closure   ：把基线输出按行重新归一化到 100（纯事后投影）
    2) +backfill  ：单缺失行用闭合约束精确回填（x = 100 - Σ已知），其余行归一化
  另报告：实际缺失条目数与单缺失行占比（论文只报名义缺失率）。

协议与 final_compare_v3.py / final_nd_v3.py 完全一致（make_missing + 同样 20 个种子），
以保证与既有结果可比。输出 HONEST_results.csv / HONEST_wilcoxon.csv。

用法：cd crisp-imputer && python honest_rerun.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from benchmark_zeros import make_missing, impute_crisp, impute_knn_direct, impute_mean  # noqa
from final_compare_v3 import impute_mice, soft_impute  # noqa
from crisp import lcrisp, auto_crisp  # noqa

N_SEED = 20
PY = ["CRISP", "MICE", "KNN", "mean", "SoftImpute", "LCRISP", "AutoCRISP"]
VARIANTS = ["raw", "+closure", "+backfill"]


# ---------------- 公平对照 ----------------
def closure_project(Xp):
    """按行重新归一化到 100，保持非负。"""
    Y = np.maximum(np.asarray(Xp, dtype=float), 0.0)
    rs = Y.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    return Y / rs * 100.0


def residual_backfill(Xp, Xm):
    """单缺失行：x_miss = 100 - Σ已知（用闭合约束精确解出）；其余行只做归一化。

    关键：观测格一律取**真实观测值**。部分基线（如中心化 + 软阈值的 SoftImpute）
    输出的观测格并非原值，若直接对其做闭包会把观测格一起缩放、破坏精确回填。
    所有方法本来就能看到观测值，因此把观测格还原为真值不构成额外信息。
    """
    Y = np.maximum(np.asarray(Xp, dtype=float), 0.0).copy()
    obs = ~np.isnan(Xm)
    Y[obs] = Xm[obs]
    n_miss = np.isnan(Xm).sum(axis=1)
    known_sum = np.nansum(Xm, axis=1)
    for i in np.where(n_miss == 1)[0]:
        j = int(np.where(np.isnan(Xm[i]))[0][0])
        Y[i, j] = max(100.0 - known_sum[i], 0.0)
    return closure_project(Y)


def variants_of(Xp, Xm):
    return {"raw": np.asarray(Xp, dtype=float),
            "+closure": closure_project(Xp),
            "+backfill": residual_backfill(Xp, Xm)}


def run_one(X, miss):
    """返回 {method: {variant: Xp}} 与掩码统计。"""
    Xm = X.copy()
    Xm[miss] = np.nan
    preds = {"CRISP": impute_crisp(Xm), "MICE": impute_mice(Xm),
             "KNN": impute_knn_direct(Xm), "mean": impute_mean(Xm),
             "SoftImpute": soft_impute(Xm),
             "LCRISP": lcrisp(Xm, comp_idx=list(range(Xm.shape[1])), total=100.0),
             "AutoCRISP": auto_crisp(Xm, comp_idx=list(range(Xm.shape[1])), total=100.0)}
    # 形状守卫：sklearn 的 KNNImputer / SimpleImputer 会删除整列全缺失的列，
    # 若发生则输出列数少于输入，按掩码取值会静默错位。宁可报错也不要静默算错。
    for m, p in preds.items():
        if np.asarray(p).shape != Xm.shape:
            raise ValueError(f"{m} 返回形状 {np.asarray(p).shape}，期望 {Xm.shape}"
                             "（该配置存在整列全缺失，属病态配置）")
    out = {m: variants_of(p, Xm) for m, p in preds.items()}
    n_miss = int(miss.sum())
    n_rows_miss = int(miss.any(axis=1).sum())
    n_single = int((miss.sum(axis=1) == 1).sum())
    return out, n_miss, n_rows_miss, n_single, Xm


def eval_config(name, X, mech, rate, seeds=range(N_SEED)):
    """对单个配置跑多种子，返回明细行 + 配对检验原料。"""
    acc = {(m, v): [] for m in PY for v in VARIANTS}
    sd = {(m, v): [] for m in PY for v in VARIANTS}
    n_miss_l, n_single_l, n_usable_l, n_used = [], [], [], 0
    for s in seeds:
        miss = make_missing(X, rate, mech, seed=s)
        if miss.sum() == 0:
            continue          # 空掩码：跳过并计入 n_used（修 C8 的 NaN 根因）
        n_used += 1
        preds, n_miss, n_rows, n_single, Xm = run_one(X, miss)
        n_miss_l.append(n_miss)
        n_single_l.append(n_single)
        n_usable_l.append(int((miss.sum(axis=1) <= 1).sum()))
        for m in PY:
            for v in VARIANTS:
                Xp = preds[m][v]
                acc[(m, v)].append(float(np.mean(np.abs(X[miss] - Xp[miss]))))
                sd[(m, v)].append(float(np.mean(np.abs(Xp.sum(axis=1) - 100.0))))
    rows, wil = [], []
    if n_used == 0:
        return rows, wil, 0
    tot_cells = X.size * n_used
    for m in PY:
        for v in VARIANTS:
            a = np.array(acc[(m, v)])
            rows.append({
                "config": name, "mechanism": mech, "nominal_rate": rate,
                "method": m, "variant": v,
                "mae_mean": round(a.mean(), 4), "mae_std": round(a.std(), 4),
                "sumdev_mean": round(float(np.mean(sd[(m, v)])), 4),
                "seeds_used": n_used,
                "actual_missing_per_seed": round(float(np.mean(n_miss_l)), 2),
                "actual_missing_pct": round(float(np.mean(n_miss_l)) / X.size * 100, 3),
                "single_missing_rows_per_seed": round(float(np.mean(n_single_l)), 2),
                "usable_rows_per_seed": round(float(np.mean(n_usable_l)), 2),
                "pct_rows_usable": round(float(np.mean(n_usable_l)) / X.shape[0] * 100, 1),
                "pct_missing_from_single_rows": round(float(np.mean(n_single_l)) /
                                                      max(float(np.mean(n_miss_l)), 1e-9) * 100, 1),
                "n_rows": int(X.shape[0]), "n_cols": int(X.shape[1]),
                "d_over_n": round(X.shape[1] / X.shape[0], 2),
            })
    # 配对检验：CRISP(raw) vs 各基线的 +closure / +backfill
    for m in PY:
        if m == "CRISP":
            continue
        for v in ["raw", "+closure", "+backfill"]:
            a, b = np.array(acc[("CRISP", "raw")]), np.array(acc[(m, v)])
            try:
                p = float(wilcoxon(a, b).pvalue)
            except ValueError:
                p = float("nan")
            wil.append({"config": name, "test": f"CRISP(raw) vs {m}({v})",
                        "p": round(p, 4),
                        "crisp_mae": round(a.mean(), 4), "other_mae": round(b.mean(), 4),
                        "winner": "CRISP" if a.mean() < b.mean() else m})
    return rows, wil, n_used


def run_stored_subsets(root, tag):
    """Aggregate pre-generated subset folders holding X_true.csv / X_missing.csv.

    Used for `tmp_realnd/` (real n<d subsets, one missing draw each) and
    `tmp_realhr/` (real high-ratio n<d subsets, 20 draws each). One result row per
    (dataset-group, method, variant).
    """
    if not os.path.isdir(root):
        print(f"[{tag}] {root} 不存在，跳过", flush=True)
        return [], []

    by_ds = {}
    skipped_illposed = 0
    for sub in sorted(os.listdir(root)):
        p = os.path.join(root, sub)
        if not os.path.isdir(p):
            continue
        ds = sub.rsplit("_s", 1)[0]
        X = pd.read_csv(os.path.join(p, "X_true.csv")).values.astype(float)
        Xm = pd.read_csv(os.path.join(p, "X_missing.csv")).values.astype(float)
        miss = np.isnan(Xm)
        if miss.sum() == 0:
            continue
        if miss.all(axis=0).any():
            # 整列全缺失：sklearn 基线会删除该列，比较无法在同一索引上进行。
            # 这类配置是病态的（该组分从未被观测），直接跳过并计数。
            skipped_illposed += 1
            continue

        preds = {"CRISP": impute_crisp(Xm), "MICE": impute_mice(Xm),
                 "KNN": impute_knn_direct(Xm), "mean": impute_mean(Xm),
                 "SoftImpute": soft_impute(Xm),
                 "LCRISP": lcrisp(Xm, comp_idx=list(range(Xm.shape[1])), total=100.0),
                 "AutoCRISP": auto_crisp(Xm, comp_idx=list(range(Xm.shape[1])), total=100.0)}

        n, d = X.shape
        parts = ds.split("_")
        mech = parts[-2] if len(parts) >= 3 else "observed"
        rate = float(parts[-1]) / 100.0 if len(parts) >= 3 else np.nan
        agg = by_ds.setdefault(ds, {
            "mae": {m: {v: [] for v in VARIANTS} for m in PY},
            "sd": {m: {v: [] for v in VARIANTS} for m in PY},
            "nm": [], "ns": [], "nu": [], "n": n, "d": d, "mech": mech, "rate": rate})
        agg["nm"].append(int(miss.sum()))
        agg["ns"].append(int((miss.sum(axis=1) == 1).sum()))
        agg["nu"].append(int((miss.sum(axis=1) <= 1).sum()))
        for m in PY:
            for v in VARIANTS:
                Xp = variants_of(preds[m], Xm)[v]
                agg["mae"][m][v].append(float(np.mean(np.abs(X[miss] - Xp[miss]))))
                agg["sd"][m][v].append(float(np.mean(np.abs(Xp.sum(axis=1) - 100.0))))
        print(f"[{tag}] {sub}: n={n} d={d} n_miss={int(miss.sum())}", flush=True)

    rows, wil = [], []
    for ds, agg in by_ds.items():
        n, d = agg["n"], agg["d"]
        mean_nm = float(np.mean(agg["nm"]))
        mean_ns = float(np.mean(agg["ns"]))
        mean_nu = float(np.mean(agg["nu"]))
        for m in PY:
            for v in VARIANTS:
                a = np.array(agg["mae"][m][v])
                rows.append({
                    "config": f"{ds}|{tag}", "mechanism": agg["mech"],
                    "nominal_rate": agg["rate"],
                    "method": m, "variant": v,
                    "mae_mean": round(a.mean(), 4), "mae_std": round(a.std(), 4),
                    "sumdev_mean": round(float(np.mean(agg["sd"][m][v])), 4),
                    "seeds_used": len(a),
                    "actual_missing_per_seed": round(mean_nm, 2),
                    "actual_missing_pct": round(mean_nm / (n * d) * 100, 3),
                    "single_missing_rows_per_seed": round(mean_ns, 2),
                    "usable_rows_per_seed": round(mean_nu, 2),
                    "pct_rows_usable": round(mean_nu / n * 100, 1),
                    "pct_missing_from_single_rows": round(mean_ns / max(mean_nm, 1e-9) * 100, 1),
                    "n_rows": n, "n_cols": d, "d_over_n": round(d / n, 2),
                })
        for m in PY:
            if m == "CRISP":
                continue
            for v in ["raw", "+closure", "+backfill"]:
                a = np.array(agg["mae"]["CRISP"]["raw"])
                b = np.array(agg["mae"][m][v])
                try:
                    p = float(wilcoxon(a, b).pvalue)
                except ValueError:
                    p = float("nan")
                wil.append({"config": f"{ds}|{tag}", "test": f"CRISP(raw) vs {m}({v})",
                            "p": round(p, 4), "crisp_mae": round(a.mean(), 4),
                            "other_mae": round(b.mean(), 4),
                            "winner": "CRISP" if a.mean() < b.mean() else m})
    if skipped_illposed:
        print(f"[{tag}] 跳过 {skipped_illposed} 个整列全缺失的病态子集", flush=True)
    return rows, wil


def main():
    all_rows, all_wil = [], []

    # ---- 1) 低缺失率数据集（tmp/），按原协议 20 种子重生成 ----
    TMP = os.path.join(BASE, "tmp")
    for folder in sorted(os.listdir(TMP)):
        d = os.path.join(TMP, folder)
        if not os.path.isdir(d):
            continue
        parts = folder.rsplit("_", 2)
        dname, mech, rate = parts[0], parts[1], float(parts[2]) / 100.0
        X = pd.read_csv(os.path.join(d, "X_true.csv")).values.astype(float)
        r, w, n = eval_config(f"{dname}|lowmiss", X, mech, rate)
        all_rows += r
        all_wil += w
        print(f"[lowmiss] {folder}: seeds={n} done", flush=True)

    # ---- 2) n<d 数据集（tmp_nd/），同样 20 种子 ----
    TND = os.path.join(BASE, "tmp_nd")
    for folder in sorted(os.listdir(TND)):
        d = os.path.join(TND, folder)
        if not os.path.isdir(d):
            continue
        parts = folder.rsplit("_", 2)
        dname, mech, rate = parts[0], parts[1], float(parts[2]) / 100.0
        X = pd.read_csv(os.path.join(d, "X_true.csv")).values.astype(float)
        r, w, n = eval_config(f"{dname}|nd", X, mech, rate)
        all_rows += r
        all_wil += w
        print(f"[nd] {folder}: seeds={n} done", flush=True)

    # ---- 3) 真实材料 n<d 子集（已落盘的 X_true / X_missing）----
    #   tmp_realnd : 原有子集，d/n 1.125–1.333，单缺失行占 75–100%
    #   tmp_realhr : 高比值子集，d/n 1.50–1.75，缺失率 20%/40%，多缺失行占主导
    for root, tag in [("tmp_realnd", "realnd"), ("tmp_realhr", "realhr")]:
        r, w = run_stored_subsets(os.path.join(BASE, root), tag)
        all_rows += r
        all_wil += w

    pd.DataFrame(all_rows).to_csv(os.path.join(BASE, "HONEST_results.csv"),
                                  index=False, encoding="utf-8-sig")
    pd.DataFrame(all_wil).to_csv(os.path.join(BASE, "HONEST_wilcoxon.csv"),
                                 index=False, encoding="utf-8-sig")
    print(f"\nDONE  rows={len(all_rows)}  tests={len(all_wil)}")


if __name__ == "__main__":
    main()
