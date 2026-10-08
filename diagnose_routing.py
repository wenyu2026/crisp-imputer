# -*- coding: utf-8 -*-
"""diagnose_routing.py — 为 auto_crisp 寻找一个有用的路由信号

现状：`auto_crisp` 只按**缺失率**切换（threshold=0.15）。但低缺失率下 LCRISP 明明在
steel/cement/glass/ge 上更好，所以它在每个低缺失率配置上都选错。

候选信号：**邻域信息量**。对每个样本 i，用它自己的已观测坐标算到其它行的距离，
取 k 个最近的有效邻居的平均距离 d_k(i)，再除以 i 到所有有效行的中位距离 d_typ(i)：

    r(i) = d_k(i) / d_typ(i)

r 接近 1 表示"最近邻并不比一般行更近"（没有局部结构）→ 全局 profile 更稳；
r 远小于 1 表示邻居确实相似 → 局部 profile 更可信。

输出：routing_diagnostic.csv（逐配置的 r 统计量与 CRISP/LCRISP 的实际胜负）
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
K = 5


def sample_ratios(X_missing, k=K):
    """Per-sample d_k / d_typ, following LCRISP's own distance definition.

    Distance between i and j = mean squared difference over the composition
    coordinates that are observed in *both* rows; inf if none. Returns the array of
    ratios for samples that have at least one missing entry and at least k valid
    neighbours.
    """
    X = np.asarray(X_missing, dtype=float)
    n, d = X.shape
    obs = ~np.isnan(X)
    filled = np.nan_to_num(X, nan=0.0)
    out = []
    for i in range(n):
        known = obs[i]
        if known.sum() == 0 or not (~known).any():      # nothing observed, or no missing
            continue
        joint = obs & known[None, :]
        denom = joint.sum(axis=1)
        num = np.where(joint, (filled - filled[i][None, :]) ** 2, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = np.where(denom > 0, num / np.maximum(denom, 1), np.inf)
        finite = np.isfinite(dist) & (dist > 1e-10)
        if finite.sum() < k + 1:
            continue
        vals = dist[finite]
        d_k = float(np.mean(np.sort(vals)[:k]))
        d_typ = float(np.median(vals))
        if d_typ > 0:
            out.append(d_k / d_typ)
    return np.array(out)


def load_configs():
    """Yield dicts (config, mechanism, rate, X_missing, group) for every benchmark config."""
    for root, tag in [("tmp", "lowmiss"), ("tmp_nd", "nd")]:
        src = os.path.join(BASE, root)
        if not os.path.isdir(src):
            continue
        for folder in sorted(os.listdir(src)):
            p = os.path.join(src, folder)
            if not os.path.isdir(p):
                continue
            parts = folder.rsplit("_", 2)
            mech, rate = parts[1], float(parts[2]) / 100.0
            X = pd.read_csv(os.path.join(p, "X_true.csv")).values.astype(float)
            miss = make_missing(X, rate, mech, seed=0)
            if miss.sum() == 0 or miss.all(axis=0).any():
                continue
            Xm = X.copy()
            Xm[miss] = np.nan
            yield {"config": f"{parts[0]}|{tag}", "mechanism": mech, "rate": rate,
                   "Xm": Xm, "group": tag}
    for root, tag in [("tmp_realnd", "realnd"), ("tmp_realhr", "realhr")]:
        src = os.path.join(BASE, root)
        if not os.path.isdir(src):
            continue
        for folder in sorted(os.listdir(src)):
            f = os.path.join(src, folder, "X_missing.csv")
            if not os.path.exists(f):
                continue
            Xm = pd.read_csv(f).values.astype(float)
            if np.isnan(Xm).sum() == 0 or np.isnan(Xm).all(axis=0).any():
                continue
            ds = folder.rsplit("_s", 1)[0]
            parts = ds.split("_")
            if len(parts) >= 3:
                mech, rate = parts[-2], float(parts[-1]) / 100.0
            else:
                mech, rate = "observed", float("nan")
            yield {"config": f"{ds}|{tag}", "mechanism": mech, "rate": rate,
                   "Xm": Xm, "group": tag}


def main():
    honest = pd.read_csv(os.path.join(BASE, "HONEST_results.csv"), encoding="utf-8-sig")
    honest = honest[honest.variant == "raw"]

    def pick(cfg, mech, rate, method):
        h = honest[(honest.config == cfg) & (honest.method == method)]
        if mech == "observed" or (isinstance(rate, float) and np.isnan(rate)):
            h = h[h.mechanism == "observed"]
        else:
            h = h[(h.mechanism == mech) & np.isclose(h.nominal_rate, rate)]
        return None if h.empty else float(h.mae_mean.iloc[0])

    rows, skipped = [], 0
    for spec in load_configs():
        c = pick(spec["config"], spec["mechanism"], spec["rate"], "CRISP")
        l = pick(spec["config"], spec["mechanism"], spec["rate"], "LCRISP")
        if c is None or l is None:
            skipped += 1
            continue
        r = sample_ratios(spec["Xm"])
        if r.size < 3:
            skipped += 1
            continue
        Xm = spec["Xm"]
        rows.append({
            "config": spec["config"], "group": spec["group"],
            "n": Xm.shape[0], "d": Xm.shape[1],
            "missing_pct": round(float(np.isnan(Xm).mean()) * 100, 2),
            "r_median": round(float(np.median(r)), 4),
            "r_mean": round(float(np.mean(r)), 4),
            "r_p75": round(float(np.percentile(r, 75)), 4),
            "crisp_mae": c, "lcrisp_mae": l,
            "winner": "LCRISP" if l < c else "CRISP",
            "lcrisp_gain": round(c / max(l, 1e-12), 3),
        })
    if skipped:
        print(f"（跳过 {skipped} 个无法配对的配置）")
    df = pd.DataFrame(rows).sort_values("r_median")
    df.to_csv(os.path.join(BASE, "routing_diagnostic.csv"), index=False, encoding="utf-8-sig")

    print(f"配置数: {len(df)}   其中 LCRISP 更好: {(df.winner=='LCRISP').sum()}, CRISP 更好: {(df.winner=='CRISP').sum()}")
    print("\n=== 按 r_median 排序（越小 = 邻域越有信息量）===")
    print(df[["config", "group", "n", "d", "missing_pct", "r_median", "crisp_mae",
              "lcrisp_mae", "winner", "lcrisp_gain"]].to_string(index=False))

    print("\n=== r_median 对胜负的区分度 ===")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = np.where(df.r_median < t, "LCRISP", "CRISP")
        acc = float((pred == df.winner).mean())
        print(f"  threshold={t:.1f}  规则命中 {int((pred==df.winner).sum())}/{len(df)} = {acc:.1%}")


if __name__ == "__main__":
    main()
