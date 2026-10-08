# -*- coding: utf-8 -*-
"""diagnose_routing3.py — 邻域匹配覆盖率：伪留出法在哪些配置上不该被信任

伪留出法在 cement/ge/glass/steel/ge_nd 上都选对了，但在 synthetic 上选错。
机制推断：synthetic 每行只有约 2.4 个非零组分，两行的距离只在 1–2 个坐标上算出来，
"最近邻"是碰巧在那一两个坐标上相近的行，对**缺失坐标**没有信息量。

度量：邻域匹配覆盖率。只用**有变化的列**（active，排除恒零/恒值列，如 ge_nd 的 21 个
填充列），对每个有缺失的行 i，取它与有效邻居 j 的公共已观测 active 列数，除以 active 列数，
再对所有邻居和所有行取中位数：

    support = median_i mean_j |known_i ∩ known_j| / |active|

support 高 = 邻居是在大部分组分上真匹配的 → 局部 profile 可信。

输出：routing_support.csv
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, BASE)
from diagnose_routing import load_configs  # noqa: E402
from diagnose_routing2 import pseudo_holdout_select  # noqa: E402


def support_ratio(X):
    """Median over rows-with-missing of the mean neighbour joint-active coverage."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    obs = ~np.isnan(X)
    # active = columns that are not constant among the observed entries
    active = np.zeros(d, dtype=bool)
    for j in range(d):
        v = X[obs[:, j], j]
        active[j] = v.size > 1 and np.unique(v).size > 1
    if active.sum() == 0:
        return np.nan, 0
    ratio = np.zeros(n)
    for i in range(n):
        if obs[i].all():
            ratio[i] = np.nan
            continue
        joint = obs & obs[i][None, :]
        cov = np.where(active[None, :], joint, False).sum(axis=1) / active.sum()
        finite = cov > 0
        finite[i] = False
        ratio[i] = np.mean(cov[finite]) if finite.any() else np.nan
    vals = ratio[np.isfinite(ratio)]
    return (float(np.median(vals)) if vals.size else np.nan), int(active.sum())


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

    acc = {}
    for spec in load_configs():
        c = pick(spec["config"], spec["mechanism"], spec["rate"], "CRISP")
        l = pick(spec["config"], spec["mechanism"], spec["rate"], "LCRISP")
        if c is None or l is None:
            continue
        Xm = spec["Xm"]
        sup, n_active = support_ratio(Xm)
        if not np.isfinite(sup):
            continue
        choice, pa, pb = pseudo_holdout_select(Xm, list(range(Xm.shape[1])))
        key = (spec["config"], spec["mechanism"], spec["rate"])
        a = acc.setdefault(key, {"crisp": c, "lcrisp": l, "sup": [], "ch": [],
                                 "n": Xm.shape[0], "d": Xm.shape[1],
                                 "active": n_active, "group": spec["group"]})
        a["sup"].append(sup)
        a["ch"].append(choice)

    rows = []
    for (cfg, mech, rate), a in acc.items():
        rows.append({
            "config": cfg, "group": a["group"], "n": a["n"], "d": a["d"],
            "active_cols": a["active"], "n_sub": len(a["sup"]),
            "support": round(float(np.median(a["sup"])), 4),
            "pseudo_choice": pd.Series([c for c in a["ch"] if c]).mode().iloc[0] if any(a["ch"]) else "CRISP",
            "crisp": a["crisp"], "lcrisp": a["lcrisp"],
            "true_winner": "LCRISP" if a["lcrisp"] < a["crisp"] else "CRISP",
            "lcrisp_gain": round(a["crisp"] / max(a["lcrisp"], 1e-12), 3),
        })
    df = pd.DataFrame(rows).sort_values("support")
    df["pseudo_correct"] = df.pseudo_choice == df.true_winner
    df.to_csv(os.path.join(BASE, "routing_support.csv"), index=False, encoding="utf-8-sig")

    print(f"配置组: {len(df)}")
    print("\n=== 按 support 排序 ===")
    print(df[["config", "n", "d", "active_cols", "support", "pseudo_choice", "true_winner",
              "pseudo_correct", "crisp", "lcrisp", "lcrisp_gain"]].to_string(index=False))

    print("\n=== 伪留出在各 support 区间是否可信 ===")
    print(f"{'support区间':>16}{'组数':>6}{'伪留出选对':>11}{'其中LCRISP该赢':>15}")
    for lo, hi in [(0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.01)]:
        s = df[(df.support >= lo) & (df.support < hi)]
        if s.empty:
            continue
        print(f"{lo:>7.2f}–{hi:<7.2f}{len(s):>6}{int(s.pseudo_correct.sum()):>11}"
              f"{int((s.true_winner=='LCRISP').sum()):>15}")

    print("\n=== 关键对照：synthetic 与真实大数据集的 support ===")
    for k in ["synthetic", "steel", "cement", "glass", "ge", "ge_nd"]:
        s = df[df.config.str.startswith(k + "|")]
        if not s.empty:
            print(f"  {k:<12} support={s.support.median():.4f}  active={s.active_cols.median():.0f}  "
                  f"d={s.d.median():.0f}  真实赢家={s.true_winner.mode().iloc[0]}")


if __name__ == "__main__":
    main()
