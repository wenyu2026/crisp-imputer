# -*- coding: utf-8 -*-
"""diagnose_routing2.py — 用「伪留出交叉验证」代替阈值规则

阈值法的问题：t=0.5 大幅改善 steel/cement/glass/ge，却把 synthetic 弄差
（9.46 vs 8.54），LODO 合计反而劣于"永远 CRISP"。而且在 16 个配置上选阈值必然过拟合。

替代方案（无需任何阈值）：**用每个样本自己的已观测位做伪留出**。
把每行一个已观测非零位人为置为缺失，分别用全局 profile 和局部 profile 去补，
谁在这些"已知答案"的伪缺失位上更准，就用谁去补真正缺失的位。

这是模型选择的标准做法（内部交叉验证），决策依据来自数据本身，不是外部调参。

输出：routing_pseudo.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crisp import crisp as _crisp, lcrisp as _lcrisp  # noqa: E402
from diagnose_routing import load_configs  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
K = 5
REPEATS = 3


def pseudo_holdout_select(X, comp_idx, n_neighbors=K, repeats=REPEATS, seed=0):
    """Return (choice, mae_crisp, mae_lcrisp) from pseudo-held-out observed entries."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    comp = np.asarray(comp_idx)
    cand = (~np.isnan(X)) & (X > 0)
    cand[:, [j for j in range(d) if j not in set(comp_idx)]] = False
    # a row needs >=3 observed non-zero composition coords so that >=2 remain after masking
    enough = cand.sum(axis=1) >= 3
    rows = np.where(enough)[0]
    if rows.size == 0:
        return None, np.nan, np.nan

    rng = np.random.default_rng(seed)
    masks = np.zeros((repeats, n, d), dtype=bool)
    for t in range(repeats):
        for i in rows:
            cols = np.where(cand[i])[0]
            masks[t, i, cols[rng.integers(len(cols))]] = True
    mask = masks.any(axis=0)

    Xp = X.copy()
    Xp[mask] = np.nan
    a = _crisp(Xp, comp_idx=list(comp_idx), total=100.0)
    b = _lcrisp(Xp, comp_idx=list(comp_idx), n_neighbors=n_neighbors, total=100.0)
    mae_a = float(np.mean(np.abs(X[mask] - a[mask])))
    mae_b = float(np.mean(np.abs(X[mask] - b[mask])))
    return ("CRISP" if mae_a <= mae_b else "LCRISP"), mae_a, mae_b


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

    per_cfg = {}
    for spec in load_configs():
        c = pick(spec["config"], spec["mechanism"], spec["rate"], "CRISP")
        l = pick(spec["config"], spec["mechanism"], spec["rate"], "LCRISP")
        if c is None or l is None:
            continue
        Xm = spec["Xm"]
        d = Xm.shape[1]
        choice, pa, pb = pseudo_holdout_select(Xm, list(range(d)))
        if choice is None:
            continue
        key = (spec["config"], spec["mechanism"], spec["rate"])
        per_cfg.setdefault(key, {"crisp": c, "lcrisp": l, "choices": [], "pa": [], "pb": [],
                                 "n": Xm.shape[0], "d": d, "group": spec["group"]})
        per_cfg[key]["choices"].append(choice)
        per_cfg[key]["pa"].append(pa)
        per_cfg[key]["pb"].append(pb)

    rows = []
    for (cfg, mech, rate), v in per_cfg.items():
        # majority vote over the subsets belonging to this group
        choice = pd.Series(v["choices"]).mode().iloc[0]
        rows.append({
            "config": cfg, "group": v["group"], "n": v["n"], "d": v["d"],
            "n_sub": len(v["choices"]), "choice": choice,
            "crisp": v["crisp"], "lcrisp": v["lcrisp"],
            "oracle": min(v["crisp"], v["lcrisp"]),
            "pseudo_crisp_mae": round(float(np.mean(v["pa"])), 4),
            "pseudo_lcrisp_mae": round(float(np.mean(v["pb"])), 4),
            "true_winner": "LCRISP" if v["lcrisp"] < v["crisp"] else "CRISP",
        })
    df = pd.DataFrame(rows)
    df["routed"] = np.where(df.choice == "LCRISP", df.lcrisp, df.crisp)
    df["correct"] = df.choice == df.true_winner
    df.to_csv(os.path.join(BASE, "routing_pseudo.csv"), index=False, encoding="utf-8-sig")

    print(f"配置组: {len(df)}（覆盖 {df.n_sub.sum()} 个子集抽取）")
    print(f"伪留出选择与真实赢家一致: {int(df.correct.sum())}/{len(df)} = {df.correct.mean():.1%}")

    print("\n=== 各路线平均 MAE ===")
    for label, val in [
        ("永远 CRISP", df.crisp.mean()),
        ("永远 LCRISP", df.lcrisp.mean()),
        ("现行规则（缺失率>15%）", np.where(df.group.isin(["lowmiss", "nd"]) & (df.config.str.contains("_20|_30")), df.lcrisp, df.crisp).mean()),
        ("★ 伪留出选择", df.routed.mean()),
        ("上帝视角", df.oracle.mean()),
    ]:
        print(f"  {label:<26} {val:8.4f}")

    print("\n=== 逐配置 ===")
    print(df[["config", "n", "d", "choice", "true_winner", "correct", "crisp", "lcrisp",
              "routed", "pseudo_crisp_mae", "pseudo_lcrisp_mae"]].to_string(index=False))

    print("\n=== 按数据集看伪留出是否更优或持平 ===")
    for ds in sorted(df.config.str.split("|").str[0].unique()):
        s = df[df.config.str.split("|").str[0] == ds]
        print(f"  {ds:<20} 伪留出 {s.routed.mean():7.4f}   CRISP {s.crisp.mean():7.4f}   "
              f"上帝 {s.oracle.mean():7.4f}   {'✓更好' if s.routed.mean() < s.crisp.mean() else ('=持平' if abs(s.routed.mean()-s.crisp.mean())<1e-9 else '✗更差')}")


if __name__ == "__main__":
    main()
