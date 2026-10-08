# -*- coding: utf-8 -*-
"""choose_routing_rule.py — 用留一数据集法（LODO）挑选路由阈值并评估

评估指标不用"猜中谁赢"（两者接近时猜错没有代价），而用**MAE 后悔值**：
    regret = MAE(规则选的那个) - min(MAE(CRISP), MAE(LCRISP))
并报告规则相对于"永远用 CRISP""/"永远用 LCRISP""现行缺失率规则"的 MAE 改善。

阈值只用**其它数据集**上的配置来选，再在被留出的数据集上评估——避免在报告集上调参。
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")

d = pd.read_csv(os.path.join(BASE, "routing_diagnostic.csv"), encoding="utf-8-sig")
# 同一 (config, group) 下的多个子集共享一份聚合 MAE，故按组取 r 的中位数
g = d.groupby(["config", "group"], as_index=False).agg(
    r=("r_median", "median"), r_p75=("r_p75", "median"),
    crisp=("crisp_mae", "first"), lcrisp=("lcrisp_mae", "first"),
    n=("n", "first"), d=("d", "first"), missing_pct=("missing_pct", "median"),
    n_sub=("config", "size"))
g["dataset"] = g.config.str.split("|").str[0].str.split("_").str[0]
g["oracle"] = g[["crisp", "lcrisp"]].min(axis=1)
g["gain_oracle"] = g.crisp / g.oracle

print(f"可决策配置组: {len(g)}（覆盖 {g.n_sub.sum()} 个子集抽取）")
print(f"其中 LCRISP 更好: {(g.lcrisp < g.crisp).sum()}，CRISP 更好: {(g.lcrisp >= g.crisp).sum()}")

print("\n=== 各路线在全部配置上的平均 MAE ===")
base = {
    "永远 CRISP": g.crisp.mean(),
    "永远 LCRISP": g.lcrisp.mean(),
    "现行规则（缺失率>15% 才用 LCRISP）": np.where(g.missing_pct > 15, g.lcrisp, g.crisp).mean(),
    "上帝视角（每次选更优）": g.oracle.mean(),
}
for k, v in base.items():
    print(f"  {k:<34} {v:8.4f}")

print("\n=== 阈值扫描（全量，仅供观察，非最终选择）===")
print(f"{'阈值':>6}{'平均MAE':>10}{'相对永远CRISP':>14}{'平均后悔':>10}{'选LCRISP数':>11}")
for t in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    pred = np.where(g.r < t, g.lcrisp, g.crisp)
    print(f"{t:>6.2f}{pred.mean():>10.4f}{pred.mean()-g.crisp.mean():>+14.4f}"
          f"{(pred-g.oracle).mean():>10.4f}{int((g.r<t).sum()):>11}")

print("\n=== 留一数据集（LODO）：阈值只用其它数据集选，再在被留出数据集上评估 ===")
GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
routed_mae, chosen_t = [], []
for ds in sorted(g.dataset.unique()):
    tr = g[g.dataset != ds]
    te = g[g.dataset == ds]
    if tr.empty or te.empty:
        continue
    best_t = min(GRID, key=lambda t: np.where(tr.r < t, tr.lcrisp, tr.crisp).mean())
    pred = np.where(te.r < best_t, te.lcrisp, te.crisp)
    routed_mae.append(pred.mean())
    chosen_t.append(best_t)
    print(f"  留出 {ds:<10} 在其余数据上选得 t={best_t:<5}  → 该集 MAE {pred.mean():7.4f}"
          f"  （永远CRISP {te.crisp.mean():7.4f}，上帝 {te.oracle.mean():7.4f}）")

te_all = g.copy()
print(f"\nLODO 合计平均 MAE: {np.mean(routed_mae):.4f}   （永远CRISP {g.crisp.mean():.4f}，"
      f"上帝 {g.oracle.mean():.4f}）")
print(f"选中的阈值分布: {pd.Series(chosen_t).value_counts().to_dict()}")

t_final = float(pd.Series(chosen_t).mode().iloc[0])
print(f"\n=== 最终候选阈值 t* = {t_final}（LODO 众数）===")
pred = np.where(g.r < t_final, g.lcrisp, g.crisp)
print(f"  平均 MAE {pred.mean():.4f}，相对永远 CRISP 改善 {g.crisp.mean()-pred.mean():+.4f} "
      f"（{(g.crisp.mean()-pred.mean())/g.crisp.mean()*100:.1f}%）")
print(f"  平均后悔 {(pred-g.oracle).mean():.4f}（上帝视角为 0）")
print(f"  选择 LCRISP 的配置数 {int((g.r<t_final).sum())} / {len(g)}")
print("\n  选 LCRISP 的配置：")
print(g[g.r < t_final][["config", "n", "d", "r", "crisp", "lcrisp"]].to_string(index=False))
print("\n  被漏掉（r 不低于阈值但 LCRISP 更好且优势 >5%）的配置：")
miss = g[(g.r >= t_final) & (g.lcrisp < g.crisp * 0.95)]
print(miss[["config", "n", "d", "r", "crisp", "lcrisp"]].to_string(index=False) if len(miss) else "  （无）")
