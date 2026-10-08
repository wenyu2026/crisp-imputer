# -*- coding: utf-8 -*-
"""report_router_change.py — 量化 auto_crisp 路由规则更换的实际效果

注意一个结构性事实：README 的"家族胜率"取 {CRISP, LCRISP, AutoCRISP} 三者最优，
所以**换路由不可能提高家族胜率**（AutoCRISP 最好也只能追平已入选的 LCRISP）。
真正变好的是 **AutoCRISP 单独**——它从"与 CRISP 逐位相同的空操作"变成有实际作用的选择器。

输出：router_change.csv
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv(os.path.join(BASE, "HONEST_results.csv"), encoding="utf-8-sig")
df["key"] = (df.config.astype(str) + "|" + df.mechanism.astype(str) + "|"
             + df.nominal_rate.astype(str))
BASELINES = ["MICE", "KNN", "mean", "SoftImpute"]

rows = []
for key, g in df.groupby("key"):
    m = g.pivot_table(index="method", columns="variant", values="mae_mean")
    if "CRISP" not in m.index or "AutoCRISP" not in m.index:
        continue
    base = min(m.loc[b, v] for b in BASELINES for v in ["raw", "+closure", "+backfill"])
    rows.append({
        "key": key, "group": key.split("|")[-2] if "|" in key else "",
        "dataset": key.split("|")[0],
        "crisp": m.loc["CRISP", "raw"], "lcrisp": m.loc["LCRISP", "raw"],
        "auto": m.loc["AutoCRISP", "raw"], "best_base": base,
        "auto_equals_crisp": bool(np.isclose(m.loc["AutoCRISP", "raw"], m.loc["CRISP", "raw"])),
        "auto_equals_lcrisp": bool(np.isclose(m.loc["AutoCRISP", "raw"], m.loc["LCRISP", "raw"])),
    })
r = pd.DataFrame(rows)
r["auto_win"] = r.auto < r.best_base
r["crisp_win"] = r.crisp < r.best_base
r["gain_vs_crisp"] = (r.crisp - r.auto) / r.crisp * 100
r.to_csv(os.path.join(BASE, "router_change.csv"), index=False, encoding="utf-8-sig")

print(f"配置数: {len(r)}")
print("\n=== AutoCRISP 现在选了什么 ===")
print(f"  与 CRISP 逐位相同: {int(r.auto_equals_crisp.sum())} / {len(r)}")
print(f"  与 LCRISP 逐位相同: {int(r.auto_equals_lcrisp.sum())} / {len(r)}")
print(f"  两者都不同（真正自己选）: {int((~r.auto_equals_crisp & ~r.auto_equals_lcrisp).sum())}")

print("\n=== AutoCRISP 相对 CRISP 的 MAE 变化 ===")
better = r[r.gain_vs_crisp > 0.5]
worse = r[r.gain_vs_crisp < -0.5]
same = r[(r.gain_vs_crisp.abs() <= 0.5)]
print(f"  更好（>0.5%）: {len(better)}   相同: {len(same)}   更差（>0.5%）: {len(worse)}")
if len(better):
    print(f"  改善的配置中位改善 {better.gain_vs_crisp.median():.1f}%，最大 {better.gain_vs_crisp.max():.1f}%")
if len(worse):
    print(f"  变差的配置中位恶化 {(-worse.gain_vs_crisp).median():.1f}%，最大 {(-worse.gain_vs_crisp).max():.1f}%")
    print(worse[["key", "crisp", "lcrisp", "auto", "gain_vs_crisp"]].to_string(index=False))

print("\n=== 胜率对比（vs 最强公平基线）===")
print(f"  CRISP 单独        {int(r.crisp_win.sum())} / {len(r)}")
print(f"  AutoCRISP 新路由  {int(r.auto_win.sum())} / {len(r)}")
print(f"  上帝（CRISP/LCRISP 取优） {int((r[['crisp','lcrisp']].min(axis=1) < r.best_base).sum())} / {len(r)}")

print("\n=== 按数据集：AutoCRISP 相对 CRISP 的平均变化 ===")
agg = r.groupby("dataset").agg(
    配置数=("gain_vs_crisp", "size"),
    平均改善pct=("gain_vs_crisp", "mean"),
    最大改善pct=("gain_vs_crisp", "max"),
    最差pct=("gain_vs_crisp", "min")).round(2)
print(agg.to_string())

print("\n=== 改善最明显的 8 个配置 ===")
print(r.nlargest(8, "gain_vs_crisp")[["key", "crisp", "lcrisp", "auto", "gain_vs_crisp"]].to_string(index=False))
