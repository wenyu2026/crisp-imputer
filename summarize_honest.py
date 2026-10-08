# -*- coding: utf-8 -*-
"""summarize_honest.py — 把 HONEST_results.csv 汇总成诚实版结果表

v2：加入 LCRISP / AutoCRISP 臂、真实高比值 n<d 子集（realhr）、lrEMplus 可运行性、
以及 SoftImpute 正则化灵敏度。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "HONEST_results.csv"), encoding="utf-8-sig")
df["key"] = (df.config.astype(str) + "|" + df.mechanism.astype(str) + "|"
             + df.nominal_rate.astype(str))

pd.set_option("display.width", 230)
METHODS = ["CRISP", "LCRISP", "AutoCRISP", "MICE", "KNN", "mean", "SoftImpute"]
# CRISP 家族（CRISP / LCRISP / AutoCRISP）不是"对手"：AutoCRISP 在低缺失率下与 CRISP
# 逐位相同，若把它计入基线，CRISP 会永远"打平自己"而永远不胜。
CRISP_FAMILY = ["CRISP", "LCRISP", "AutoCRISP"]
BASELINES = [m for m in METHODS if m not in CRISP_FAMILY]


def sel(config, mech, rate):
    m = (df.config == config) & (df.mechanism == mech)
    if rate is None:
        m &= df.nominal_rate.isna()
    else:
        m &= np.isclose(df.nominal_rate, rate)
    return df[m]


def show(title, specs, mech="MCAR", rate=0.10):
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    for c in specs:
        s = sel(c, mech, rate)
        if s.empty:
            print(f"\n[{c}] 无数据")
            continue
        m = s.pivot_table(index="method", columns="variant", values="mae_mean").reindex(METHODS)
        sd = s.pivot_table(index="method", columns="variant", values="sumdev_mean").reindex(METHODS)
        i = s.iloc[0]
        extra = ""
        if not pd.isna(i.get("d_over_n")):
            extra = f"  n={int(i.n_rows)} d={int(i.n_cols)} d/n={i.d_over_n}"
        print(f"\n[{c}]{extra}")
        print(f"  实际缺失={i.actual_missing_per_seed} 格/种子 ({i.actual_missing_pct}% of 矩阵)  "
              f"来自单缺失行={i.pct_missing_from_single_rows}%  "
              f"可用行={i.usable_rows_per_seed}/{int(i.n_rows)} ({i.pct_rows_usable}%)  "
              f"seeds={i.seeds_used}")
        print(f"{'method':<12}{'raw':>10}{'+closure':>10}{'+backfill':>11}{'SumDev(raw)':>13}")
        for mth in METHODS:
            print(f"{mth:<12}{m.loc[mth,'raw']:>10.3f}{m.loc[mth,'+closure']:>10.3f}"
                  f"{m.loc[mth,'+backfill']:>11.3f}{sd.loc[mth,'raw']:>13.4f}")


show("A. 低缺失率数据集 · MCAR 10%（论文 Table 1 诚实版）",
     ["ge|lowmiss", "glass|lowmiss", "synthetic|lowmiss", "steel|lowmiss", "cement|lowmiss"])
show("B. 合成 n<d · MCAR 10%（论文 Table 2 诚实版）",
     ["synA1|nd", "synA2|nd", "synA3|nd", "synA4|nd"])
show("B2. ge_nd（有效 d=9，实为 n>d）· MCAR 10%", ["ge_nd|nd"])
show("C. 原真实 n<d 子集（d/n 1.13–1.33，单缺失行占 75–100%）· 单次抽取",
     ["glass|realnd", "cement|realnd", "ge|realnd", "steel|realnd"], mech="observed", rate=None)

print("\n" + "=" * 118)
print("D. ★ 新增：真实材料**高比值** n<d 子集（d/n 1.50–1.75，缺失率 20%/40%，多缺失行为主）")
print("=" * 118)
rh = df[df.config.str.endswith("|realhr")].copy()
if rh.empty:
    print("（无数据）")
else:
    rh["grp"] = rh.config.str.replace("|realhr", "", regex=False)
    for grp in sorted(rh.grp.unique()):
        s = rh[rh.grp == grp]
        i = s.iloc[0]
        m = s.pivot_table(index="method", columns="variant", values="mae_mean").reindex(METHODS)
        print(f"\n[{grp}]  n={int(i.n_rows)} d={int(i.n_cols)} d/n={i.d_over_n}  "
              f"缺失={i.actual_missing_per_seed} 格/子集  来自单缺失行={i.pct_missing_from_single_rows}%  "
              f"可用行={i.usable_rows_per_seed}/{int(i.n_rows)}  subsets={i.seeds_used}")
        print(f"{'method':<12}{'raw':>10}{'+closure':>10}{'+backfill':>11}")
        for mth in METHODS:
            print(f"{mth:<12}{m.loc[mth,'raw']:>10.3f}{m.loc[mth,'+closure']:>10.3f}"
                  f"{m.loc[mth,'+backfill']:>11.3f}")

# ---------- E. 胜负统计 ----------
print("\n" + "=" * 118)
print("E. 加入公平对照后，各方法相对 CRISP 的表现")
print("=" * 118)
rows = []
for key, g in df.groupby("key"):
    m = g.pivot_table(index="method", columns="variant", values="mae_mean").reindex(METHODS)
    if m.loc["CRISP"].isna().all():
        continue
    cris = m.loc["CRISP", "raw"]
    group = key.split("|")[-1]
    for mth in BASELINES:
        for v in ["raw", "+closure", "+backfill"]:
            o = m.loc[mth, v]
            rows.append({"key": key, "group": group, "baseline": f"{mth}({v})",
                         "crisp": cris, "other": o, "win": bool(cris < o),
                         "ratio": (o / cris) if cris > 0 else np.nan})
w = pd.DataFrame(rows)

wb = w[w.baseline.str.contains(r"\+backfill")]
print(f"\n配置总数 {df.key.nunique()}")
print("\n按数据组：CRISP 是否优于**最强的公平基线**（4 个基线 × 3 变体中取最小）")
best = wb.groupby("key").apply(
    lambda g: pd.Series({"best_base_mae": g.other.min(),
                         "best_base": g.loc[g.other.idxmin(), "baseline"],
                         "crisp": g.crisp.iloc[0]}), include_groups=False)
best["win"] = best.crisp < best.best_base_mae
best["ratio"] = (best.best_base_mae / best.crisp).round(2)
best["group"] = [k.split("|")[-1] for k in best.index]
print(best.groupby("group").agg(CRISP胜=("win", "sum"), 总=("win", "size"),
                                中位优势比=("ratio", "median")).to_string())

print("\n按名义缺失率分组（仅 lowmiss/nd）:")
b2 = best[best.group.isin(["lowmiss", "nd"])].copy()
b2["rate"] = [k.split("|")[-2] for k in b2.index]
print(b2.groupby("rate").agg(CRISP胜=("win", "sum"), 总=("win", "size"),
                             中位优势比=("ratio", "median")).to_string())

print("\nCRISP 输给最强公平基线的配置数:", int((~best.win).sum()), "/", len(best))

# CRISP 家族整体（CRISP / LCRISP / AutoCRISP 取最优）与最强公平基线对比
fam_best = df[df.method.isin(CRISP_FAMILY)].groupby("key").mae_mean.min().rename("family_best")
comp = pd.concat([fam_best, best[["best_base_mae", "best_base"]]], axis=1).dropna()
comp["family_win"] = comp.family_best < comp.best_base_mae
comp["ratio"] = (comp.best_base_mae / comp.family_best).round(2)
comp["group"] = [k.split("|")[-1] for k in comp.index]
print("\n【CRISP 家族（三者取最优）vs 最强公平基线】")
print(comp.groupby("group").agg(家族胜=("family_win", "sum"), 总=("family_win", "size"),
                                中位优势比=("ratio", "median")).to_string())
print("家族输掉的配置:", int((~comp.family_win).sum()), "/", len(comp))
print("\n家族仍占优（优势比 > 1.05）的配置：")
winners = comp[comp.ratio > 1.05].sort_values("ratio", ascending=False)
print(winners[["family_best", "best_base", "best_base_mae", "ratio"]].to_string())

# ---------- F. 实际缺失率 ----------
print("\n" + "=" * 118)
print("F. 名义缺失率 vs 实际缺失率")
print("=" * 118)
u = df.drop_duplicates("key")[["key", "actual_missing_per_seed", "actual_missing_pct",
                               "pct_missing_from_single_rows", "usable_rows_per_seed",
                               "n_rows", "n_cols", "d_over_n"]]
print(u[u.key.str.contains(r"\|nd\|")].sort_values("key").to_string(index=False))

# ---------- G. lrEMplus ----------
p = os.path.join(BASE, "lremplus_baseline.csv")
if os.path.exists(p):
    lrem = pd.read_csv(p)
    ok = lrem[lrem.status == "ok"]
    print("\n" + "=" * 118)
    print("G. lrEMplus —— zCompositions 官方「零 + 缺失」函数")
    print("=" * 118)
    print(f"配置总数 {len(lrem)}：可运行 {len(ok)}，失败 {len(lrem)-len(ok)}")
    for k, v in lrem.status[lrem.status != "ok"].value_counts().items():
        print(f"  {v:>4}  {k}")
    print("\n可运行的配置：")
    for _, r in ok.iterrows():
        rel = r.folder.replace("tmp_nd/", "").replace("tmp/", "")
        base = rel.rsplit("_", 1)[0]
        parts = rel.split("_")
        s = df[(df.config.str.startswith(parts[0])) & (df.mechanism == parts[1])
               & (np.isclose(df.nominal_rate, float(parts[2]) / 100.0))
               & (df.method == "CRISP") & (df.variant == "raw")]
        cm = s.mae_mean.iloc[0] if len(s) else float("nan")
        print(f"  {r.folder:<26} MAE lrEMplus={r.mae_backfill:<9.4f} "
              f"CRISP={cm:<8.4f} → CRISP 优势 {r.mae_backfill/cm:.1f}×")

# ---------- H. SoftImpute 灵敏度 ----------
p = os.path.join(BASE, "softimpute_sensitivity.csv")
if os.path.exists(p):
    ss = pd.read_csv(p, encoding="utf-8-sig")
    print("\n" + "=" * 118)
    print("H. SoftImpute 正则化灵敏度：v1.0 旧实现 vs 参考实现")
    print("=" * 118)
    print(f"{'config':<34}{'v1.0 legacy':>13}{'ref 最优':>11}{'夸大':>8}{'ref 默认0.05':>14}")
    for cfg, g in ss.groupby("config"):
        leg = g[g.variant.str.startswith("v1.0")].mae.iloc[0]
        refs = g[g.variant.str.startswith("softimpute(")]
        bestref = refs.mae.min()
        d05 = refs[refs.variant.str.contains(r"0\.05\)")].mae
        d05 = d05.iloc[0] if len(d05) else float("nan")
        print(f"{cfg:<34}{leg:>13.4f}{bestref:>11.4f}{leg/max(bestref,1e-9):>7.2f}×{d05:>14.4f}")
    print("\n（ref 最优 = 在 lam_ratio 网格上取最优，属『上帝视角』上界；"
          "默认 0.05 是预先声明的固定值）")
