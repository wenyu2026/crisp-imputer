# -*- coding: utf-8 -*-
"""summarize_honest.py — 把 HONEST_results.csv 汇总成诚实版结果表

修正：config 键必须带上 mechanism 与 nominal_rate，否则同名配置会被平均。
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "HONEST_results.csv"), encoding="utf-8-sig")
df["key"] = (df.config.astype(str) + "|" + df.mechanism.astype(str) + "|"
             + df.nominal_rate.astype(str))
lrem = None
p = os.path.join(BASE, "lremplus_baseline.csv")
if os.path.exists(p):
    lrem = pd.read_csv(p)

pd.set_option("display.width", 220)
METHODS = ["CRISP", "MICE", "KNN", "mean", "SoftImpute"]


def sel(config, mech, rate):
    m = (df.config == config) & (df.mechanism == mech)
    if rate is None:
        m &= df.nominal_rate.isna()
    else:
        m &= np.isclose(df.nominal_rate, rate)
    return df[m]


def show(title, specs, mech="MCAR", rate=0.10):
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    for c in specs:
        s = sel(c, mech, rate)
        if s.empty:
            print(f"\n[{c}] 无数据")
            continue
        m = s.pivot_table(index="method", columns="variant", values="mae_mean").reindex(METHODS)
        sd = s.pivot_table(index="method", columns="variant", values="sumdev_mean").reindex(METHODS)
        i = s.iloc[0]
        print(f"\n[{c}]  实际缺失={i.actual_missing_per_seed} 格/种子 "
              f"({i.actual_missing_pct}% of 矩阵)  "
              f"来自单缺失行={i.pct_missing_from_single_rows}%  seeds={i.seeds_used}")
        print(f"{'method':<12}{'raw':>10}{'+closure':>10}{'+backfill':>11}")
        for mth in METHODS:
            print(f"{mth:<12}{m.loc[mth,'raw']:>10.3f}{m.loc[mth,'+closure']:>10.3f}"
                  f"{m.loc[mth,'+backfill']:>11.3f}")
        print(f"{'SumDev(raw)':<12}{sd.loc['CRISP','raw']:>10.4f}{sd.loc['MICE','raw']:>10.4f}"
              f"{sd.loc['KNN','raw']:>11.4f}   ← CRISP / MICE / KNN")


# ---------- A / B / C ----------
show("A. 低缺失率数据集 · MCAR 10%（论文 Table 1 诚实版）",
     ["ge|lowmiss", "glass|lowmiss", "synthetic|lowmiss", "steel|lowmiss", "cement|lowmiss"])
show("B. n<d 数据集 · MCAR 10%（论文 Table 2 诚实版）",
     ["synA1|nd", "synA2|nd", "synA3|nd", "synA4|nd", "ge_nd|nd"])
show("C. 真实材料 n<d 子集（论文 Finding 3 诚实版）",
     ["glass|realnd", "cement|realnd", "ge|realnd", "steel|realnd"], mech="observed", rate=None)

# ---------- D. 加入公平对照后 CRISP 还赢多少 ----------
print("\n" + "=" * 112)
print("D. 加入公平对照后，CRISP 到底还赢多少")
print("=" * 112)
rows = []
for key, g in df.groupby("key"):
    m = g.pivot_table(index="method", columns="variant", values="mae_mean").reindex(METHODS)
    if "CRISP" not in m.index or m.loc["CRISP"].isna().all():
        continue
    cris = m.loc["CRISP", "raw"]
    for mth in METHODS[1:]:
        for v in ["raw", "+closure", "+backfill"]:
            o = m.loc[mth, v]
            rows.append({"key": key, "group": str(g.config.iloc[0]).split("|")[-1],
                         "baseline": f"{mth}({v})", "crisp": cris, "other": o,
                         "win": bool(cris < o), "ratio": (o / cris) if cris > 0 else np.nan})
w = pd.DataFrame(rows)

print("\nCRISP 胜率（按基线变体；分母 = 100 个配置）:")
t = w.groupby("baseline").agg(胜=("win", "sum"), 总=("win", "size"))
t["胜率"] = (t.胜 / t.总 * 100).round(1).astype(str) + "%"
print(t.to_string())

wb = w[w.baseline.str.contains(r"\+backfill")]
print("\n【最公平对照 +backfill 下】逐基线胜负：")
for mth in ["MICE", "KNN", "mean", "SoftImpute"]:
    s = wb[wb.baseline == f"{mth}(+backfill)"]
    print(f"  {mth:<11} CRISP 赢 {s.win.sum():>2}/{len(s)}   中位优势比 {s.ratio.median():.2f}×")

best = wb.groupby("key").apply(
    lambda g: pd.Series({"best_base_mae": g.other.min(),
                         "best_base": g.loc[g.other.idxmin(), "baseline"],
                         "crisp": g.crisp.iloc[0]}), include_groups=False)
best["win"] = best.crisp < best.best_base_mae
best["ratio"] = (best.best_base_mae / best.crisp).round(2)
best["group"] = [k.split("|")[-1] for k in best.index]
print("\n按数据组统计（对照 = 最强基线 +backfill）：")
print(best.groupby("group").agg(CRISP胜=("win", "sum"), 总=("win", "size"),
                                中位优势比=("ratio", "median")).to_string())
print("\nCRISP 输给最强公平基线的配置：")
print(best[~best.win][["crisp", "best_base", "best_base_mae", "ratio"]].to_string())

# ---------- E. 实际缺失率 ----------
print("\n" + "=" * 112)
print("E. 名义缺失率 vs 实际缺失率（论文从未报告实际值）")
print("=" * 112)
u = df.drop_duplicates("key")[["key", "nominal_rate", "actual_missing_per_seed",
                               "actual_missing_pct", "pct_missing_from_single_rows", "seeds_used"]]
u = u.sort_values("key")
print(u[u.key.str.contains("nd|") | u.key.str.contains("ge_nd")].to_string(index=False))

# ---------- F. lrEMplus ----------
if lrem is not None:
    print("\n" + "=" * 112)
    print("F. lrEMplus 基线（官方 zCompositions 的『零+缺失』函数）")
    print("=" * 112)
    ok = lrem[lrem.status == "ok"]
    print(f"总计 {len(lrem)} 个配置：成功 {len(ok)}，失败 {len(lrem)-len(ok)}")
    print("\n失败原因分布：")
    for k, v in lrem.status[lrem.status != "ok"].value_counts().items():
        print(f"  {v:>4}  {k}")
    print("\n能跑通的配置：")
    for _, r in ok.iterrows():
        rel = r.folder.replace("tmp_nd/", "").replace("tmp/", "")
        s = df[(df.config.str.startswith(rel.split("_")[0])) & (df.mechanism == rel.split("_")[1])
               & (np.isclose(df.nominal_rate, float(rel.rsplit("_", 1)[1]) / 100.0))]
        cm = s[s.method == "CRISP"].mae_mean.iloc[0] if len(s) else float("nan")
        print(f"  {r.folder:<26} n_miss={int(r.n_miss):<3} MAE raw={r.mae_raw:<9.4f} "
              f"+backfill={r.mae_backfill:<9.4f} | CRISP={cm:<7.4f} → 优势 {r.mae_backfill/cm:.1f}×")
