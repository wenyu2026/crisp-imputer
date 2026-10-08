# -*- coding: utf-8 -*-
"""make_figures.py — 生成论文 5 张图（figures/ 目录）。

Fig 1: method schematic
Fig 2: low-missingness MAE (20-seed means) across datasets
Fig 3: n<d MAE + SumDev (read from final_results_nd_v3.csv)
Fig 4: profile-error convergence (median vs mean profile)
Fig 5: downstream predictive R2 (read from downstream_results.csv)

所有数值均从结果表读取；图 3/图 5 此前使用的是硬编码字面量，已修正。
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = {"CRISP": "#c0392b", "MICE": "#2980b9", "SoftImpute": "#7f8c8d", "impKNNa": "#27ae60",
     "missForest": "#8e44ad", "KNN": "#f39c12", "mean": "#95a5a6"}


# ---------------- Fig 2: low-missingness MAE ----------------
def fig2():
    df = pd.read_csv(os.path.join(BASE, "final_results_v3.csv"))
    sub = df[(df.mechanism == "MCAR") & (df.missing == 0.1)]
    datasets = ["ge", "glass", "synthetic", "steel", "cement"]
    methods = ["CRISP", "MICE", "impKNNa", "KNN", "SoftImpute"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(datasets)); w = 0.16
    for i, m in enumerate(methods):
        vals = [sub[(sub.dataset == d) & (sub.method == m)].mae_mean.values[0] if len(sub[(sub.dataset == d) & (sub.method == m)]) else np.nan for d in datasets]
        err = [sub[(sub.dataset == d) & (sub.method == m)].mae_std.values[0] if len(sub[(sub.dataset == d) & (sub.method == m)]) else np.nan for d in datasets]
        ax.bar(x + (i - 2) * w, vals, w, label=m, color=C[m], edgecolor="white", linewidth=0.4)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["ge (alloy)", "glass", "synthetic", "steel", "cement"])
    ax.set_ylabel("Missing-entry MAE (log)"); ax.set_title("Low-missingness data, MCAR 10% (20-seed means)")
    ax.legend(ncol=5, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_mae_lowmiss.pdf")); plt.close(fig)


# ---------------- Fig 3: n<d MAE + SumDev ----------------
def fig3():
    df = pd.read_csv(os.path.join(BASE, "final_results_nd_v3.csv"))
    sub = df[df.dataset.str.contains("MCAR_10")]
    keys = [k for k in sub.dataset.unique() if "synA" in k][:4]
    methods = ["CRISP", "MICE", "SoftImpute"]

    def shape_of(folder):
        a = pd.read_csv(os.path.join(BASE, "tmp_nd", folder, "X_true.csv")).values
        return a.shape

    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.3))
    x = np.arange(len(keys)); w = 0.26
    for i, m in enumerate(methods):
        vals = [sub[(sub.dataset == k) & (sub.method == m)].mae.values[0] for k in keys]
        ax[0].bar(x + (i - 1) * w, vals, w, label=m, color=C[m], edgecolor="white")
    labels = [f"n={shape_of(k)[0]} d={shape_of(k)[1]}" for k in keys]
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("MAE"); ax[0].set_title("Synthetic n<d, MCAR 10%"); ax[0].legend(frameon=False, fontsize=8)

    # SumDev is read from the result table — it used to be a hard-coded literal
    sdev = {m: float(np.mean([sub[(sub.dataset == k) & (sub.method == m)].sumdev.values[0]
                              for k in keys])) for m in methods}
    ax[1].bar(list(sdev), [sdev[m] for m in sdev], color=[C[m] for m in sdev], edgecolor="white")
    ax[1].set_ylabel("SumDev (simplex violation)")
    ax[1].set_title("Constraint violation (lower = better)")
    for i, m in enumerate(sdev):
        ax[1].text(i, sdev[m] + 0.15, f"{sdev[m]:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_nd.pdf")); plt.close(fig)


# ---------------- Fig 4: profile-error convergence ----------------
def fig4():
    rng = np.random.default_rng(0)
    d, alpha = 6, rng.random(6) + 0.5
    true_ratio = alpha / alpha.sum()
    X = rng.dirichlet(alpha, size=2000)
    Ms = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    med, men = [], []
    for M in Ms:
        em, ea = [], []
        for _ in range(30):
            idx = rng.permutation(len(X))[:M]
            Z = X[idx] / X[idx].sum(axis=1, keepdims=True)
            em.append(np.abs(np.median(Z, axis=0) - true_ratio).mean())
            ea.append(np.abs(Z.mean(axis=0) - true_ratio).mean())
        med.append(np.mean(em))
        men.append(np.mean(ea))

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(Ms, med, "o-", color="#c0392b", ms=3, label="median profile (shipped default)")
    ax.plot(Ms, men, "s-", color="#2980b9", ms=3, label="mean profile")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Complete rows M"); ax.set_ylabel("Profile error (L1 to true ratio)")
    ax.set_title("Profile consistency")
    bm = np.polyfit(np.log(Ms), np.log(med), 1)[0]
    ba = np.polyfit(np.log(Ms), np.log(men), 1)[0]
    ax.text(0.04, 0.90, f"fitted slope: median {bm:.2f}, mean {ba:.2f}",
            transform=ax.transAxes, fontsize=8)
    ax.text(0.04, 0.79, r"$O(M^{-1/2})$ reference slope $= -0.5$",
            transform=ax.transAxes, fontsize=8, style="italic")
    ax.text(0.04, 0.68, "median has a bias floor (converges to the population\nmedian, not to $\\alpha/\\Sigma\\alpha$)",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_convergence.pdf")); plt.close(fig)


# ---------------- Fig 5: downstream R2 ----------------
def fig5():
    """Reads downstream_results.csv, produced by downstream_eval.py.

    The previous version of this figure plotted four hard-coded literals
    ([0.109, -0.131, -0.306, -0.128]) for which no generating script existed.
    """
    path = os.path.join(BASE, "downstream_results.csv")
    if not os.path.exists(path):
        print("fig5 skipped: run downstream_eval.py first")
        return
    df = pd.read_csv(path)
    datasets = ["steel", "cement", "ge", "glass"]
    methods = ["CRISP", "MICE", "KNN", "mean", "SoftImpute"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(datasets)); w = 0.16
    for i, m in enumerate(methods):
        vals = []
        for ds in datasets:
            s = df[(df.dataset == ds) & (df.method == m)]
            vals.append(float(s.r2_mean.iloc[0]) if len(s) else np.nan)
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=C[m], edgecolor="white", linewidth=0.4)
    refs = [float(df[(df.dataset == ds) & (df.method == "none (reference)")].r2_mean.iloc[0])
            for ds in datasets]
    ax.plot(x, refs, "k_", ms=16, mew=2, label="no missingness (reference)")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("Out-of-fold R²")
    ax.set_title("Downstream surrogate on imputed features, MCAR 20% missing")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_downstream.pdf")); plt.close(fig)


# ---------------- Fig 1: schematic ----------------
def fig1():
    fig, ax = plt.subplots(figsize=(8, 3.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    # boxes
    boxes = [
        (0.2, 1.3, 1.9, 1.3, "Input\ncomposition matrix\n(zeros + NA)", "#fef9e7"),
        (2.6, 1.3, 1.9, 1.3, "Step 1\nprofile p-hat\nmedian of\ncomplete rows", "#fdecea"),
        (5.0, 1.3, 1.9, 1.3, "Step 2\nallocate R by\np-hat over\nmissing coords", "#eaf2f8"),
        (7.4, 1.3, 1.9, 1.3, "Output\ncomplete, feasible\nimputed rows sum\nto 100", "#e8f8f5"),
    ]
    for x, y, w, h, t, fc in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
    for i in range(3):
        ax.annotate("", xy=(boxes[i + 1][0], 1.95), xytext=(boxes[i][0] + boxes[i][2], 1.95),
                    arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(5, 0.35, "Structural zeros (0) are never touched; every row sums to 100 by construction.",
            ha="center", fontsize=9, style="italic")
    ax.set_title("CRISP: compositional ratio-based imputation with simplex projection", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_schematic.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("已生成:", os.listdir(FIG))
