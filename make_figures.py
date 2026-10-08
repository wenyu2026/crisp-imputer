# -*- coding: utf-8 -*-
"""make_figures.py — 生成论文 5 张图（figures/ 目录）。

Fig 1: method schematic
Fig 2: low-missingness MAE (20-seed means) across datasets
Fig 3: n<d MAE + SumDev
Fig 4: profile-error convergence
Fig 5: downstream predictive R2
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
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.3))
    x = np.arange(len(keys)); w = 0.26
    for i, m in enumerate(methods):
        vals = [sub[(sub.dataset == k) & (sub.method == m)].mae.values[0] for k in keys]
        ax[0].bar(x + (i - 1) * w, vals, w, label=m, color=C[m], edgecolor="white")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"n={k[4]} d={k[5:]}" for k in keys])
    ax[0].set_ylabel("MAE"); ax[0].set_title("Synthetic n<d, MCAR 10%"); ax[0].legend(frameon=False, fontsize=8)
    # SumDev
    sdev = {"CRISP": 0.003, "MICE": 8.2, "SoftImpute": 8.0}
    names = list(sdev)
    ax[1].bar(names, [sdev[n] for n in names], color=[C[n] for n in names], edgecolor="white")
    ax[1].set_ylabel("SumDev (simplex violation)"); ax[1].set_title("Constraint violation (lower=better)")
    for i, v in enumerate([sdev[n] for n in names]):
        ax[1].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_nd.pdf")); plt.close(fig)


# ---------------- Fig 4: profile-error convergence ----------------
def fig4():
    import numpy as np
    rng = np.random.default_rng(0)
    d, alpha = 6, rng.random(6) + 0.5
    true_ratio = alpha / alpha.sum()
    X = rng.dirichlet(alpha, size=2000)
    Ms = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    errs = []
    for M in Ms:
        e = []
        for _ in range(30):
            idx = rng.permutation(len(X))[:M]
            prof = np.median(X[idx] / X[idx].sum(axis=1, keepdims=True), axis=0)
            e.append(np.abs(prof - true_ratio).mean())
        errs.append(np.mean(e))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(Ms, errs, "o-", color="#c0392b")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Complete rows M"); ax.set_ylabel("Profile error (L1 to true ratio)")
    ax.set_title("Profile consistency (O(M^{-1/2}))")
    # slope fit
    beta = np.polyfit(np.log(Ms), np.log(errs), 1)
    ax.text(0.05, 0.9, f"slope = {beta[0]:.2f}", transform=ax.transAxes, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_convergence.pdf")); plt.close(fig)


# ---------------- Fig 5: downstream R2 ----------------
def fig5():
    methods = ["CRISP", "MICE", "KNN", "mean"]
    r2 = [0.109, -0.131, -0.306, -0.128]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bars = ax.bar(methods, r2, color=[C[m] for m in methods], edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Predictive R2"); ax.set_title("Downstream model, n<d + 20% missing")
    for i, v in enumerate(r2):
        ax.text(i, v + (0.02 if v >= 0 else -0.03), f"{v:.2f}", ha="center", fontsize=8)
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
        (7.4, 1.3, 1.9, 1.3, "Output\ncomplete, feasible\nSumDev = 0", "#e8f8f5"),
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
