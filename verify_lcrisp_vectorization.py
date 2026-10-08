# -*- coding: utf-8 -*-
"""一次性验证：向量化后的 LCRISP 是否与旧的三层循环实现逐位一致。"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from crisp import LCRISPImputer  # noqa: E402
from benchmark_zeros import make_missing  # noqa: E402


class LCRISPOld(LCRISPImputer):
    """旧实现（三层 Python 循环），仅用于比对。"""

    def _estimate_local_profiles(self, X_filled, X_missing, n_neighbors):
        n_samples = X_filled.shape[0]
        n_comp = self.n_comp_
        local_profiles = np.zeros((n_samples, n_comp))
        for i in range(n_samples):
            if len(self.non_comp_idx) > 0:
                features_for_distance = self.non_comp_idx
            else:
                features_for_distance = [idx for idx in self.comp_idx if not np.isnan(X_missing[i, idx])]
                if len(features_for_distance) == 0:
                    local_profiles[i] = self.global_profile_
                    continue
            distances = np.zeros(n_samples)
            for j in range(n_samples):
                valid_count = 0
                dist_sum = 0.0
                for f in features_for_distance:
                    if not np.isnan(X_missing[i, f]) and not np.isnan(X_missing[j, f]):
                        dist_sum += (X_filled[i, f] - X_filled[j, f]) ** 2
                        valid_count += 1
                distances[j] = np.inf if valid_count == 0 else dist_sum / valid_count
            sorted_idx = np.argsort(distances)
            neighbor_idx = []
            for idx in sorted_idx:
                if len(neighbor_idx) >= n_neighbors:
                    break
                if distances[idx] < np.inf and distances[idx] > 1e-10:
                    neighbor_idx.append(int(idx))
            if len(neighbor_idx) == 0:
                local_profiles[i] = self.global_profile_
                continue
            neighbor_comps = []
            for idx in neighbor_idx:
                comp_values = X_filled[idx, self.comp_idx]
                s = comp_values.sum()
                if s > 0:
                    neighbor_comps.append(comp_values / s)
            if len(neighbor_comps) == 0:
                local_profiles[i] = self.global_profile_
                continue
            lp = np.median(np.array(neighbor_comps), axis=0)
            lp = np.maximum(lp, self.min_positive)
            local_profiles[i] = lp / lp.sum()
        return local_profiles


def sparse_composition(rng, n, d, total=100.0):
    X = np.zeros((n, d))
    for i in range(n):
        k = int(rng.integers(3, min(6, d) + 1))
        cols = rng.choice(d, size=k, replace=False)
        X[i, cols] = rng.dirichlet(np.ones(k)) * total
    return X / X.sum(axis=1, keepdims=True) * total


print("=== 数值一致性（向量化 vs 三层循环）===")
all_same = True
for tag, n, d in [("synA1-like", 25, 50), ("glass-like", 40, 8), ("ge-like", 86, 9), ("steel-like", 300, 13)]:
    for k in [3, 5]:
        rng = np.random.default_rng(hash((tag, k)) % 2**31)
        X = sparse_composition(rng, n, d)
        miss = make_missing(X, 0.2, "MCAR", seed=1)
        Xm = X.copy()
        Xm[miss] = np.nan
        a = LCRISPOld(comp_idx=list(range(d)), n_neighbors=k, total=100.0).fit_transform(Xm)
        b = LCRISPImputer(comp_idx=list(range(d)), n_neighbors=k, total=100.0).fit_transform(Xm)
        same = np.array_equal(a, b)
        all_same &= same
        print(f"  {tag:<12} n={n:<4} d={d:<3} k={k}  identical={same}  max|diff|={np.max(np.abs(a-b)):.3e}")
print("全部逐位一致" if all_same else "**存在差异**")

print("\n=== 速度（n=1030, d=7）===")
rng = np.random.default_rng(0)
Xbig = sparse_composition(rng, 1030, 7)
miss = make_missing(Xbig, 0.2, "MCAR", seed=1)
Xmb = Xbig.copy()
Xmb[miss] = np.nan
t0 = time.time()
LCRISPImputer(comp_idx=list(range(7)), n_neighbors=5, total=100.0).fit_transform(Xmb)
t_new = time.time() - t0
print(f"  向量化: {t_new:.2f} s")
t0 = time.time()
LCRISPOld(comp_idx=list(range(7)), n_neighbors=5, total=100.0).fit_transform(Xmb)
t_old = time.time() - t0
print(f"  旧循环: {t_old:.2f} s   加速 {t_old/max(t_new,1e-9):.0f}×")
