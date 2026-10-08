"""CRISP: Compositional Ratio-based Imputation with Simplex Projection.

CRISP       (Global profile):   best for low missing rates (< 15%).
LCRISP      (Local k-NN):       best for high missing rates (>= 15%).
AutoCRISP   (Adaptive switch):  automatically selects between CRISP and LCRISP.

All variants guarantee strict simplex constraints (sum = total, non-negative).
"""

import warnings

import numpy as np
from typing import List, Optional

__all__ = [
    "CRISPImputer",
    "LCRISPImputer",
    "AutoCRISPImputer",
    "crisp",
    "lcrisp",
    "auto_crisp",
    "project_to_simplex",
    "check_compositional_validity",
    "compositional_mae",
]

__version__ = "1.1.0"


# ============================================================================
# Utility functions
# ============================================================================

def _quiet_nanmean(a, axis=0):
    """`np.nanmean` without the "Mean of empty slice" RuntimeWarning on all-NaN slices."""
    a = np.asarray(a, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _quiet_nanmedian(a, axis=0):
    """`np.nanmedian` without the "All-NaN slice encountered" RuntimeWarning."""
    a = np.asarray(a, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)

def project_to_simplex(
    x: np.ndarray,
    total: float = 100.0,
    feature_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Project a vector onto the simplex (sum=total, non-negative).
    
    Two-step projection:
    1. Non-negative projection: x_j^+ = max(0, x_j)
    2. Normalization: y_j = x_j^+ / sum(x^+) * total
    """
    x = np.asarray(x, dtype=float).copy()
    if x.ndim == 1:
        x = x.reshape(1, -1)
        return_1d = True
    else:
        return_1d = False
    if feature_indices is None:
        feature_indices = list(range(x.shape[1]))
    for i in range(x.shape[0]):
        comp = x[i, feature_indices].copy()
        comp = np.maximum(comp, 0.0)
        comp_sum = comp.sum()
        if comp_sum > 0:
            comp = comp / comp_sum * total
        else:
            comp = np.ones_like(comp) * total / len(feature_indices)
        x[i, feature_indices] = comp
    return x[0] if return_1d else x


def check_compositional_validity(
    X: np.ndarray,
    comp_idx: List[int],
    total: float = 100.0,
    tol: float = 1e-6,
) -> dict:
    """Check if data satisfies compositional constraints."""
    X = np.asarray(X, dtype=float)
    comp_sum = X[:, comp_idx].sum(axis=1)
    sum_dev = np.abs(comp_sum - total)
    neg_count = (X[:, comp_idx] < -tol).sum()
    neg_pct = neg_count / (X.shape[0] * len(comp_idx)) * 100
    return {
        "sum_dev_mean": float(sum_dev.mean()),
        "sum_dev_max": float(sum_dev.max()),
        "sum_violation_pct": float((sum_dev > tol).mean() * 100),
        "neg_count": int(neg_count),
        "neg_violation_pct": float(neg_pct),
        "valid": bool(sum_dev.mean() < tol and neg_count == 0),
    }


def compositional_mae(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    missing_mask: np.ndarray,
    comp_idx: Optional[List[int]] = None,
) -> dict:
    """Compute MAE for compositional data."""
    X_true = np.asarray(X_true, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    missing_mask = np.asarray(missing_mask, dtype=bool)
    if comp_idx is None:
        comp_idx = list(range(X_true.shape[1]))
    overall_mae = np.mean(np.abs(X_true[missing_mask] - X_pred[missing_mask]))
    comp_missing = missing_mask.copy()
    comp_mae = 0.0
    if comp_missing.shape[1] == X_true.shape[1]:
        comp_mae = np.mean(
            np.abs(X_true[:, comp_idx][comp_missing[:, comp_idx]] - X_pred[:, comp_idx][comp_missing[:, comp_idx]])
        ) if comp_missing[:, comp_idx].any() else 0.0
    per_comp = {}
    for j in comp_idx:
        if missing_mask[:, j].any():
            per_comp[j] = float(np.mean(np.abs(X_true[missing_mask[:, j], j] - X_pred[missing_mask[:, j], j])))
    return {
        "overall_mae": float(overall_mae),
        "comp_mae": float(comp_mae),
        "per_component_mae": per_comp,
    }


# ============================================================================
# Base class (shared logic)
# ============================================================================

class _BaseCRISPImputer:
    """Base class with shared CRISP logic."""
    
    def __init__(
        self,
        comp_idx: List[int],
        non_comp_idx: Optional[List[int]] = None,
        total: float = 100.0,
        min_positive: float = 1e-6,
    ):
        self.comp_idx = list(comp_idx)
        self.non_comp_idx = list(non_comp_idx) if non_comp_idx else []
        self.total = total
        self.min_positive = min_positive
        self.n_comp_ = len(self.comp_idx)

    def _project_sample(self, X_filled: np.ndarray, i: int) -> None:
        """Project one sample's compositional features to simplex."""
        row_sum = sum(X_filled[i, self.comp_idx[j]] for j in range(self.n_comp_))
        if row_sum > 0:
            for j in range(self.n_comp_):
                X_filled[i, self.comp_idx[j]] = X_filled[i, self.comp_idx[j]] / row_sum * self.total

    def _column_mean_init(self, X: np.ndarray) -> np.ndarray:
        """Initialise missing values in *compositional* columns with column means.

        Columns outside `comp_idx` are not part of the composition and CRISP does not
        impute them, so a NaN there is left untouched rather than silently replaced by
        a column mean. (Changed in 1.1.0: earlier versions filled every column.)
        """
        X_filled = np.array(X, dtype=float, copy=True)
        col_means = _quiet_nanmean(X, axis=0)
        for j in self.comp_idx:
            col = X_filled[:, j]
            mask = np.isnan(col)
            if mask.any() and not np.isnan(col_means[j]):
                col[mask] = col_means[j]
        return X_filled

    def _estimate_global_profile(self, X_missing: np.ndarray) -> np.ndarray:
        """Estimate global profile from complete samples."""
        n_comp = self.n_comp_
        complete_rows = ~np.isnan(X_missing).any(axis=1)
        
        # Also accept "nearly complete" rows (<=1 missing) to increase robustness
        nearly_complete = (~complete_rows) & (np.isnan(X_missing).sum(axis=1) <= 1)
        usable_rows = complete_rows | nearly_complete
        
        if np.sum(usable_rows) >= 3:
            X_usable = X_missing[usable_rows][:, self.comp_idx]
            # For nearly-complete rows, fill the single missing with column mean
            for col in range(n_comp):
                col_mean = _quiet_nanmean(X_usable[:, col], axis=0)
                mask_nan = np.isnan(X_usable[:, col])
                X_usable[mask_nan, col] = col_mean if not np.isnan(col_mean) else self.min_positive
            
            row_sums = X_usable.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums <= 0, self.min_positive, row_sums)
            X_norm = X_usable / row_sums
            profile = np.median(X_norm, axis=0)
            profile = np.maximum(profile, self.min_positive)
            profile = profile / profile.sum()
        else:
            # Fallback: use column median ratios (more informative than uniform)
            col_medians = _quiet_nanmedian(X_missing[:, self.comp_idx], axis=0)
            col_medians = np.maximum(col_medians, self.min_positive)
            if np.any(np.isnan(col_medians)):
                profile = np.ones(n_comp) / n_comp
            else:
                profile = col_medians / col_medians.sum()
        return profile

    def _estimate_local_profiles(self, X_filled: np.ndarray, X_missing: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Estimate a local k-NN profile for every sample.

        The distance computation is vectorised over candidate neighbours and features.
        The original implementation used a triply nested Python loop
        (`for i: for j: for f:`), which made LCRISP impractical beyond a few hundred
        rows — at n = 1030 a single call took ~10 s.
        """
        n_samples = X_filled.shape[0]
        n_comp = self.n_comp_
        local_profiles = np.empty((n_samples, n_comp))

        comp_vals = X_filled[:, self.comp_idx]
        comp_sum = comp_vals.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = comp_vals / comp_sum[:, None]
        normalized[~np.isfinite(normalized)] = 0.0
        comp_filled = np.nan_to_num(comp_vals, nan=0.0)
        obs_comp = ~np.isnan(X_missing[:, self.comp_idx])

        if len(self.non_comp_idx) > 0:
            feats = list(self.non_comp_idx)
            vals_f = np.nan_to_num(X_filled[:, feats], nan=0.0)
            obs_f = ~np.isnan(X_missing[:, feats])
        else:
            feats = vals_f = obs_f = None

        for i in range(n_samples):
            if feats is not None:
                joint = obs_f & obs_f[i][None, :]
                vals, vi = vals_f, vals_f[i]
            else:
                known = obs_comp[i]
                if not known.any():
                    local_profiles[i] = self.global_profile_
                    continue
                joint = obs_comp & known[None, :]
                vals, vi = comp_filled, comp_filled[i]

            denom = joint.sum(axis=1)
            with np.errstate(invalid="ignore"):
                num = np.where(joint, (vals - vi[None, :]) ** 2, 0.0).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                distances = np.where(denom > 0, num / np.maximum(denom, 1), np.inf)

            order = np.argsort(distances)
            usable = np.isfinite(distances[order]) & (distances[order] > 1e-10)
            candidates = order[usable][:n_neighbors]

            if candidates.size == 0:
                local_profiles[i] = self.global_profile_
                continue

            keep = comp_sum[candidates] > 0
            if not keep.any():
                local_profiles[i] = self.global_profile_
                continue

            local_profile = np.median(normalized[candidates[keep]], axis=0)
            local_profile = np.maximum(local_profile, self.min_positive)
            local_profiles[i] = local_profile / local_profile.sum()

        return local_profiles

    def _allocate_missing(self, X_filled: np.ndarray, X_missing: np.ndarray, i: int, profile: np.ndarray) -> None:
        """Allocate missing values using a profile (global or local)."""
        missing_mask = np.isnan(X_missing[i, self.comp_idx])
        if not missing_mask.any():
            return
        
        known_mask = ~missing_mask
        if known_mask.any():
            known_idx = [self.comp_idx[j] for j in range(self.n_comp_) if known_mask[j]]
            missing_idx = [self.comp_idx[j] for j in range(self.n_comp_) if missing_mask[j]]
            known_sum = X_filled[i, known_idx].sum()
            remaining = self.total - known_sum
            
            if remaining <= 0:
                scale = self.total / known_sum
                for idx in known_idx:
                    X_filled[i, idx] *= scale
                for idx in missing_idx:
                    X_filled[i, idx] = 0.0
            else:
                missing_profile = np.array([profile[j] for j in range(self.n_comp_) if missing_mask[j]])
                if missing_profile.sum() > 0:
                    allocated = (missing_profile / missing_profile.sum()) * remaining
                    for idx, val in zip(missing_idx, allocated):
                        X_filled[i, idx] = val
        else:
            for j in range(self.n_comp_):
                X_filled[i, self.comp_idx[j]] = profile[j] * self.total
        
        self._project_sample(X_filled, i)


# ============================================================================
# CRISP: Global Profile
# ============================================================================

class CRISPImputer(_BaseCRISPImputer):
    """CRISP: Compositional Ratio-based Imputation with Simplex Projection - Global profile.
    
    Best for: low missing rates (< 15%).
    
    Algorithm:
    1. Column-mean initialization for missing values.
    2. Estimate global profile from complete samples (median of normalized ratios).
    3. Allocate missing values proportionally to the global profile.
    4. Project to simplex (sum = total, non-negative).
    
    Parameters
    ----------
    comp_idx : list of int
        Column indices of compositional features (must sum to `total`).
    non_comp_idx : list of int, optional
        Non-compositional feature indices (for distance calculation only).
    total : float, default=100.0
        Sum constraint for compositional features.
    """
    
    def __init__(
        self,
        comp_idx: List[int],
        non_comp_idx: Optional[List[int]] = None,
        total: float = 100.0,
    ):
        super().__init__(comp_idx, non_comp_idx, total)
    
    def fit(self, X: np.ndarray) -> "CRISPImputer":
        X = np.asarray(X, dtype=float)
        self.global_profile_ = self._estimate_global_profile(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_filled = self._column_mean_init(X)
        if not hasattr(self, 'global_profile_'):
            self.fit(X)
        for i in range(X_filled.shape[0]):
            self._allocate_missing(X_filled, X, i, self.global_profile_)
        return X_filled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================================
# LCRISP: Local Profile (k-NN)
# ============================================================================

class LCRISPImputer(_BaseCRISPImputer):
    """LCRISP: Local CRISP - k-NN local profile.
    
    Best for: high missing rates (>= 15%) or when global profile is unreliable.
    
    For each sample, find k nearest neighbors and estimate a local compositional
    profile (median of neighbor ratios). Then allocate missing values proportionally.
    
    Parameters
    ----------
    comp_idx : list of int
        Column indices of compositional features.
    non_comp_idx : list of int, optional
        Non-compositional feature indices for distance calculation.
    n_neighbors : int, default=5
        Number of neighbors for local profile estimation.
    total : float, default=100.0
        Sum constraint.
    """
    
    def __init__(
        self,
        comp_idx: List[int],
        non_comp_idx: Optional[List[int]] = None,
        n_neighbors: int = 5,
        total: float = 100.0,
    ):
        super().__init__(comp_idx, non_comp_idx, total)
        self.n_neighbors = n_neighbors
    
    def fit(self, X: np.ndarray) -> "LCRISPImputer":
        X = np.asarray(X, dtype=float)
        self.global_profile_ = self._estimate_global_profile(X)
        X_filled = self._column_mean_init(X)
        self.local_profiles_ = self._estimate_local_profiles(X_filled, X, self.n_neighbors)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_filled = self._column_mean_init(X)
        if not hasattr(self, 'local_profiles_'):
            self.fit(X)
        for i in range(X_filled.shape[0]):
            self._allocate_missing(X_filled, X, i, self.local_profiles_[i])
        return X_filled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================================
# AutoCRISP: Adaptive switch
# ============================================================================

class AutoCRISPImputer:
    """AutoCRISP: Automatically select between CRISP and LCRISP based on missing rate.
    
    Parameters
    ----------
    comp_idx : list of int
        Compositional feature indices.
    non_comp_idx : list of int, optional
        Non-compositional feature indices.
    threshold : float, default=0.15
        Missing rate threshold. Below threshold -> CRISP, above -> LCRISP.
    n_neighbors : int, default=5
        Number of neighbors for LCRISP (used only if selected).
    total : float, default=100.0
        Sum constraint.
    """
    
    def __init__(
        self,
        comp_idx: List[int],
        non_comp_idx: Optional[List[int]] = None,
        threshold: float = 0.15,
        n_neighbors: int = 5,
        total: float = 100.0,
    ):
        self.comp_idx = list(comp_idx)
        self.non_comp_idx = list(non_comp_idx) if non_comp_idx else []
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.total = total
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        missing_rate = np.isnan(X).sum() / X.size
        if missing_rate < self.threshold:
            return crisp(X, comp_idx=self.comp_idx, non_comp_idx=self.non_comp_idx, total=self.total)
        else:
            return lcrisp(X, comp_idx=self.comp_idx, non_comp_idx=self.non_comp_idx, n_neighbors=self.n_neighbors, total=self.total)


# ============================================================================
# Convenience functions
# ============================================================================

def crisp(
    X: np.ndarray,
    comp_idx: List[int],
    non_comp_idx: Optional[List[int]] = None,
    total: float = 100.0,
) -> np.ndarray:
    """CRISP - low missing rate optimal (global profile)."""
    imputer = CRISPImputer(comp_idx=comp_idx, non_comp_idx=non_comp_idx, total=total)
    return imputer.fit_transform(X)


def lcrisp(
    X: np.ndarray,
    comp_idx: List[int],
    non_comp_idx: Optional[List[int]] = None,
    n_neighbors: int = 5,
    total: float = 100.0,
) -> np.ndarray:
    """LCRISP - high missing rate survival (local k-NN profile)."""
    imputer = LCRISPImputer(comp_idx=comp_idx, non_comp_idx=non_comp_idx, n_neighbors=n_neighbors, total=total)
    return imputer.fit_transform(X)


def auto_crisp(
    X: np.ndarray,
    comp_idx: List[int],
    non_comp_idx: Optional[List[int]] = None,
    total: float = 100.0,
    threshold: float = 0.15,
) -> np.ndarray:
    """AutoCRISP - adaptive switch between CRISP and LCRISP."""
    imputer = AutoCRISPImputer(comp_idx=comp_idx, non_comp_idx=non_comp_idx, threshold=threshold, total=total)
    return imputer.fit_transform(X)
