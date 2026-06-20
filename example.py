"""Example: CRISP / LCRISP on synthetic alloy data."""
import numpy as np
from crisp import crisp, lcrisp, auto_crisp, check_compositional_validity, compositional_mae

# Generate synthetic steel-like compositions (Fe dominant, trace elements)
np.random.seed(42)
n_samples = 100
n_elements = 5  # Fe, C, Mn, Cr, Ni

# Fe is dominant (~70%), others are minor
X = np.random.dirichlet(np.ones(n_elements) * 0.5, size=n_samples) * 30.0
Fe = 100.0 - X.sum(axis=1, keepdims=True)
Fe = np.clip(Fe, 0.0, 100.0)
X = np.concatenate([X, Fe], axis=1)
row_sums = X.sum(axis=1, keepdims=True)
X = X / row_sums * 100.0

print("Original data shape:", X.shape)
print("First 3 samples:")
print(X[:3])

# Introduce 10% missing values
missing_rate = 0.10
n_missing = int(X.size * missing_rate)
missing_indices = np.random.choice(X.size, n_missing, replace=False)
X_missing = X.copy()
X_missing.flat[missing_indices] = np.nan

print(f"\nMissing rate: {missing_rate:.0%}")
print("Missing positions (first 5):")
print(np.argwhere(np.isnan(X_missing))[:5])

# Impute with CRISP
X_crisp = crisp(X_missing, comp_idx=list(range(6)), total=100.0)
valid_crisp = check_compositional_validity(X_crisp, comp_idx=list(range(6)), total=100.0)
print("\n--- CRISP Results ---")
print(f"Sum deviation: {valid_crisp['sum_dev_mean']:.2e}")
print(f"Valid: {valid_crisp['valid']}")

# Impute with LCRISP
X_lcrisp = lcrisp(X_missing, comp_idx=list(range(6)), total=100.0, n_neighbors=5)
valid_lcrisp = check_compositional_validity(X_lcrisp, comp_idx=list(range(6)), total=100.0)
print("\n--- LCRISP Results ---")
print(f"Sum deviation: {valid_lcrisp['sum_dev_mean']:.2e}")
print(f"Valid: {valid_lcrisp['valid']}")

# Compare accuracy (only on missing positions)
mask = np.isnan(X_missing)
if mask.any():
    mae_crisp = compositional_mae(X, X_crisp, mask)
    mae_lcrisp = compositional_mae(X, X_lcrisp, mask)
    print(f"\n--- Accuracy (MAE on missing positions) ---")
    print(f"CRISP MAE:  {mae_crisp['overall_mae']:.4f}")
    print(f"LCRISP MAE: {mae_lcrisp['overall_mae']:.4f}")

print("\n--- First 3 imputed samples (CRISP) ---")
print(X_crisp[:3].round(2))
print("\n--- First 3 imputed samples (LCRISP) ---")
print(X_lcrisp[:3].round(2))
