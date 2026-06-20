# CRISP: Compositional Ratio-based Imputation with Simplex Projection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **CRISP** is a fast, constraint-guaranteed imputation method for **compositional data** — e.g. alloy compositions, glass formulations, cement recipes — where all components must sum to a fixed total (e.g. 100%).

## 🔑 Key Features

- **Strict constraint satisfaction**: Sum deviation = 0 (machine precision), non-negative guaranteed
- **No covariance estimation**: Only needs column medians — robust for **n < d** (small samples, many features)
- **No log-ratio transforms**: Works natively with zeros (no pseudo-counts needed)
- **Three variants**:
  - **CRISP** (global profile) — simple & fast, best for low missing rates (< 15%)
  - **LCRISP** (local k-NN profile) — adaptive, best for high missing rates (>= 15%)
  - **AutoCRISP** — automatically selects between CRISP and LCRISP
- **Single-pass, no iteration**: O(n·d) complexity, orders of magnitude faster than iterative methods

## ⚡ Quick Start

```python
import numpy as np
from crisp import crisp, lcrisp, auto_crisp, check_compositional_validity

# Example: 3 alloy compositions, missing Fe and C
X = np.array([
    [70.0, 15.0, 15.0],   # Fe, C, Mn — complete
    [np.nan, 20.0, 10.0], # Fe missing
    [65.0, np.nan, 20.0], # C missing
])

# CRISP: global profile (fast, simple)
X_crisp = crisp(X, comp_idx=[0, 1, 2], total=100.0)
print(X_crisp)
# [[70.0  15.0  15.0]
#  [70.0  20.0  10.0]   <- Fe allocated from global profile
#  [65.0  15.0  20.0]]  <- C allocated from global profile

# Check constraints
valid = check_compositional_validity(X_crisp, comp_idx=[0, 1, 2], total=100.0)
print(valid["sum_dev_mean"])  # ~1e-15 (machine precision zero)

# LCRISP: local k-NN profile (for high missing rates)
X_lcrisp = lcrisp(X, comp_idx=[0, 1, 2], total=100.0, n_neighbors=2)

# AutoCRISP: let the algorithm decide
X_auto = auto_crisp(X, comp_idx=[0, 1, 2], total=100.0)
```

## 📖 When to use which?

| Variant | Missing rate | Speed | Use case |
|---------|-------------|-------|----------|
| **CRISP** | < 15% | ⚡ Fastest | Default choice; simple & robust |
| **LCRISP** | >= 15% | ⚡ Fast | When global profile is unreliable |
| **AutoCRISP** | Unknown | ⚡ Fast | When you don't want to decide |

## 🧪 Validation Results

Tested on 6 synthetic material datasets + MatMiner `steel_strength` (312 real samples, 14 compositional features):

| Dataset | Missing rate | CRISP MAE | LCRISP MAE | MICE MAE | SumDev (CRISP) |
|---------|-------------|----------|----------|----------|--------------|
| Steel (real) | 5% | 0.306 | **0.088** | 0.255 | **~0** |
| Steel (real) | 10% | 0.605 | **0.239** | 0.486 | **~0** |
| Steel (real) | 20% | 0.934 | **0.508** | 0.704 | **~0** |

> **Note**: MICE, KNN, MissForest violate constraints (SumDev = 0.3–3.3%). CRISP/LCRISP guarantee SumDev = 0.

## 🔧 Installation

```bash
pip install numpy pandas scikit-learn
```

Then copy `crisp.py` to your project directory.

## 📚 API Reference

### `crisp(X, comp_idx, total=100.0)` — Global profile
```python
crisp(X, comp_idx=[0,1,2], total=100.0)
```

### `lcrisp(X, comp_idx, n_neighbors=5, total=100.0)` — Local k-NN profile
```python
lcrisp(X, comp_idx=[0,1,2], n_neighbors=5, total=100.0)
```

### `auto_crisp(X, comp_idx, threshold=0.15, total=100.0)` — Adaptive switch
```python
auto_crisp(X, comp_idx=[0,1,2], threshold=0.15, total=100.0)
```

### `project_to_simplex(x, total=100.0)` — Post-hoc constraint projection
```python
project_to_simplex([70, 15, 15], total=100.0)  # [70.0, 15.0, 15.0]
```

### `check_compositional_validity(X, comp_idx, total=100.0)` — Verify constraints
```python
check_compositional_validity(X_imp, comp_idx=[0,1,2], total=100.0)
# Returns: {'sum_dev_mean': 0.0, 'valid': True, ...}
```

## 📝 Citation

If you use CRISP in your research, please cite:

```bibtex
@software{crisp_imputer,
  title = {CRISP: Compositional Ratio-based Imputation with Simplex Projection},
  author = {wenyu2026},
  year = {2026},
  url = {https://github.com/wenyu2026/crisp-imputer}
}
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

CRISP was developed as part of the [SmallMatPrep](https://github.com/wenyu2026/smallmatprep) toolkit for small-sample materials data preprocessing.
