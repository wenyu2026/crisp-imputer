# data/ — 数据集来源与许可

本目录随仓库提供 CRISP 基准所用的一切数据，clone 后无需额外下载即可复现 `results/`。

| 文件 | 内容 | 来源 | 许可 |
|---|---|---|---|
| `steel_strength.csv` | 钢的抗拉强度（312 行 × 14 组成列 + 目标）；本基准实际使用前 300 行 | [Matminer](https://hackingmaterials.lbl.gov/matminer/) 内置 `steel_strength` 数据集 | MIT / 依 Matminer |
| `glass_40samples.csv` | 玻璃折射率（40 行 × 8 组成列） | Matminer 内置 `glass_ur` 子集 | 同上 |
| `ge_refractory_alloy.csv` | Ge 基难熔合金硬度（86 行 × 9 个 at% 列，55% 结构零） | 随项目提供 | 随项目 |
| `cement_concrete.xls`（位于仓库根目录） | UCI Concrete Compressive Strength（1030 行 × 7 组成列） | [UCI ML Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | CC BY 4.0 |

## 预处理

所有组成列按行归一化到总和 100（`X / X.sum(axis=1, keepdims=True) * 100`），与
`final_compare_v3.py: load_real()` / `load_cement()` 一致。

## 引用

若使用这些数据，请引用其原始出处（Matminer、UCI）以及本仓库。
