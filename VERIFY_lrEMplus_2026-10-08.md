# lrEMplus 基线核实记录（2026-10-08）

> 触发原因：`paper_draft.md` 的 Abstract 与 §3.2 声称「The standard compositional zero-package
> `zCompositions` cannot run on mixed structural-zero/NA data at all」。投稿前必须核实此声称。

## 一、结论（先行）

**该声称不成立，必须改写。** 且**论文完全没有把 `lrEMplus` 纳入基线**——而它正是官方为
"零与缺失同时存在"这一场景设计的工具。

## 二、官方依据

`zCompositions` v1.6.2（本机实测已安装），函数清单：

```
cmultRepl | lrDA | lrEM | lrEMplus | lrSVD | lrSVDplus | multKM | multLN |
multRepl | multReplus | perLog | splineKM | zPatterns
```

CRAN 官方文档对 `lrEMplus` 的描述（原文）：

> This function implements an extended version of the log-ratio EM algorithm (`lrEM` function)
> to **simultaneously deal with both zeros** (i.e. data below detection limit, rounded zeros)
> **and missing data** in compositional data sets.
> **Note: zeros and missing data must be labelled using 0 and `NA` respectively to use this function.**

文档示例矩阵中同一行同时出现 `0.00` 与 `NA`，并注明 "zeros and missing in the same row or
column are allowed"。

→ 编码约定（结构零 = 0、缺失 = NA）与本项目 `X_missing.csv` **完全一致**。
→ 来源：<https://search.r-project.org/CRAN/refmans/zCompositions/html/lrEMplus.html>

## 三、本机实测结果

脚本：`verify_lrEMplus_baseline.R`（本目录）。R 4.6.1 / zCompositions 1.6.2。
数据：项目自带 `tmp/<dataset>_MCAR_10/{X_missing.csv, X_true.csv}`。
MAE 仅在"真值非零且被标为 NA"的位置上计算。

| 数据 | n×d | 结构零占比 | 被插补数 | lrEMplus 结果 |
|---|---|---|---|---|
| `ge` | 86×9 | 423/761 ≈ 56% | 38 | **失败**：`lrEM based on alr requires at least one complete column` |
| `cement` | 1030×7 | 1411 | 588 | **失败**：同上 |
| `glass` | 40×8 | 68 | 24 | **成功**，残留 NA = 0，**MAE ≈ 1.428**（默认）/ **1.402**（`ini.cov="multRepl"`） |

对照 CRISP 在 glass 同配置下的 MAE = **0.073**（见 `wilcoxon_v3.csv`）。
→ **CRISP 在 glass 上比 lrEMplus 好约 20 倍** —— 这个头对头数字论文里目前完全没有。

`cmultRepl` 单独吃含 NA 的矩阵确实报错（`NA values not labelled as count zeros were found`），
但这只说明"不能只调 cmultRepl"，**不能推出"zCompositions 无法处理混合数据"**。
`impRZilr` 不在 v1.6.2 中（v1.6 已由 `lrEMplus`/`lrSVDplus`/`multReplus` 取代）。

## 四、对论文的修改建议（按重要性）

1. **删掉/改写核心卖点**。不能再写 "zCompositions cannot run at all"。改为措辞更强、且可复现的
   真命题：

   > 官方为"零+缺失"设计的 `lrEMplus` 依赖 alr 变换，**要求至少存在一个完整列**
   > （该列既无零也无缺失）。在结构零占比高的材料配方数据中（ge 56%），**没有任何一列是完整的**，
   > `lrEMplus` 直接报错无法启动；在存在完整列的数据（glass）上它可以运行，但会把结构零当作
   > *删失值*（低于检出限）替换为正值——对"该组分未添加"这一真实信息注入伪影。

2. **补 `lrEMplus` 为正式基线**。这是本方向真正的最新 SOTA，不补一定会被 CoDA 背景的审稿人要求补。
   补上之后，glass 上的 20× 差距是论文最有力的一行数字。

3. **修正 n<d 真实数据的比较公平性（第二大风险）**。`final_results_real_nd.csv` 中 CRISP 在
   ge/glass/cement 上为 MAE = 0.000 ± 0.000，论文 §3.2 已承认这些子集中
   100%（glass/cement）、89%（steel）的缺失行只有**单个缺失组分**——此时
   `x̂ = 100 − Σ已知` 是**算术恒等式**，任何施加了闭合约束的方法都会得到 MAE = 0。
   因此该表**不能**用来论证"插补精度更优"。必须补一个 **"基线 + 投影到单纯形"** 的对照组，
   把差异归因到真正的多缺失行上。

4. **补 `ge_nd_MNAR_10` 缺失的统计量**（`wilcoxon_nd_v3.csv` 中 p 值为空，需补跑或说明原因）。

## 五、附：本目录下未落盘的操作

`test_zcomp.R` / `test_zcomp2.R` 记录了对 zCompositions 的探索，但**输出未保存**，
且 `test_zcomp2.R` 中 "完整标准管线" 的注释把 `cmultRepl → lrEM` 写成了官方管线——
实际官方管线是 **`lrEMplus`**（单函数同时处理两类缺失）。建议一并整理。
