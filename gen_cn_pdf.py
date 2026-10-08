# -*- coding: utf-8 -*-
"""gen_cn_pdf.py — 生成 CRISP 论文中文版 PDF（用于理解/给导师看/答辩）"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.lib import colors
import os

BASE = os.path.dirname(os.path.abspath(__file__))
pdfmetrics.registerFont(TTFont("Hei", "C:/Windows/Fonts/simhei.ttf"))

title_s = ParagraphStyle("t", fontName="Hei", fontSize=15, leading=20, alignment=TA_CENTER, spaceAfter=6)
sub_s = ParagraphStyle("st", fontName="Hei", fontSize=11, leading=16, alignment=TA_CENTER, spaceAfter=12)
h1 = ParagraphStyle("h1", fontName="Hei", fontSize=12, leading=16, spaceBefore=10, spaceAfter=4)
h2 = ParagraphStyle("h2", fontName="Hei", fontSize=10.5, leading=15, spaceBefore=6, spaceAfter=3)
body = ParagraphStyle("b", fontName="Hei", fontSize=9.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=3)
refs = ParagraphStyle("r", fontName="Hei", fontSize=8.5, leading=12, spaceAfter=2)

def P(txt, style=body):
    return Paragraph(txt, style)

def make_table(header, rows, colw=None):
    data = [header] + rows
    t = Table(data, colWidths=colw, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Hei"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efefef")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t

story = []
story.append(P("CRISP：面向稀疏与高维配方数据的单纯形约束成分比例插补", title_s))
story.append(P("Compositional Ratio-based Imputation with Simplex Projection for Sparse and High-Dimensional Formulation Data", sub_s))
story.append(P("（中文版，用于理解与展示；投稿版为英文）", sub_s))

# ---------------- 摘要 ----------------
story.append(P("摘  要", h1))
story.append(P(
    "材料配方数据（合金、玻璃、水泥、电解液）是成分数据：每个样本是各组分比例的向量，总和固定；"
    "配方中刻意不添加的组分表现为结构零。实践中这类矩阵包含两种不同的缺失信息——结构零（某组分未添加）"
    "与缺失值（记录丢失）——而标准的插补方法都无法同时处理两者：对数比变换在零值处崩溃（log 0），"
    "朴素方法则违反单纯形约束、产生物理上不可行的配方。", body))
story.append(P(
    "我们提出 CRISP（成分比例插补 + 单纯形投影），一类约束保证的插补器，显式区分结构零与缺失值。"
    "缺失项按完整行估计的配比轮廓（归一化比例的中位数）比例分配，结果构造性满足单纯形约束。"
    "我们证明该分配是 Dirichlet 条件均值的 plug-in 经验贝叶斯估计（单纯形上信息量最少的补法），"
    "并给出插补误差上界：可收缩的轮廓项 O(M<sup>−1/2</sup>) + 不可约的数据不确定性项。", body))
story.append(P(
    "方法的主战场是材料数据库常见的 n&lt;d 高维小样本配方数据。在合成 n&lt;d 实验中，CRISP 显著降低插补误差"
    "（Wilcoxon 配对检验 p&lt;0.01，多数配置；误差低 2.0–2.5 倍），并显著优于低秩矩阵补全 SoftImpute"
    "（其在成分数据上失败，因为低秩假设被结构零与闭合约束破坏）。在真实材料 n&lt;d 子集（玻璃、水泥、钢、合金）上，"
    "CRISP 显著优于 MICE（p&lt;0.01，各 20 个子集）。关键机制是 CRISP 利用闭合约束：当一行仅一个缺失组分时，"
    "CRISP 精确恢复它（x̂ = 100 − Σ已知），而回归插补器忽略这条信息。基线违反单纯形约束"
    "（SumDev 最高 14）而 CRISP 到机器精度满足。在低缺失率稠密数据上，CRISP 与 MICE 精度在 5 个数据集的 3 个上"
    "无统计显著差异（2 个上略差），但严格保证 SumDev=0、零参数、n&lt;d 可用。标准成分零处理包 zCompositions "
    "完全无法运行于结构零+缺失混合数据。", body))
story.append(P("关键词：成分数据；结构零；插补；单纯形约束；n&lt;d；小样本；材料信息学", body))
story.append(Spacer(1, 8))

# ---------------- 1 引言 ----------------
story.append(P("1  引言", h1))
story.append(P(
    "材料配方优化——合金设计、玻璃配方、水泥与电解液配方——把每个候选样本视为组分分数向量。"
    "两个数据质量问题主导真实数据集：（1）结构零：配方刻意省略某些组分，0 是真实信息（未添加）而非缺失测量，"
    "在我们的合金数据中 55–86% 的组分项是结构零；（2）缺失值：个别记录丢失，产生需插补的 NA。", body))
story.append(P(
    "两者的并存是材料数据库的常态，但标准方法都处理不好。对数比变换方法（Aitchison 距离的 impKNNa、impCoda、"
    "zCompositions::lrEM/lrSVD）要求严格正数；结构零迫使伪计数，把未添加变成微量存在。这些方法的 Aitchison 模式"
    "在含零数据上直接崩溃（log 0 / NA 不允许）。因此我们用官方基线 impKNNa 的欧氏模式（把 0 当普通数值）"
    "作为可运行基线，但它违反单纯形约束。朴素/回归/低秩插补（KNN、均值、MICE、SoftImpute）不做闭合步骤，"
    "产出违反约束的矩阵。违反在 CRISP 约束保证真正重要的场景最严重：n&lt;d 数据与高缺失率"
    "（SumDev 7–8；50–70% 缺失率时基线最高 24–55）。", body))
story.append(P("主要贡献：（1）显式区分结构零与缺失——零保留，只插补缺失；（2）构造性单纯形保证——SumDev=0 是设计出来而非后处理投影；（3）plug-in 经验贝叶斯解释——分配匹配 Dirichlet 条件均值（默认中位数轮廓，亦支持均值/几何均值），带可证明的误差上界；（4）n&lt;d 能力——只需列中位数，无协方差、无调参，是回归（MICE）与低秩（SoftImpute）插补器都退化的场景。", body))

# ---------------- 2 方法 ----------------
story.append(P("2  方法", h1))
story.append(P("2.1 记号", h2))
story.append(P(
    "设 S = { x ∈ ℝ₊ᵈ : Σⱼxⱼ = 100 } 为成分单纯形。完整成分 x/100 ~ Dirichlet(α)。样本 i 的已知部分为 x_K，"
    "缺失部分为 x_M，剩余量 R = 100 − Σ_{k∈K} x_k。C 为完整行集合，M = |C|。", body))
story.append(P("2.2 CRISP 算法", h2))
story.append(P(
    "轮廓 p̂ 是完整行归一化比例的逐列中位数。缺失项按比例分配：", body))
story.append(P("x̂ⱼ = ( p̂ⱼ / Σ_{m∈M} p̂ₘ ) · R ，   ∀ j ∈ M 。", body))
story.append(P(
    "结构零从不被改动。实现还提供 LCRISP（k 近邻局部轮廓）与 AutoCRISP（自适应切换）变体；"
    "本文仅评估基础 CRISP，变体的实证比较留待后续。复杂度为 O(d·M)（轮廓）与 O(d)（每样本）。", body))
story.append(P("2.3 理论", h2))
story.append(P(
    "引理 1（轮廓一致性）：在 MCAR 且 x/100 ~ Dirichlet(α) 下，均值与几何均值轮廓收敛到 α/Σα，"
    "且 E‖p̂ − α/Σα‖₁ = O(M^{−1/2})（大数定律/中心极限定理；几何均值用 Dirichlet 的 digamma 恒等式）。"
    "中位数轮廓以同速率收敛到总体中位数，仅当 Dirichlet 分量满足对称性条件时等于 α/Σα；"
    "实践中三种轮廓的插补 MAE 差距 &lt;4%（消融见 §3.2）。我们默认保留中位数（鲁棒性，命题 5），"
    "并提供几何均值轮廓作为可选变体。", body))
story.append(P(
    "定理 2（插补误差上界）：对缺失坐标 j，E|x̂ⱼ − xⱼ*| ≤ O(M^{−1/2}·R) + √(2/π)·sd(xⱼ|x_K)。"
    "第一项随完整行收缩；第二项是不可约的条件不确定性（√(2/π) 假设正态条件残差，一般分布用 Jensen 界 ≤ sd）。", body))
story.append(P(
    "plug-in 说明：CRISP 分配等于 Dirichlet 条件均值这一贝叶斯解读，对均值/几何均值轮廓严格成立；"
    "对中位数轮廓需对称性条件，否则中位数收敛到总体中位数、等式近似成立。所有情况下轮廓都是 plug-in 估计，"
    "而非完整后验计算。完整陈述与证明见补充材料（PROOFS）。", body))
story.append(P(
    "命题 5（维度鲁棒性，经验性）：CRISP 轮廓逐列估计（中位数），无跨维度回归，故对任意 d/n 良定义。"
    "回归型插补（MICE）每步拟合 xⱼ ~ X_{−j}：n&lt;d 时无正则最小二乘不可识别，正则化实现的外推误差随 d/n 增大——"
    "与 n&lt;d 实验中 MICE 误差急剧退化而 CRISP 保持平坦一致。", body))
story.append(P("2.4 实验设计", h2))
story.append(P(
    "低缺失率数据集（5 个，四个材料体系）：ge 耐火合金（86×9，55% 零）、玻璃（40×8，21%）、"
    "合成（300×8，42%）、钢（312×13，17%）、水泥（UCI，1030×7，20%）。"
    "n&lt;d 数据集：合成 n∈{20,25,30}、d∈{40,50,60}；以及钢（n=10,d=13）、玻璃（n=6,d=8）、"
    "水泥（n=6,d=7）、ge（n=8,d=9）的真实 n&lt;d 子集，各 20 个随机子集。"
    "缺失机制：MCAR/MAR/MNAR，低缺失率 10%/20%、n&lt;d 10%/30%。", body))
story.append(P(
    "关键协议细节：缺失只在非零项上施加——结构零从不被标记为缺失。所有方法收到同一缺失掩码，"
    "结构零不被任何方法改动，故评估隔离了缺失值插补任务。", body))
story.append(P(
    "基线：官方 robCompositions::impKNNa（欧氏模式）、R missForest、scikit-learn IterativeImputer（MICE）、"
    "KNNImputer、列均值、SoftImpute（低秩矩阵补全，秩 5 SVD 阈值化；秩 2/5/10 敏感性确认其在成分数据上全失败）。"
    "Aitchison/impCoda/lrEM/zCompositions 报告为无法运行于混合零/NA 数据。", body))
story.append(P(
    "指标：缺失项 MAE（Python 方法 20 个随机缺失种子，均值±标准差；R 基线单种子并排除于配对检验之外）、"
    "SumDev（单纯形违反）、CRISP 对每个 Python 基线的配对 Wilcoxon 符号秩检验。", body))

# ---------------- 3 结果 ----------------
story.append(P("3  结果", h1))
story.append(P("3.1 低缺失率数据（20 种子均值；标准差见附录）", h2))
story.append(P("表 1. 缺失项 MAE，MCAR 10%。", body))
story.append(make_table(
    ["数据集（零占比）", "CRISP", "MICE", "SoftImpute", "impKNNa", "KNN", "均值"],
    [["ge 合金（55%）", "2.44", "1.65", "29.3", "10.86", "10.99", "12.19"],
     ["玻璃（21%）", "0.073", "0.075", "0.31", "0.198", "0.143", "0.252"],
     ["合成（42%）", "4.29", "3.74", "38.4", "12.55", "11.38", "14.46"],
     ["钢（17%）", "1.51", "1.55", "5.88", "1.76", "1.89", "6.11"],
     ["水泥（20%）", "1.07", "0.57", "3.45", "0.59", "0.75", "2.26"]],
    colw=[2.6, 1.2, 1.2, 1.5, 1.4, 1.2, 1.2]))
story.append(Spacer(1, 4))
story.append(P(
    "诚实发现 1：低缺失率数据上，CRISP 在稀疏数据集（玻璃）最好或并列，而 MICE 在水泥/合成上中等程度更好"
    "（p&lt;0.001 / p=0.009 有利于 MICE），其余相当（ge p=0.11、钢 p=0.37、玻璃 p=0.60）。"
    "因此 CRISP 的原始精度与 MICE 在 5 个数据集的 3 个上无统计显著差异、在 2 个上略差——"
    "其差异化在于严格约束保证（SumDev=0）、结构零语义、零调参、n&lt;d 能力，而非精度。", body))

story.append(P("3.2 n&lt;d 场景（CRISP 的主要优势）", h2))
story.append(P("表 2. 合成 n&lt;d 数据，MCAR 10%（20 种子均值）。", body))
story.append(make_table(
    ["数据集", "CRISP", "MICE", "SoftImpute", "Wilcoxon(CRISP vs MICE)", "SumDev (CRISP/基线)"],
    [["n=25, d=50", "3.46", "8.39", "12.7", "p&lt;0.001", "0.003 / 8.2"],
     ["n=30, d=40", "2.97", "6.67", "10.8", "p&lt;0.001", "0.003 / 7–8"],
     ["n=20, d=40", "3.05", "7.74", "11.9", "p&lt;0.001", "0.003 / 7–8"],
     ["n=25, d=60", "3.14", "6.44", "8.7", "p&lt;0.001", "0.003 / 7–8"]],
    colw=[1.8, 1.0, 1.0, 1.5, 3.2, 2.2]))
story.append(Spacer(1, 4))
story.append(P(
    "关键发现 2：合成 n&lt;d 场景，CRISP 显著降低插补误差（配对 Wilcoxon p&lt;0.01 占 70% 配置，MCAR 10% 时 p&lt;0.001）"
    "并显著优于 SoftImpute（p&lt;0.001 全配置——低秩补全在成分数据上失败，误差高 3–12 倍）。"
    "基线违反单纯形约束（SumDev 7–8）而 CRISP 到机器精度满足。不显著的 30% 配置集中在真实扩展数据集（ge_nd）"
    "与高缺失率（30%）MAR/MNAR 情形，完整 p 值矩阵见附录。", body))
story.append(P(
    "真实材料 n&lt;d 证据（各 20 子集）：", body))
story.append(make_table(
    ["数据集（n&lt;d 子集）", "CRISP", "MICE", "Wilcoxon p"],
    [["玻璃 (n=6,d=8)", "0.000", "0.136", "p&lt;0.001"],
     ["水泥 (n=6,d=7)", "0.000", "1.807", "p&lt;0.001"],
     ["钢 (n=10,d=13)", "3.70", "5.71", "p=0.007"],
     ["ge (n=8,d=9)", "0.000", "9.76", "p&lt;0.001"]],
    colw=[4.2, 1.4, 1.4, 2.6]))
story.append(Spacer(1, 4))
story.append(P(
    "关键发现 3：四个真实材料数据的 n&lt;d 子集上，CRISP 全部显著优于 MICE（p&lt;0.01）。"
    "注意：这些子集从小池重叠抽样（玻璃仅 40 行，n=6 的 20 次抽样高度重叠），配对 Wilcoxon 的 p 值应视为指示性"
    "而非正式显著性；主要统计证据是合成 n&lt;d 实验（独立缺失抽取）与闭合恢复机制本身。", body))
story.append(P(
    "近精确恢复的机制：这些小样本子集中，缺失被单缺失组分行主导——玻璃/水泥子集 100% 的缺失行、钢 89%、"
    "合成 n&lt;d 40% 是单缺失组分。对单缺失组分行，CRISP 分配退化为 x̂ = R = 100 − Σ已知，即利用闭合约束"
    "（行和=100）精确恢复缺失项。这是玻璃/水泥达到 MAE 0.000 的原因——不是伪影，而是方法利用了"
    "回归插补器忽略的信息（行和已知）。我们把这种闭合约束恢复视为 CRISP 在 n&lt;d 场景的核心性质。", body))
story.append(P(
    "扩展真实高维数据：一个 ge 衍生的 n=20,d=30 数据集（86% 零）在 MCAR 10% 时未显示 CRISP 优势"
    "（MICE 3.87 vs CRISP 4.27，不显著）；n&lt;d 优势建立在小子集与合成池上，在更大真实高维数据上仍有待确认。"
    "我们不声称普遍的 n&lt;d 支配。", body))
story.append(P(
    "MAR/MNAR（低缺失率）：表 3 报告 10% 缺失率下的 MAR/MNAR。CRISP 在 ge/玻璃/合成的 MNAR/MAR 下最好或接近最好，"
    "并处处优于对数比基线；MICE 在水泥上仍具竞争力或更好。完整网格见附录。", body))
story.append(make_table(
    ["数据集", "机制", "CRISP", "MICE", "impKNNa"],
    [["ge(55%)", "MAR", "1.70", "1.50", "14.08"],
     ["ge(55%)", "MNAR", "0.51", "0.73", "10.35"],
     ["玻璃(21%)", "MAR", "0.062", "0.071", "0.174"],
     ["玻璃(21%)", "MNAR", "0.017", "0.017", "0.159"],
     ["合成(42%)", "MAR", "3.16", "3.29", "11.89"],
     ["合成(42%)", "MNAR", "2.71", "2.82", "10.68"],
     ["水泥(20%)", "MAR", "0.99", "0.55", "0.53"],
     ["水泥(20%)", "MNAR", "0.55", "0.32", "0.44"]],
    colw=[2.4, 1.4, 1.4, 1.4, 1.6]))
story.append(Spacer(1, 4))
story.append(P(
    "轮廓选择（消融）：合成 n&lt;d 数据上，几何均值轮廓（CoDA 标准中心，满足子组合一致性）比中位数略精确"
    "（MAE 3.28 vs 3.39；均值轮廓 3.97，差距 &lt;4%）。我们保留中位数出于鲁棒性：在人为污染一个完整行"
    "（某组分×10 后重新闭合）时，中位数轮廓的插补误差不变而均值轮廓退化。两种选择都保持闭合约束，"
    "定性结论不依赖轮廓选择；几何均值变体在发布代码中作为可选提供。", body))

story.append(P("3.3 约束保证", h2))
story.append(P(
    "CRISP 输出恒满足 Σx̂=100（SumDev=0，构造性）。基线违反：低缺失率下 MICE 0.0–0.3、n&lt;d 时 7–8；"
    "SoftImpute 与 impKNNa 0.2–7.8；KNN 0.1–8；均值 0.1–13。50–70% 缺失率时基线违反增至 24–55 而 CRISP 保持 0；"
    "我们诚实说明：稠密数据极端缺失率下 CRISP 原始精度中等（MICE 可能更准）——高缺失率不是 CRISP 声称的强项，"
    "其保证在那里是约束而非精度。", body))
story.append(P("3.4 结构零保留", h2))
story.append(P(
    "在共享协议下（缺失仅在非零项、结构零不被改动），所有方法都保留现有零；CRISP 的独特之处在于在可行分配内"
    "构造性保留它们，而非把零留在违反约束的矩阵中。", body))
story.append(P("3.5 理论验证", h2))
story.append(P(
    "轮廓误差随 M 递减（0.065→0.039）；插补 MAE 稳定在不可约项（≈9.1），与定理 2 一致。", body))
story.append(P("3.6 下游应用", h2))
story.append(P(
    "在 n&lt;d 数据 + 20% 缺失下，仅 CRISP 插补矩阵在下游代理模型中保持正的预测 R²（0.11），"
    "而 MICE/KNN/均值崩塌为负 R²（−0.13 至 −0.31）。低缺失率数据上 CRISP 插补矩阵在玻璃/ge 给出最接近无缺失基线的 R²。"
    "闭环配方优化效率未被任何插补器在这些小数据集上一致改善——如实的局限。", body))

# ---------------- 4 讨论 ----------------
story.append(P("4  讨论", h1))
story.append(P(
    "n&lt;d 场景是 CRISP 的主战场。材料数据库常是高维小样本：几十个配方、几十个组分、每样本大多缺失。"
    "该场景回归插补器（MICE）退化（命题 5）、低秩补全（SoftImpute）在成分数据上直接失败，其输出物理不可行"
    "（SumDev 7–8）。CRISP 只需列中位数，显著更准（p&lt;0.01，合成与真实子集）且严格可行。"
    "下游演示确认后果：仅 CRISP 插补矩阵在 n&lt;d 数据保持正预测 R²。", body))
story.append(P(
    "CRISP 不占优之处：低缺失率稠密或主导组分数据（水泥、合成、部分 ge）上，MICE 精度相当或中等更好；"
    "CRISP 那里的差异化是严格约束保证、结构零语义、零调参与可解释性。我们不声称该场景的精度支配。", body))
story.append(P(
    "为什么 plug-in 贝叶斯观点重要：CRISP 的分配是与观测部分一致的信息量最少的补法（plug-in 意义下），"
    "避免注入伪结构；误差上界解释了完整行为何有帮助（轮廓项）以及为何无法消除误差（不可约项）。", body))
story.append(P(
    "局限：（i）全部实验为回溯，无前瞻性实验验证；（ii）贝叶斯解读是 plug-in 而非精确；（iii）"
    "一个扩展真实 n&lt;d 数据集（ge）不显著；（iv）方法简单，我们声称的是对真实问题的稳健、约束保证、"
    "可证界的方案，而非全新估计器类别；（v）LCRISP/AutoCRISP 变体已实现但未实证评估。", body))

# ---------------- 5 结论 ----------------
story.append(P("5  结论", h1))
story.append(P(
    "CRISP 是一个约束保证、结构零友好的插补方法，其主优势在材料配方数据的 n&lt;d 高维小样本场景——"
    "那里回归与低秩插补器既不准又物理不可行；其严格单纯形保证、零调参、可证明误差上界使其在其他场景成为"
    "可靠默认。实现、脚本与原始结果全部发布以保证可复现。", body))

# ---------------- 参考文献 ----------------
story.append(P("参考文献", h1))
refs_list = [
    "1. Aitchison, J. (1982). The Statistical Analysis of Compositional Data. Chapman and Hall, London.",
    "2. Templ, M., Hron, K., Filzmoser, P. (2011). robCompositions: an R-package for robust statistical analysis of compositional data. In Compositional Data Analysis: Theory and Applications, pp. 341–355. Wiley.",
    "3. Palarea-Albaladejo, J., Martín-Fernández, J. A. (2015). zCompositions — R package for multivariate imputation of left-censored values. Chemometrics and Intelligent Laboratory Systems, 143, 85–96.",
    "4. Stekhoven, D. J., Bühlmann, P. (2012). MissForest—non-parametric missing value imputation for mixed-type data. Bioinformatics, 28(1), 112–118.",
    "5. van Buuren, S., Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. Journal of Statistical Software, 45(3), 1–67.",
    "6. Mazumder, R., Hastie, T., Tibshirani, R. (2010). Spectral regularization algorithms for learning large incomplete matrices. Journal of Machine Learning Research, 11, 2287–2322.",
]
for r in refs_list:
    story.append(P(r, refs))

story.append(Spacer(1, 8))
story.append(P("注：本文中文版由英文投稿稿翻译，用于理解与展示；正式投稿请使用英文版并遵守期刊 AI 政策声明。", refs))

out = os.path.join(BASE, "CRISP_PAPER_CN.pdf")
doc = SimpleDocTemplate(out, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                        leftMargin=2.2 * cm, rightMargin=2.2 * cm)
doc.build(story)
print("已生成:", out)
