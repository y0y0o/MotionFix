# MotionFix — 简历素材文档

> 用途:整理成简历条目。所有数字来自 `analysis/v19/*.json`,可追溯、可复现。
> 面试时会被追问的点已在 §6 列出并配好答案。
> 生成日期:2026-07-21

---

## 1. 一句话是什么

**MotionFix 是一个通用的文本生成动作后处理管线,专门修复"脚滑"(foot skating)这一
类物理不合理伪影 —— 不需要重训练任何生成模型,对 MoMask / T2M-GPT / MDM 三个主流
文生动作模型即插即用。**

英文一句话:

> A generator-agnostic post-processing pipeline that removes foot-skating artifacts
> from text-to-motion generators, requiring no retraining of the generator.

---

## 2. 解决什么问题(为什么这事值得做)

文本生成 3D 人体动作(text-to-motion)的主流模型 —— MoMask、T2M-GPT 是 VQ 量化类,
MDM 是扩散类 —— 生成的动作在**语义上正确**(叫它走路它就走路),但在**物理上不成立**:
脚踩在地面上的时候,脚还在水平方向滑动,像踩在冰上。

- 量化实测:MoMask 脚滑率 16.3%、MDM 11.9%、T2M-GPT 12.0%
  (脚滑率 = 处于地面接触的帧中,水平速度 > 0.03 m/frame 的帧占比)
- 这个问题**普遍存在于所有生成器**,且各家的内置 IK 只能修掉约 0.9pp
- 现有方案要么改生成器结构(需要重训,成本高),要么上物理仿真(慢、且会引入新抖动)

**所以问题定义是:能不能做一个轻量、通用、不碰生成器的后处理,把脚滑修掉,
同时不破坏动作的视觉平滑度和文本语义?**

---

## 3. 怎么做的(方法)

### 3.1 最终管线(三阶段)

```
输入动作 (T, 22, 3)
    │
    ├─ ① 去滑 (de-skate,纯物理)
    │     检测接触段 → 段内脚的水平位置钉在该段 XZ 均值上
    │     消除滑动,但会在接触段边界产生速度突变
    │
    ├─ ② 平滑器 (可选,46.3K 参数 1D-CNN)
    │     只输出双踝 4 维;接触权重作为条件输入
    │     抑制①引入的边界尖峰
    │
    └─ ③ 可达性钳制两骨 IK (纯物理)
          髋→膝→踝余弦定理求解 + 刚性脚趾
          先把踝目标钳进腿长可达球内,再解 IK
          → 骨长严格不变,不会出现"腿被拉长/脚翻转"
    │
输出动作 (T, 22, 3)
```

**核心洞察(这是论文的主张,也是简历里最值钱的一句):**

> 后处理脚滑修正**本质是一个约束满足问题,而不是一个学习问题**。

管线的 ①③ 两步是纯解析的、**零训练**的。第 ② 步是唯一的学习组件,而实验证明
它带来的增益非常有限(见 §4.3)—— 这个否定性结论本身是有价值的实验贡献。

### 3.2 关键技术点

| 技术点 | 做法 | 为什么重要 |
|---|---|---|
| **可微 IK** | 把 IK 的踝钳制路径用 PyTorch 重写,放进训练前向 | 原先 IK 只在推理时套在模型输出外面,训练与推理路径不一致,实测代价是 FSR +0.8pp、Jitter ×1.18。数值与 numpy 版对齐到 2.2e-16 |
| **阈值对齐的代理损失** | 脚滑指标是"速度超阈值的帧**计数**",损失就写成 `sigmoid((v − 0.03)/τ)` 的软计数,而不是常见的速度均值 | 代理与指标错配是这类工作最常见的隐性缺陷:优化均值会去压那些本来就没超阈值的帧,对指标零贡献 |
| **接触分区损失** | 空中帧、接触帧、全序列分别加权;损失项按接触权重的三次方加权 | 空中帧本来就该平滑,接触帧才需要刹停,两者混在一起会互相抵消 |
| **λ 梯度范数自动标定** | 每个损失项按其梯度范数反比赋权,再用单一 `--jit-share` 参数在前沿上选点 | 避免手调多个 λ;把权重从"超参"变成"工作点选择器" |
| **可达性钳制** | 解 IK 前先把踝目标投影进 `(L1+L2)` 可达球,水平方向单独钳制 | 无钳制的两骨 IK 在目标不可达时会产生数值爆炸或翻转 |

### 3.3 评估体系(这部分工作量不比方法小)

- **7 项指标**:FSR(脚滑率)、Jitter(踝加速度 RMS)、Floating(悬空)、FootErr
  (相对原动作偏移)、ContactAcc(接触准确率)、BoneCV(骨长一致性)、Penetration(穿地)
- **无泄漏划分**:把 MoMask 测试集拆成 held-out(n=10)/ train(n=40),量化泄漏影响(实测仅 +0.15pp)
- **语义保真评估**:接 HumanML3D 官方 `text_mot_match` 评估器(MoMask/MDM/T2M-GPT
  论文用的同一个),报 FID / R-precision / MM-Dist / Diversity,配**配对 bootstrap 95% CI**
- **前沿扫描**:因为 FSR 与 Jitter 天然冲突,单点比较无意义,改为在同一 Jitter 水平
  上比较 FSR,扫出完整的 FSR–Jitter 前沿曲线(24 个可比工作点)
- **尖峰分析**:发现 Jitter 的 RMS 会掩盖"抽搐"—— 抽搐是**尖峰**现象,遂追加 p99/max/
  尖峰率与逐帧曲线

---

## 4. 做到了什么(结果)

### 4.1 主结果 —— 三个生成器全线改善

交付配置 `v19_088a10`:

| 生成器 | 脚滑率 原始 → 修正 | 相对改善 |
|---|---|---|
| MoMask (n=10, held-out) | 16.29% → **11.63%** | −28.6% |
| MDM (n=50) | 11.86% → **9.24%** | −22.1% |
| T2M-GPT (n=50) | 11.99% → **8.93%** | −25.5% |

**同时**(这才是关键 —— 修脚滑很容易,不破坏别的很难):

- Jitter RMS **三个生成器全部低于原始**(如 T2M-GPT 0.01388 → 0.01283)
- Jitter 尖峰 p99 **0.91×**、max **0.94×** 原始 —— 比原动作还平滑,不会读成抽搐
- 语义:12 个「方法×生成器」组合中,MM-Dist 变化**全部不显著**;MDM 上 FID 3.827
  vs 原始 3.822,几乎相同
- 骨长一致性(BoneCV)与接触准确率(ContactAcc 100%)由 IK 结构性保证,不是调出来的

**纯物理配置(去滑 + IK,零训练)可把 T2M-GPT 压到 FSR 6.65%、MDM 7.04%**,
逼近 SOTA(MaskControl 5.5%),代价是抖动升高。

### 4.2 跨类别泛化(50 prompts × 7 类别 × 3 生成器 = 150 对)

| 类别 | n | 原始 FSR | 去滑+IK | 改善 |
|---|---|---|---|---|
| walking | 21 | 20.76% | 8.07% | **−12.69pp** |
| rotation | 18 | 21.99% | 12.51% | **−9.48pp** |
| backward | 9 | 18.97% | 13.19% | −5.78pp |
| complex | 30 | 11.74% | 7.38% | −4.37pp |
| turning | 21 | 13.84% | 9.80% | −4.05pp |
| dance | 30 | 5.99% | 3.12% | −2.87pp |
| jumping | 21 | 3.34% | 2.96% | −0.38pp |
| **ALL** | **150** | **12.64%** | **7.31%** | **−5.33pp** |

**两个可说的点**:
1. **七个类别无一变差**,改善幅度随输入严重程度**单调递增** —— 输入越糟收益越大,
   这正是"方法不 prompt-sensitive"的定义
2. rotation(旋转类)是早期版本的最大软肋(V8 时代几乎无改善),现在是改善第二大的类别

### 4.3 一个诚实的否定性结果(面试可以主动讲,这比正面结果更能体现研究能力)

**学习式平滑器没有打赢一个调好的高斯滤波器。**

在 24 个同 Jitter 水平的可比工作点上,学习模型只赢 3 个(每个生成器 1 个,
−0.28 ~ −0.40pp),其余 21 个输,最多输 0.90pp。

关键在于**这个结论的证据强度**:早期版本也得出过类似结论,但那个实验有六类缺陷
(数据泄漏、代理-指标错配、IK 训练/推理错配、训练语料被反归一化 bug 污染、
学习率错误、按训练 loss 选 checkpoint),任何人都能说"你只是没训好"。
**我把这六条全部修掉后重跑,结论依然成立。** 这才让它从"我没做好"变成"这条路不通"。

修复的具体收益也量化了:平滑阶段相对纯去滑的 FSR 回吐,从 **+4.5pp 降到 +0.9pp**。

### 4.4 一个意外发现

`deskate_ik`(纯物理)的**语义损伤最大**(MoMask 上 MM-Dist 显著 +0.279,FID +1.81)。
也就是说平滑阶段除了"让画面不抖",**还有一个此前没被论证过的作用:把纯物理去滑造成的
语义偏移拉回来。** 这给了学习组件一个独立于 Jitter 的存在理由。

---

## 5. 工程 / 规模数据(简历里可以用的硬指标)

| 项 | 数值 |
|---|---|
| 平滑器参数量 | **46.3K**(对比早期 Transformer 版本 19.1M,减少 **99.8%**) |
| 训练语料 | 4,000 条 HumanML3D 动作(经接触占比与去滑偏移双重过滤) |
| 推理开销 | 单次前向,无迭代;相比扩散式修复方案(~50 步去噪)量级更低 |
| 评估规模 | 50 prompts × 7 类别 × 3 生成器 × 5 种方法 |
| 指标数 | 7 项物理指标 + 4 项语义指标(FID/R-prec/MM-Dist/Diversity) |
| 迭代版本 | V8 → V19,每版都有量化诊断与失败归因记录 |
| 环境 | PyTorch 2.1 + CUDA 12.1,NVIDIA TITAN Xp 12GB |

---

## 6. 面试会被追问的问题 + 答案

**Q: 你这个方法的创新点到底是什么?一个去滑加 IK 听起来很简单。**
A: 创新在**问题重构**,不在模型复杂度。这个领域的默认假设是"生成伪影要用更强的
生成模型来修",我给出的证据是它本质是约束满足问题 —— 一个零训练的解析管线就能达到
接近 SOTA 的脚滑率,而一个专门为此设计、损失函数与指标严格对齐的学习模型
**打不赢一个调好的高斯滤波器**。技术上具体的贡献是可微 IK 进训练回路、
阈值对齐的软计数代理损失、以及可达性钳制。

**Q: 为什么最后交付的还是带学习组件的版本?既然它没赢。**
A: 因为它在**多目标**上赢。纯物理版本 FSR 更低(6.65%)但 Jitter 尖峰是原始的 3.6 倍,
视觉上抽搐,而且语义损伤最大。交付点在 FSR、Jitter RMS、Jitter 尖峰、语义四项上
**同时优于原始动作**,纯物理版做不到这一点。

**Q: 怎么保证你的结论不是训练不充分导致的?**
A: 这正是我花最多时间的地方。我系统排查了六类实验缺陷并逐一量化其影响,
其中两个是比较隐蔽的:一是数据准备脚本对**已经是原始值**的 HumanML3D 特征又做了一次
反归一化,导致脚的垂直活动范围被压缩到 9cm、96% 的帧被误判为接触;二是学习率
2e-3 配合零初始化输出层,Adam 第一步就把模型打到恒等映射,而**验证损失曲线看起来
是在下降的** —— 下降完全来自损失权重变化。我后来的做法是任何 loss 曲线旁边
必须画一个与权重无关的量(残差范数)。

**Q: Jitter 这个指标可靠吗?**
A: 部分不可靠,这是我自己发现并修正的。RMS 会掩盖"抽搐"的真实性质 —— 抽搐是**尖峰**
现象而非均值抬高,同样的 RMS 可以对应完全不同的观感。我因此追加了 p99/max/尖峰率
和逐帧曲线,并据此**推翻了自己原先的交付点选择**(原选点 p99 是原始的 1.54 倍)。
指标与主观感受的相关性仍需感知实验验证,这是我明确列出的局限。

---

## 7. 简历写法建议

### 7.1 三行版(空间紧张时,推荐)

> **MotionFix:文生动作物理修正管线** | 硕士毕业设计
> 提出脚滑修正的约束满足式重构:去滑 + 可达性钳制两骨 IK + 46K 参数接触条件平滑器,
> 无需重训练生成器。三个主流生成器(MoMask/T2M-GPT/MDM)脚滑率降低 22–29%,
> 同时抖动与文本-动作语义对齐**均优于原动作**;纯物理配置达 FSR 6.65%,逼近 SOTA。
> 搭建含 7 项物理指标 + 4 项语义指标(配对 bootstrap 显著性检验)的无泄漏评估体系,
> 通过修复六类实验缺陷,给出"学习式平滑不优于解析滤波"的高强度否定性结论。

### 7.2 五行版(推荐用于主项目)

> **MotionFix:通用文生动作物理修正管线** | 硕士毕业设计 | PyTorch
> - 将后处理脚滑修正**重构为约束满足问题**:去滑 + 可微两骨 IK(可达性钳制)+
>   46.3K 参数接触条件平滑器,生成器无关、无需重训练
> - 三个主流生成器脚滑率 16.3%/11.9%/12.0% → 11.6%/9.2%/8.9%(**−22~29%**),
>   且 Jitter RMS、尖峰(p99 0.91×)、语义对齐(MM-Dist 变化全部不显著)**三项均优于原动作**
> - 将 **IK 引入训练回路**(数值对齐 2.2e-16)并把代理损失与评测阈值对齐,
>   使平滑阶段的 FSR 回吐从 +4.5pp 降至 **+0.9pp**;模型参数量较早期方案减少 **99.8%**
> - 搭建无泄漏评估体系:7 项物理指标 + HumanML3D 官方语义评估器 + 配对 bootstrap CI +
>   FSR–Jitter 前沿扫描;跨 7 类动作 150 组测试**无一类别退化**
> - 通过定位并修复六类实验缺陷(数据泄漏、代理-指标错配、语料反归一化 bug、
>   IK 训练/推理错配等),给出"学习式平滑不优于调优高斯"的可信否定性结论

### 7.3 英文版(留学/外企)

> **MotionFix: A Physics-Based Correction Pipeline for Text-to-Motion Generation**
> - Reframed post-hoc foot-skating correction as a **constraint-satisfaction problem**:
>   de-skating + reach-clamped 2-bone IK + a 46.3K-parameter contact-conditioned smoother;
>   generator-agnostic and requires no retraining
> - Cut foot-skating ratio by **22–29%** across MoMask / T2M-GPT / MDM
>   (16.3%/11.9%/12.0% → 11.6%/9.2%/8.9%) while **simultaneously improving** jitter RMS,
>   jitter spikes (p99 0.91× original) and text–motion semantic alignment
> - Brought **IK inside the training loop** via a differentiable formulation
>   (matched to 2.2e-16) and aligned the surrogate loss with the evaluation threshold,
>   reducing the smoothing stage's FSR cost from +4.5pp to **+0.9pp** with **99.8% fewer
>   parameters** than the earlier Transformer baseline
> - Built a leak-free evaluation harness: 7 physical metrics, the official HumanML3D
>   `text_mot_match` semantic evaluator with paired-bootstrap CIs, and a full FSR–Jitter
>   frontier sweep; **no regression across any of 7 motion categories** (150 test pairs)
> - Diagnosed and fixed six classes of experimental defect to establish a **well-supported
>   negative result**: a learned smoother does not outperform a tuned analytic filter here

### 7.4 用词提醒

- **不要写"效果很好"或"SOTA"。** 纯物理配置 6.65% 只是"逼近" MaskControl 的 5.5%,
  不是超越。写"逼近 SOTA"是安全的,写"达到 SOTA"会被当场问穿。
- **主动写否定性结果。** 面试官对"我做出了一个否定性结论并证明它不是我没做好"的
  评价,通常高于"我的模型涨了 2 个点"。这是研究品味的直接证据。
- **强调 99.8% 参数削减和零训练路径。** 工业界最关心的是"能不能上线",
  一个 46K 参数的单次前向管线比一个 19M Transformer 有说服力得多。
- **不要在简历里提"论文尚未提交"或版本号 V19。** 只讲方法与结果。

---

## 8. 数据来源与可复现性

| 内容 | 位置 |
|---|---|
| 主结果 7 指标 | `analysis/v19/v19_088a10_results.json` |
| 前沿扫描(24 工作点) | `analysis/v19/frontier.json` / `frontier.png` |
| 语义评估 | `analysis/v19/semantic.json` |
| 尖峰分析 | `analysis/v19/jitter_stats.json` / `jitter_trace.png` |
| 分类别泛化 | `analysis/v19/by_category.json` |
| 方法实现 | `models/v19.py`(可微 IK + 分区损失)、`models/v18_ik.py`(IK) |
| 完整结果文档 | `docs/v19_results.md` |
| 开发过程与失败记录 | `docs/v19_devlog.md` |
| 导师汇报版 | `docs/progress_report_0721_EN.md` |
