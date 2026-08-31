# 论文修改工作清单(照此改,改完发我核)
> 对象:`Generator_Agnostic_Post_Hoc_Correction_of_Foot_Skating_in_Text_to_Motion_Generation.pdf`
> 生成:2026-08-31。按论文章节顺序排。所有数字来自 `analysis/v19/*.json`(本会话已重跑核对)。
> 分析依据见 `docs/thesis_revision_notes.md`;这份是**纯执行清单**。
> 图例:🔴必改(事实/矛盾) · 🟠必加(硬性缺失) · 🟡framing/风格 · 🟢可选加分
>
> ⚠️ 先确认:外部审稿人用的表号(Table 3=分类别、Table 4=FID)与磁盘这版 PDF
> (Table 3=FID、无分类别表)不一致——**请先确认你手上要改的是哪一版**,以免表号错位。

---

## PART A — 必改 🔴(不改答辩必被抓)

### A1 §5.E 尖峰 —— "0.91×/0.94× on all three generators" 是错的
**找到**那句 →**替换为**:
> Relative to the original output, the delivery point reduces the 99th-percentile and
> maximum ankle acceleration on T2M-GPT (0.93×/0.98×) and MDM (0.90×/0.83×), while on
> MoMask the spikes are essentially unchanged to slightly higher (1.03×/1.07×). The
> spike behaviour is therefore generator-dependent, and we report it per generator; the
> FSR and RMS-jitter improvements in Tables 1–2 hold on all three.
- 数据:`analysis/v19/spike_ratios_088a10.json`

### A2 §5.C 分类别 —— "no category regresses" 和 "rotation largest residual" 都错
**替换为**:
> Most categories improve on the original output; the two exceptions are *backward*
> (0.190 → 0.196, n = 9) and *jumping* (0.033 → 0.037), where the smoother gives back
> enough FSR to slightly exceed the original — a category-level echo of the frontier
> trade-off. The largest residual FSR is in *backward* (0.196), followed by *rotation*
> (0.175). Rotation nonetheless shows the largest absolute reduction from the de-skate
> stage (0.220 → 0.125); the reach-clamp fires most often on rotating and backward
> motions, which is where the residual concentrates.
- 数据:`analysis/v19/by_category_088a10.json`

### A3 §2.2 + §3.2 接触来源写错了(方法与代码不符)
- §3.2 现写"contact window when the **HumanML3D contact channel** indicates planted",
  但代码用的是**几何高度软权重**(foot height < 5cm),263 维 contact 通道**没被 pipeline 用**。
- **§3.2 替换为**:
> Contact windows are derived geometrically from the recovered joint positions: a foot
> is considered planted when its height falls below a small threshold (5 cm above the
> estimated ground plane), yielding a soft per-frame contact weight. The pipeline does
> not rely on the generator's predicted contact channel, which keeps it applicable to
> any generator whose output maps to joint positions.
- **§2.2**:删掉/弱化"the contact channels are particularly important … encode exactly
  the information needed"(它没被用)。

### A4 §5.1 Gaussian 支配矛盾 —— 用 CI 化解(配 A? 见下表)
在 §5.1 讨论 V19 vs Gaussian 处**加**:
> A paired bootstrap (B = 10,000) shows the difference between the learned smoother and
> the tuned Gaussian is significant on both metrics only for MoMask (FSR +0.24 pp,
> p = 0.002; jitter +0.23×10⁻³, p = 0.006), where the Gaussian is marginally better; on
> T2M-GPT the FSR difference is within noise (p = 0.115) and on MDM neither metric
> differs significantly. The two smoothers are statistically indistinguishable except on
> one generator — a precise statement of the central finding: the learned component does
> not beat a well-tuned classical baseline.
- **同时删掉** §5.1 里"V19 is the only configuration beating the original on both"这类话
  (Gaussian 也打败原始、且赢更多)。
- 数据:`analysis/v19/bootstrap_ci.json`

### A5 §5.1 "beats the original on both metrics" 要弱化(jitter 只在 T2M-GPT 显著)
**替换为**:
> Relative to the original output, the delivery point significantly reduces FSR on all
> three generators (paired bootstrap, all p < 0.001). The RMS-jitter reduction over the
> original is significant on T2M-GPT (−0.89×10⁻³, p < 0.001) but within noise on MoMask
> (p = 0.43) and MDM (p = 0.09); on those two it holds jitter at the original level while
> reducing FSR, rather than improving both.

### A6 §6.1 删掉 "learned is a drop-in … per-generator tuning" 主张(已被 #12 证伪)
**替换为**:
> We further tested whether the learned smoother's fixed-parameter nature is itself an
> advantage. A Gaussian tuned on T2M-GPT (σ = 1.08) and frozen under-smooths MoMask and
> MDM (jitter 14–16 % above their original level), so a Gaussian does need per-generator
> σ retuning to hit a fixed jitter target. This confers no benefit on the learned model:
> the frozen, untuned Gaussian still attains lower FSR than the learned smoother on all
> three generators (by 0.55–0.93 pp). The learned component does not win even on the
> cross-generator transfer axis, reinforcing that the contribution is the pipeline and
> protocol.
- 数据:`analysis/v19/transfer.json`

### A7 §3.7 delivery point 选法 + 参数定义
- `--jit-share 0.88 / --anch-share 0.10` 首次出现要定义(它们是训练损失里 jitter 项/anchor
  项的权重占比)。
- **必须明说 delivery point 是在 validation split 上选的**(否则与 leak-free 自相矛盾)。

### A8 §3.6/§4.1 训练数据量 "3.4–4 k" → 确切条数
- 查 `data/prep/v19.py` 实际产出 / `data/training/v19_cache.pt` 的样本数,写死一个数。

### A9 §3 + §5.1 de-skate 后 FSR 为何不是 0 —— 加分解表 + 定义 reach-clamp
- **方法章补定义 reach-clamp**(§5.3/§6.5 用了两次却从没定义):IK 把钉住的踝目标 clamp 到
  刚性腿长可达的球面内。
- **§5.1 加一张分解表**(T2M-GPT n=200):

  | 阶段 | FSR |
  |---|---|
  | original | 6.43% |
  | de-skate (pre-IK) | 1.96% |
  | de-skate + IK | 3.98% |
  | reach-clamp fired | 11.96% of contact frames |

  一句话解释:de-skate 把踝钉住使 pre-IK FSR 降到 1.96%(残留来自度量与去滑的 contact 判据
  不同);IK 的 reach-clamp 在目标超出腿长时把踝拉回,使 FSR 回升到 3.98%。
- 数据:`analysis/v19/deskate_decomp.json`

### A10 §5.4 语义表只有 FID,正文却说报 MM-Dist/R-precision —— 补完整表
- 补每生成器×每配置的 **MM-Dist、R-precision(值 + 95% CI + p)**,别只写"不显著"无数字。
- 数据:`analysis/v19/semantic_largeref.json`(字段 R_top1/2/3、MMDist、dMMDist_CI95、
  dMMDist_significant)。

### A11 §5.4 MDM 的 FID 方向异常 —— 加一句
- 其他两个 de-skate 让 FID 变差,MDM 变好(15.94→14.90)。**加**:MDM 本地样本滑动极重、已远
  离自然分布,de-skate 把它往回拉一点;不解释读者当 bug。

### A12 FID 表(及分类别表若加)加表注:生成器、n、τ
- 明确每张表的**生成器构成、样本数 n、位移阈值 τ**;分类别若成表,注明是 3 生成器聚合 n=150、
  τ 与主表相同、prompt 子集与主表不同(所以绝对值更高)。

---

## PART B — 必加 🟠(硬性缺失)

### B1 Table 1/2/3 每格加 95% CI(paired bootstrap)
**T2M-GPT (n=200)** FSR% [CI] / Jitter×10³ [CI]:

| 配置 | FSR% [95% CI] | Jitter×10³ [95% CI] |
|---|---|---|
| original | 6.43 [5.17, 7.88] | 9.16 [8.26, 10.12] |
| de-skate+IK | 3.98 [3.32, 4.70] | 16.55 [14.63, 18.53] |
| Gaussian+IK | 4.92 [4.04, 5.88] | 7.78 [6.92, 8.68] |
| learn+IK (V18) | 6.00 [4.88, 7.21] | 7.04 [6.27, 7.85] |
| V19 (delivery) | 5.08 [4.21, 6.00] | 8.27 [7.40, 9.18] |

**MoMask (n=200)**:

| 配置 | FSR% [95% CI] | Jitter×10³ [95% CI] |
|---|---|---|
| original | 7.86 [6.50, 9.29] | 8.43 [7.54, 9.38] |
| de-skate+IK | 4.73 [3.97, 5.58] | 18.96 [16.80, 21.15] |
| Gaussian+IK | 6.33 [5.27, 7.52] | 8.09 [7.20, 8.99] |
| learn+IK (V18) | 7.18 [5.96, 8.57] | 7.13 [6.36, 7.94] |
| V19 (delivery) | 6.57 [5.52, 7.72] | 8.32 [7.45, 9.22] |

**MDM (n=50)**:

| 配置 | FSR% [95% CI] | Jitter×10³ [95% CI] |
|---|---|---|
| original | 11.86 [8.77, 15.94] | 14.18 [12.00, 16.46] |
| de-skate+IK | 7.04 [5.70, 8.37] | 28.50 [24.57, 32.52] |
| Gaussian+IK | 9.06 [7.38, 10.69] | 13.63 [11.91, 15.34] |
| learn+IK (V18) | 11.18 [8.99, 13.34] | 11.46 [9.93, 13.00] |
| V19 (delivery) | 9.24 [7.59, 10.86] | 13.63 [11.84, 15.39] |

关键配对显著性(放正文或脚注):deskate−original(FSR,三个 p<.001)、gauss−deskate(FSR,三个
p<.001)、v19−original(FSR,三个 p<.001);v19−gauss 见 A4。数据 `analysis/v19/bootstrap_ci.json`。

### B2 图(现在全文只有一张流程框图,一篇讲脚滑的论文没有任何动作可视化)
至少补:
1. **FSR–jitter frontier 曲线**(learned vs Gaussian,每生成器一条)—— 核心结论现在只有文字!
2. **踝 XZ 轨迹叠加图**:同段动作 original / de-skate / V19,标出 contact window。
3. **踝速度/加速度时序**:展示边界抖动 + smoother 修复。
   → 这三张**我可以直接生成**(数据现成),你说一声。可选④学习核 vs Gaussian 核。

### B3 §4/附录 补网络结构 + 训练超参(硬性要求)
- 从 `models/v19.py`:1D-CNN hidden=64, kernel=5, n_layers=4, in_dim=12, out_dim=4。
- 从 `training/v19.py`:optimizer、lr(注意用 2e-4 非 2e-3)、batch、epoch、train/val 划分、
  验证 loss 选点、λ 梯度范数标定。补一条训练曲线。

### B4 §3.3 损失函数完整化
- 式(3)分母:确认实现是否只对 planted 帧(Σc_t)——若是要把 contact mask 写进公式,否则
  "threshold-aligned"卖点与公式矛盾。
- 补:g(·) 是什么(sigmoid?)、温度 s 取值、**L_smooth 的定义**、λ 标定公式、
  **anchor 保真项**(`--anch-share`,防止踝轨迹塌成常数)。全在 `models/v19.py::V19Loss`。

### B5 文献(现只有 9 篇,§2.3 建立空白那节零引用)—— 扩到 25–35 篇
- **必补 Kovar et al. 2002, "Footskate Cleanup for Motion Capture Editing" (SCA)** —— 它就是
  generator-agnostic、retrain-free、IK-based footskate 后处理,你的新颖性主张不讨论它站不住。
  写差异:Kovar 是 mocap 编辑、假设已知可靠 contact、无学习组件、无文本对齐评估。
- 再补:物理约束生成(PhysDiff / GMD / ReMoDiffuse)= 你说的"第一 camp";learned pose
  refinement(支撑 §2.3 "learned smoothers effective in adjacent problems"那句);FSR 定义差异
  的参照文献。至少一个外部基线,或论证为何无可公平移植的对照。

### B6 复现性 / 开销 / 摘要数字 / 数据完整性
- 复现性声明:确切硬件(gpu3, TITAN Xp 12GB)+软件(torch 2.1.0+cu121)——别写"recorded in
  project log"。
- **运行时开销**:post-hoc 最大卖点是便宜——每条动作处理耗时?相比重训省多少?现在完全没提。
- 摘要加核心数字(FSR 6.43→5.08 等),现在全是定性。
- 解释本地 HumanML3D 只有 8177/14616:为什么不完整、训练抽样是否有系统性类别偏差。

### B7 "三个生成器"主张 vs 表里只有两个 —— 二选一
- **数据其实在**:`v19_088a10_expanded_results.json` 有 mdm 块(n=50)全五路。
- 要么把 MDM 物理表补进正文(n=50,标注),要么把主张降级为"两个生成器完整评估 + 第三个作
  OOD 案例"。别标题说三个、表里两个。

---

## PART C — framing / 风格 🟡

### C1 版本号语义化
- V19 / V18 / v19_088a10 / --jit-share / `data/prep/v19.py` → `Ours (full)` /
  `Ours w/o IK-in-loop` / `De-skate only` / `Gaussian baseline`。摘要第三句的"V19"尤其要改。

### C2 代码路径移附录/repo
- §4.5 那些 `models/v19.py` 之类路径不算方法描述,移到附录或 GitHub 链接。

### C3 贡献列表保持一致
- Intro 三条贡献已经是对的(pipeline / IK-in-loop / 协议+诚实发现),确保全文口径一致:
  价值在管线+协议+IK-in-loop,不在学习组件。#12/#15 都指向这个,别再出现"学习更省调参"式暗示。

---

## PART D — 可选加分 🟢(有时间,ROI 高)

- **D1 感知实验(审稿人排第一)**:工具就绪,找 15–20 人各 36 次比较,一天收齐。即便"无显著偏好"
  也是有价值发现。只有你能组织人;我可帮你扩渲染到 MoMask/MDM 或核对打分脚本。
- **D2 #7 独立 contact 标签重算 FSR**:用更严的几何判据重算,验证 FSR 下降不是构造性的。我能跑。
- **D3 #8 训练/测试分布不匹配的机制分析**:把负结果从"诚实"提到"有诊断"(smoother 训练在
  几乎不滑的 GT 上,没见过测试时的 artifact)。这是最能拔高第 6 章的一段。
- **D4 学习核 vs Gaussian 核可视化**:若近似→"学习≈Gaussian"从经验升级为机制解释。我能画。

---

## 我这轮已落盘、可直接引用的数据/脚本
| 数据 | 文件 | 脚本 |
|---|---|---|
| 088a10 尖峰(3 生成器) | `analysis/v19/spike_ratios_088a10.json` | `analysis/v19_spikes_3gen.py` |
| 088a10 分类别 | `analysis/v19/by_category_088a10.json` | `analysis/v19_by_category.py` |
| de-skate FSR 分解 | `analysis/v19/deskate_decomp.json` | `analysis/v19_deskate_decomp.py` |
| 跨生成器迁移(#12) | `analysis/v19/transfer.json` | `analysis/v19_transfer.py` |
| paired bootstrap CI(#15) | `analysis/v19/bootstrap_ci.json` | `analysis/v19_bootstrap_ci.py` |

---

## 建议改的顺序
1. **PART A 全部**(A1–A12,事实/矛盾,粘贴文本已给)——最优先。
2. **B1 CI + B7 MDM 口径**(数字现成)。
3. **B2 图**(我来生成)+ B3/B4 方法补全(从代码抄)。
4. **B5 文献 + B6 复现性/开销**。
5. **PART C** 顺手改。
6. **PART D** 有时间再做,D1 感知实验 ROI 最高。

改完发我,我逐条核对新稿与磁盘数据是否一致。
