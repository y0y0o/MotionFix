# 论文修改清单 R2(第二轮 review · 照此改)
> 2026-08-31 第二轮。**上一版进步很大**;本轮按审稿人严重程度排。
> 所有数字来自本会话磁盘实测。图例:🔴必改 · 🟠必加 · 🟡待跑/待你确认 · ✅我已核。
> ⚠️ 一条**结果待填**:IK-in-the-loop frontier 正在后台训练,完成我补上(见 A1)。

---

## A1 🔴✅【最严重·已定论·贡献#2 成立且更强】IK-in-the-loop 的 frontier 证据(审稿人「一」)
**实验已跑完**(`analysis/v19_ikloop_frontier.py` → `analysis/v19/ikloop_frontier.json` +
`ikloop_frontier.png`;诊断见下)。同模型同损失,`--no-ik-in-loop` 只切 loss 算在 pre-IK/post-IK,
with/without 各 5 个 jit-share × 2 生成器。**结果不是"frontier 平移",而是 without-IK 直接崩溃:**

| 模型(jit=0.88) | pre-IK 残差 | pre-IK FSR | **post-IK FSR** |
|---|---|---|---|
| with-IK | 0.003 m | 5.48% | **5.64%** |
| without-IK | **0.335 m** | **0.00%** | **15.53%** |

- **机制(= 论文 §3.4 想堵的 proxy-gaming)**:without-IK 模型没见过 IK,就把踝推到 0.34 m 远、
  让 pre-IK FSR 完美归零(val loss 0.00026);但该目标远超腿长可达域,推理时 reach-clamp 把踝拉回
  → post-IK FSR 爆到 15.5%(**比原始 6.43% 还差一倍**)。
- **跨设置全复现**:5 个 jit-share(0.45–0.94)× 2 anchor(0.05–0.20),without-IK 全部塌到
  t2mgpt ~14.5% / momask ~16%,**没有可用 frontier**;with-IK 是正常 frontier
  (t2mgpt FSR 4.16→5.52%,jit 0.0129→0.0074)。
- **贡献#2 的结论(比原文更强)**:IK-in-the-loop **不是**沿 frontier 挪工作点,而是**承重的**——
  它使阈值对齐的软计数损失不被 trivially 钻空子;去掉它,学习平滑器**不可用**(比不修更差)。
- **诚实边界**:这证明"IK-in-loop 对 V19 损失是必需的",**不改变**"learned ≈ Gaussian"结论(见 A2)。
- **可粘(§3.4 / §5 消融 / 贡献#2 论证)**:
  > To test whether IK-in-the-loop provides a benefit beyond moving along the FSR–jitter
  > frontier, we trained the identical smoother and loss with the loss computed on the
  > pre-IK output instead (inference still applies IK). Without IK-in-the-loop the model
  > games the proxy: it drives the pre-IK ankle 0.34 m from the de-skated reference to
  > reach 0.00% pre-IK FSR, but this target lies far outside the leg's reachable range, so
  > the inference-time IK clamp pulls the ankle back and the post-IK FSR rises to 15.5%
  > (T2M-GPT) / 16% (MoMask) — worse than the uncorrected 6.4% / 7.9%. This collapse is
  > invariant to the jit-share (0.45–0.94) and anchor (0.05–0.20) settings: without
  > IK-in-the-loop there is no usable frontier. IK-in-the-loop is therefore load-bearing
  > rather than a convenience — it is what makes the threshold-aligned loss safe — while
  > the "learned ≈ tuned Gaussian" finding (Section 5.1) concerns the with-IK model.
- **图**:`analysis/v19/ikloop_frontier.png` 已生成(without-IK 点全悬在 with-IK 曲线上方远处)。
  放正文时把 caption 说清楚这是"proxy-gaming 崩溃",不是普通 frontier。

---

## A2 🔴✅【第二严重】补全 Gaussian vs Ours 的 6 个配对检验,改结论句(审稿人「二」)
- 审稿人对:**T2M-GPT 的 jitter 显著性被跳过了**。完整 6 格(差=Ours−Gauss,正=Gauss 更优,
  `analysis/v19/bootstrap_ci.json`):

  | 生成器 | FSR 差 pp (p) | Jitter 差 ×10³ (p) |
  |---|---|---|
  | T2M-GPT | +0.16 (p=.115 **ns**) | **+0.49 (p<.001 SIG)** |
  | MoMask | +0.24 (p=.002 SIG) | +0.23 (p=.006 SIG) |
  | MDM | +0.18 (p=.275 ns) | −0.00 (p=.999 ns) |

- **§5.1 结论句必须改**(现写"indistinguishable except on one generator"是错的)。可粘:
  > A paired bootstrap (B = 10,000) over motions gives the following per-generator
  > comparisons of the learned smoother against the tuned Gaussian (positive = Gaussian
  > lower/better): T2M-GPT FSR +0.16 pp (p = 0.115, ns) but jitter +0.49×10⁻³
  > (p < 0.001); MoMask FSR +0.24 pp (p = 0.002) and jitter +0.23×10⁻³ (p = 0.006);
  > MDM FSR +0.18 pp (p = 0.275, ns) and jitter −0.00×10⁻³ (p = 0.999, ns). The tuned
  > Gaussian is thus significantly better on at least one metric on two of the three
  > generators (T2M-GPT jitter; MoMask both); the two smoothers are statistically
  > indistinguishable only on MDM. This is a stronger and more honest statement of the
  > central finding: the learned component does not beat, and on some generators is
  > slightly beaten by, a well-tuned classical baseline.

---

## A3 🔴✅【方法学无效比较】§6.1 迁移比较换成匹配-jitter(审稿人「三」)
- 审稿人对:欠平滑必然低 FSR,拿高 14–16% jitter 的配置去比 FSR = 无效,而且方向自己吃亏。
- **我重做了匹配-jitter 版**(把冻结 Gaussian 的 σ 调到与 learned 同 jitter 再比 FSR,
  数据 `analysis/v19/transfer.json` 重分析):

  | 生成器 | learned FSR @jit | Gauss FSR @同 jit | σ_needed |
  |---|---|---|---|
  | T2M-GPT | 5.08% @0.0083 | 4.75% | 1.33 |
  | MoMask | 6.57% @0.0083 | 6.25% | 1.42 |
  | MDM | 9.24% @0.0136 | 9.06% | 1.50 |

- **诚实结论:没翻盘**——匹配 jitter 后 learned FSR 仍高 0.18–0.33pp。**但旧比较无效,必须换。**
- **§6.1 可粘**(替换现有那段):
  > We compare the learned smoother against the Gaussian at matched jitter to avoid the
  > confound that an under-smoothed Gaussian trivially attains lower FSR. Re-tuning the
  > Gaussian's σ per generator so that its jitter equals the learned smoother's, the
  > Gaussian attains a slightly lower FSR on all three generators (learned +0.33 / +0.32 /
  > +0.18 pp on T2M-GPT / MoMask / MDM). The σ required differs across generators
  > (1.33 / 1.42 / 1.50), so a single fixed Gaussian cannot match the learned model's
  > operating point everywhere; the learned model reaches a consistent jitter with one
  > fixed set of weights. The learned component therefore offers tuning-free consistency
  > across generators but not lower error, again consistent with the central finding.

---

## A4 🔴🟡【可复现性硬伤】损失权重 λ/share 说明 + 式(3)(审稿人「四」)
**(1) share → λ 的映射必须写全**。现状:`--jit-share 0.88` 但 λ_air+λ_all=0.965;
`--anch-share 0.10` 但 λ_anch=0.033——读者无法重建。**精确机制(可粘)**:
> The `*-share` values are targets for each term's share of the *gradient* norm, not the
> loss weights. On a single batch at initialisation we compute each term's gradient norm
> nₖ and set λₖ = targetₖ / nₖ, then normalise {λₖ} to sum to 1. Because λₖ is inversely
> scaled by the term's gradient magnitude, it differs from the share: with
> `--jit-share 0.88 --anch-share 0.10` the resulting calibrated weights are
> λ_jit_air = 0.838, λ_jit_all = 0.127, λ_skate = 0.0014, λ_anch = 0.033.

- 份额目标本身:jit_air = jit·2/3,jit_all = jit/3,anch = anch-share,
  **skate = max(1e-3, 1 − jit-share − anch-share)**(残差项,不是独立设的)——这句要写进去。

**(2) 式(3)修正(✅ 我已核代码)**:代码是 `_wmean(over, mask·w³)` =
分子 Σ(over·mask·w³) / 分母 **Σ(mask·w³)**——**是带 w³ 的加权平均,分子分母同权**。
论文式(3)写成 /T 且分母 Σw 都不对。→ 把式(3)改成带 contact 权重 w³、mask 归一化的加权平均。
**这是写作错误,不是代码 bug。**

**(3) w³ 要解释**:contact 权重取三次方是为压掉接触边界处软权重的拖尾、只让确实站定的帧计入
skate 惩罚。→ 给一句理由,或做一次 w¹ vs w³ 消融(可选)。

---

## A5 🔴✅ 七个物理指标全表 + 修正"不变"表述(审稿人「六.1」)
- 审稿人担心 reach-clamp 改 Y 破坏 Floating/Penetration——**实测他判断错了**
  (`v19_088a10_expanded_results.json` / `momask_pool_results.json`,跨 5 配置极差):
  - **Floating = 0(spread 0)、Penetration spread 1e-10、BoneCV spread 1e-8** → 三项到浮点精度不变。
  - **ContactAcc 微动 ≤2.4e-4**(deskate_ik 略降)。
- **改法**:把七指标全部列成表(5 配置 × 3 生成器,数值现成),把"invariant by construction"
  改成"**三项(Floating/Penetration/BoneCV)实测不变到浮点精度;ContactAcc 变化 ≤2.4×10⁻⁴**"。
- **回答 L1/L2 来源**:`apply_ik` 里 L1/L2 **逐帧取自原动作并精确保留**,故 BoneCV 不变。写进 §4.2。

---

## A6 🔴✅ 定义评测器 contact 判据 + 去滑窗口/ground plane(审稿人「六.2/六.3」)
**评测器 c_t(§4.2 必须写)**:`compute_contact_labels` —— 帧 t 判为 planted 当
**踝高 < ground + 0.05 m 且 水平速度 < 0.5 m/frame**;ground = 该踝 Y 的**第 5 百分位**。
FSR 再在 planted 帧里数**水平速度 > 0.03 m/frame(SKATE_THRESH)**的比例。

**去滑窗口/权重(§3.2)**:软 contact 权重 w = sigmoid((ground + 0.05 − foot_y)/0.02);
硬窗口 W = 对 w 以 **0.5** 阈值化得到的接触段;ground = foot Y 的第 5 百分位(逐序列、逐踝)。
- **🟡 待你确认的不一致**:`deskated_target` 的 docstring 说接触段内"hold at **segment onset**",
  但论文 eq(2) 写的是"**window mean**"。**这两者不同**,请对照实现确认 eq(2) 到底是 onset 还是 mean,
  据实改。(我只看到函数前半段,没法替你断。)

---

## A7 🔴✅ ReMoDiffuse 引错(审稿人「七.1」)
- §2.3 把 ReMoDiffuse 归入"physics-constrained / contact loss"是错的——它是**检索增强扩散**,
  与物理约束无关。**换成** Rempe et al., *Contact and Human Dynamics from Monocular Video*
  (ECCV 2020),或 Karunratanakul et al., *Guided Motion Diffusion* (GMD, ICCV 2023)。
  GMD 顺带可作 §6.2"FSR 定义跨论文不可比"的具体参照(它用 height-weighted FSR)。PhysDiff 保留。

## A8 🔴✅ 摘要与 6.3 关于 MDM 自相矛盾(审稿人「七.2」)
- 摘要"significantly reduces FSR on **all three** generators"把 MDM 算进主结论,但 5.6/6.3 说
  MDM 不用于推结论。**二选一**:摘要改"on two generators, with a third reported as an
  out-of-distribution case",或放弃 MDM 排除声明。建议前者。

## A9 🔴✅ "reducing jitter" 系统性夸大(审稿人「七.3」)
- 5.2/6.1 写"also reducing jitter"——但 jitter 显著下降**只在 T2M-GPT**(MoMask p=.43,MDM p=.09)。
  **改成**"significantly reduces jitter on T2M-GPT and holds it at the original level on the
  other two"。spike 部分(2/3 生成器降)可保留 "most"。

## A10 🔴✅ 交叉引用坏了(审稿人「七.6」)
- 多处"Section C"实为 6.3(MDM 复现性):4.4、Table 3 表注、5.6、Table 7 表注。附录 A/B/C/D
  编号与正文子节撞了,**全文过一遍交叉引用**。

---

## B 必加 🟠(图 + 表)

### B1 三张图(审稿人「五」)—— ✅ 全部完成(脚本 `analysis/v19_frontier_fig.py` + `v19_figs.py`)
1. ✅ **主 Frontier 图** → `analysis/v19/fig_frontier.png`(三联:T2M-GPT/MoMask/MDM;三曲线
   With-IK / Gaussian / Without-IK;**断轴**上段放 without-IK 崩溃、下段放真实 frontier;
   uncorrected 虚线 + 交付点星标)。With-IK≈Gaussian(差 0.2–0.4pp)、without-IK 14–25%。**放 §5.B。**
2. ✅ **踝 XZ 轨迹图** → `analysis/v19/fig_trajectory.png`(全长 + zoom inset;串字体;clamp 底部 rug;
   Ours(full);zoom 目视证实无过冲)。
3. ✅ **踝速度图** → `analysis/v19/fig_speed.png`(拆双 panel:上尖峰/下 0-0.1 决策带;τ=0.03 标注;
   图例移出;contact 阴影)。
- 三图都用 serif 字体、语义 label、无图内问句标题;caption 用我给的带结论句版本。
- 旧文件 `fig2_*/fig3_*/ikloop_frontier.png` 已被上述取代,可删。
- FootErr 列(证 v19 无过冲):T2M-GPT 0.0315 / MoMask 0.0361 / MDM 0.0250(≈deskate<gauss<learn),补进 Table。

### B2 §5.4 补 R-precision / Diversity 显著性 + 解释异常值(审稿人「七.4/七.5」)
- 摘要/结论说"no significant change in R-precision"但只给了 MM-Dist 的 CI。**补 R-precision 的
  配对 bootstrap**,或把断言改为"R-precision 变化在 ±0.01 内"(裸值支持:各配置 R@1 差 ≤0.01)。
- **Table 7 两个没提的数字**:MoMask 的 Ours Diversity = **8.98**(最大语义退化,原始 9.38、参考≈9.5);
  MDM 的 de-skate 让 R@1 从 0.430 跳到 **0.490**。各给一句(大概率 n=200/50 采样噪声),或对
  Diversity 也做 bootstrap。Diversity 整列正文从未讨论,至少提一句。
- 🟡 R-precision/Diversity 的 per-sample bootstrap 我能跑(需确认 evaluator 是否留了 per-sample 值),你要就说。

---

## C 结构 / 仍未处理(审稿人「六.4/六.5」「八.14」)

- **C1** Frontier 的 MoMask 仍 n=10:补到 50,或 §5.2 明确"MoMask frontier 仅作定性参考"。
- **C2** 感知实验仍未收:强烈建议收(15 人半天)。**新设计**:加一个 **Gaussian vs learned** 对比臂
  ——无偏好=对"打平"的独立佐证,有偏好=指标测不到的东西,两种都比现在有价值。
- **C3** 把 **6.6(plateau 机制)挪到 6.1 之后**——它是全文最有洞察的一段,现在排在 future work 后,位置不对。
- **C4** Table 5 分类别(审稿人「七.7」):① 3 倍量纲差"不同子集"撑不住,要说明子集为何更难,或
  **在主表 n=200 上重跑分类别**;② 聚合含 MDM,与"MDM 不用于推结论"矛盾——重跑时**只用 T2M-GPT/MoMask**。
  🟡 这个重跑我能做(在 n=200 上、剔除 MDM),你要就说。

---

## 我能立刻替你做的(等你点)
1. **图 2、3**(踝轨迹 + 速度时序)—— 数据现成,直接出。
2. **Table 5 分类别在 n=200 主表上、剔除 MDM 重跑**(A/C4)。
3. **R-precision / Diversity 的 bootstrap**(B2)。
4. **A1 的 frontier 图** —— 后台训练完自动出,我读了给你。

## 优先级(审稿人「八」)
提交前必做:A1(定贡献#2)→ B1 图 → A2 六格检验 → A4 λ/式(3)→ A5 七指标 → A6 判据定义
→ A7 换引用 → A8/A9 口径 → A10 交叉引用。
可能翻正方向:A3 已做(未翻)、C2 感知实验。结构:C3 挪 6.6。
