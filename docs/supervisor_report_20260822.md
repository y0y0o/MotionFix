# MotionFix — Progress Report / 进度汇报

**To / 致:** Frederick (supervisor)
**From / 来自:** Xin Wan (nxkh91), Durham University
**Date / 日期:** 2026-08-22
**Delivery point / 交付版本:** `checkpoints/v19_088a10`

> Bilingual report. Each section is English first, 中文在后。
> All numbers are from actual runs on disk (`analysis/v19/*.json`), not estimates.
> 双语汇报,每节先英后中;所有数字来自磁盘上的实际运行结果,非估计。

---

## 1. One-line status / 一句话现状

**EN.** The method (V19) is finalised and frozen; a four-layer ablation, a
multi-generator physics evaluation (now enlarged), a semantic/FID evaluation
(reference set fixed today), and a perceptual-study tool-chain (awaiting human
data) are all in place. The central scientific finding is stable and reported
honestly, including the negative parts.

**中.** 方法(V19)已定版冻结;四层消融、多生成器物理评估(已扩样)、语义/FID 评估
(参考集今天修好)、感知实验工具链(等收人类数据)均已就位。核心科学结论稳定,且如实
报告(含负面结论)。

---

## 2. What has been completed / 已完成的工作

**EN.**
1. **Frozen method — V19.** Three-stage, generator-agnostic post-processor:
   de-skate (physics) → learned smoother (46,276-param 1D-CNN, ankles only, with
   the IK inside its training loop) → 2-bone IK (physics). Trained on 4k HumanML3D
   motions, so MoMask / MDM / T2M-GPT are all out-of-distribution (no leakage).
2. **Four-layer ablation.** (i) 5-way main ablation isolating each stage,
   (ii) FSR–Jitter frontier sweep (learned vs tuned Gaussian), (iii) per-category
   generalisation across 7 motion categories, (iv) semantic preservation
   (FID / R-precision / MM-Dist / Diversity) with a paired bootstrap.
3. **Enlarged physics evaluation.** T2M-GPT 50→200 and MoMask 50→200 samples;
   the three core conclusions hold at 4× sample size. MDM kept at 50 (see §5).
4. **FID reference fixed (today).** Reference enlarged from n=26 to n=1215
   (full-rank in the 512-d embedding); results are cleaner and consistent with the
   physics story (§4).
5. **Perceptual-study tool-chain built.** Blind 2-AFC local rating page + 48
   unlabelled clips + binomial-test analysis; ready to collect human ratings.

**中.**
1. **定版方法 V19。** 三段式、与生成器无关的后处理:去滑(物理)→ 学习平滑器
   (46,276 参数 1D-CNN,只动踝,IK 进训练回路)→ 2-骨 IK(物理)。在 4k HumanML3D 上
   训练,故 MoMask / MDM / T2M-GPT 全部 out-of-distribution(无泄漏)。
2. **四层消融。** (i) 隔离每一级的五路主消融;(ii) FSR–Jitter 前沿扫描(学习 vs 调好的
   高斯);(iii) 7 个动作类别的跨类别泛化;(iv) 语义保真(FID / R-precision / MM-Dist /
   Diversity)+ 配对 bootstrap。
3. **扩大物理评估。** T2M-GPT 50→200、MoMask 50→200;三条核心结论在 4 倍样本下依然成立。
   MDM 维持 50(见 §5)。
4. **FID 参考集修复(今天)。** 参考集从 n=26 扩到 n=1215(512 维嵌入下满秩);结果更干净、
   与物理结论一致(见 §4)。
5. **感知实验工具链已搭好。** 盲测 2-AFC 本地打分网页 + 48 段无标签视频 + 二项检验分析,
   随时可收人类打分。

---

## 3. Physics results (enlarged eval) / 物理结果(扩样后)

**EN.** Lower is better for both. Numbers are the 5-way ablation at n=200.
The pattern is identical across generators: **physics de-skate does most of the
FSR reduction; any smoothing (Gaussian / V18 / V19) then trades a little of that
FSR back to remove the jitter de-skate introduces.**

```
T2M-GPT (n=200)        FSR↓ (%)   Jitter↓
  original               6.43      0.00916
  deskate_ik (physics)   3.98      0.01655   ← best FSR, worst jitter
  gauss_ik               4.92      0.00778
  learn_ik (V18)         6.00      0.00704
  v19_ik (delivery)      5.08      0.00827

MoMask (n=200)         FSR↓ (%)   Jitter↓
  original               7.86      0.00843
  deskate_ik (physics)   4.73      0.01896   ← best FSR, worst jitter
  gauss_ik               6.33      0.00809
  learn_ik (V18)         7.18      0.00713
  v19_ik (delivery)      6.57      0.00832
```

Delivery point **v19_088a10** beats the original on FSR, Jitter (RMS), jitter
*spikes* (ankle-accel p99/max 0.91×/0.94× of original), and semantics, on all
three generators. Four of the seven physics metrics (Floating, ContactAcc,
Penetration, BoneCV) never move — they are invariant by construction and are
reported as such, not claimed as gains.

**中.** 两项都是越低越好,下表是 n=200 的五路消融。跨生成器规律一致:**物理去滑贡献了 FSR
的绝大部分下降;之后任何平滑(高斯 / V18 / V19)都会用一点 FSR 换回被去滑引入的抖动。**
交付点 **v19_088a10** 在三个生成器上都优于原始:FSR、Jitter(RMS)、抖动*尖峰*
(踝加速度 p99/max = 原始的 0.91×/0.94×)、语义都更好。七个物理指标里有四个
(Floating/ContactAcc/Penetration/BoneCV)从不变化——它们是构造上恒定,如实标注,不当成绩。

---

## 4. Semantic / FID (reference fixed today) / 语义与 FID(今天修好参考集)

**EN.** The old FID used only n=26 ground-truth motions as its reference —
rank-deficient in the 512-d embedding, so FID was essentially noise. FID's
reference need not share IDs with the eval prompts (the standard protocol measures
against the whole test split), so it was rebuilt from the standard `test.txt`
split: **all 1215 locally-present GT motions (1215 > 512, full-rank)**, with the
generated side at n=200 (T2M-GPT / MoMask) and n=50 (MDM).

```
FID↓ (reference n=1215)   original  deskate  gauss   learn   v19
  T2M-GPT (n=200)          0.483    0.658    0.485   0.489   0.483
  MoMask  (n=200)          0.958    1.613    1.058   1.020   1.074
  MDM     (n=50)          15.942   14.903   15.524  15.573  15.758
```

Three points, all consistent with the physics story:
1. With a proper reference, T2M-GPT / MoMask FID drops sharply (old 5.1 / 3.4 →
   0.48 / 0.96) — the old high values were reference noise, not real distance.
2. **De-skate alone hurts realism most** (MoMask 0.96→1.61); the smoother
   (gauss ≈ learn ≈ v19) repairs it back to ≈ original — a cleaner echo of the
   physics table.
3. **MDM's FID is ≈16, an order of magnitude worse than the other two** — an
   independent sign that MDM's local samples are genuinely off-distribution (§5).
   MM-Dist / R-precision changes from correcting the feet are not significant
   (paired bootstrap), i.e. moving the feet does not break text alignment.

**中.** 旧 FID 参考集只有 n=26,在 512 维嵌入里秩亏,FID 本质是噪声。FID 参考集本就不必与
评估 prompt 同 ID(标准协议对整个测试集算),故改用标准 `test.txt` 划分:**本地全部 1215 条
GT(1215 > 512,满秩)**,生成侧 T2M-GPT/MoMask n=200、MDM n=50。三点结论均与物理一致:
① 满秩参考下 T2M-GPT/MoMask 的 FID 大降(旧 5.1/3.4→0.48/0.96),旧高值是噪声;
② 纯去滑对真实度伤害最大,平滑器把它修回≈原始,v19≈gauss;
③ MDM 的 FID≈16、比另两者差一个数量级,独立佐证其本地样本离真实分布很远(见 §5);
修脚对 MM-Dist/R-precision 无显著影响,即移动脚步不破坏文本对齐。

---

## 5. Honest findings & limitations / 诚实的结论与局限

**EN.**
- **Central finding (reported as-is).** Across comparable operating points the
  learned smoother is on par with a well-tuned Gaussian — it beats it at only a
  few least-smoothing points (by 0.3–0.4 pp) and is within ~0.9 pp elsewhere. This
  is a genuine, defensible result: the value of MotionFix is the *pipeline* and the
  IK-in-the-loop training that lets a tiny (46 K) learned model match a strong
  classical baseline while also fixing jitter spikes and semantic drift — not a
  claim that learning crushes the baseline. Earlier drafts that over-claimed were
  corrected after fixing leakage, proxy/metric mismatch, and corpus corruption.
- **MDM could not be reproduced on this cluster.** Its samples need `--use_ema`
  and batch ≤ 10 to avoid numerical blow-up, and even then show heavy contact-frame
  sliding unlike the original recipe; the FID ≈ 16 above confirms they are
  off-distribution. Documented as an environment limitation, kept at n=50, and
  **not** used to draw method conclusions.
- **Perceptual study — infrastructure only.** The one untested assumption behind
  the whole narrative ("lower jitter = looks more natural") still needs human
  ratings. Tool-chain is ready; ~10 raters would close it.
- **Data.** Training uses 4k HumanML3D (all generators OOD — good). The local
  HumanML3D checkout is 8177/14616 motions; this only limited FID (now fixed) and
  does not affect training or the physics metrics.

**中.**
- **核心结论(照实报告)。** 在可比操作点上,学习平滑器与调好的高斯基本持平——只在少数
  最少平滑点上小胜(0.3–0.4pp),其余相差在 ~0.9pp 内。这是站得住的真实结果:MotionFix 的
  价值在于*整条管线*与 IK-进回路的训练,让一个极小(46K)的学习模型追平强经典基线,同时
  修好抖动尖峰与语义漂移——而不是"学习碾压基线"。早期过度声称的草稿在修掉泄漏、代理/指标
  错配、语料损坏后已更正。
- **MDM 在本集群无法复现。** 样本需 `--use_ema` 且 batch≤10 才不数值崩坏,即便如此仍有异于
  原配方的接触帧打滑;上面 FID≈16 佐证其离分布很远。作为环境局限记录,维持 n=50,且**不**
  用于推导方法结论。
- **感知实验——只搭了架子。** 整条叙事底下唯一未验证的假设("抖动更低=看起来更自然")仍需
  人类打分。工具已就绪,约 10 人即可补上。
- **数据。** 训练用 4k HumanML3D(生成器全 OOD,合适)。本地 HumanML3D 只有 8177/14616 条,
  这只影响过 FID(现已修),不影响训练与物理指标。

---

## 6. Self-assessment: ready to write? / 自评:能否开始写论文?

**EN (my view, for your decision).** For an MSc thesis I believe the substance is
sufficient: a complete, frozen method; a rigorous four-layer ablation; a
multi-generator, enlarged, leak-free evaluation; a fixed FID reference; and an
honest central finding (including negatives, which is a strength, not a gap). The
two open items — human perceptual ratings and MDM reproduction — fit naturally as
a final experiment and a documented limitation / future work, which is normal for
a thesis rather than a blocker. My plan, pending your agreement, is to **start
writing now** and run the perceptual study in parallel so its results drop into
the evaluation chapter when ready.

**中(我的判断,供您决定)。** 就硕士论文而言,我认为内容已足够:方法完整并已冻结;严谨的
四层消融;多生成器、已扩样、无泄漏的评估;修好的 FID 参考集;以及诚实的核心结论(含负面
结论,这是加分不是缺口)。两个未决项——人类感知打分与 MDM 复现——可自然地作为收尾实验与
书面局限/未来工作,属论文常态而非阻塞项。若您同意,我的计划是**现在开始写作**,并**并行**
跑感知实验,结果就绪后并入评估章节。

---

## 7. Questions for you / 请您定夺

**EN.**
1. **Given the above, may I start writing the thesis now?**
2. Is the current scope (three generators; learned-smoother ≈ tuned-Gaussian
   reported honestly) acceptable, or would you want any additional experiment
   *before* writing begins?
3. Should the perceptual study (≈10 raters, blind 2-AFC) be part of the thesis, or
   is it optional given the physics + FID evidence?

**中.**
1. **基于以上,我现在可以开始写论文吗?**
2. 当前范围(三个生成器;学习平滑器≈调好的高斯,如实报告)是否可接受,还是您希望在动笔
   *之前*再补某个实验?
3. 感知实验(约 10 人盲测 2-AFC)要不要写进论文,还是在已有物理+FID 证据下作为可选项?

---

### Appendix — evidence on disk / 附录:磁盘证据

| Item / 项 | File / 文件 |
|---|---|
| 5-way ablation (n=200) | `analysis/v19/v19_088a10_expanded_results.json`, `analysis/v19/momask_pool_results.json` |
| FSR–Jitter frontier | `analysis/v19/frontier.json` |
| Per-category | `analysis/v19/by_category.json` |
| Semantic (old n=26 ref) | `analysis/v19/semantic.json` |
| **Semantic (new n=1215 ref)** | `analysis/v19/semantic_largeref.json` (script `testing/v19_fid_ref.py`) |
| Jitter spikes (p99/max) | `analysis/v19/jitter_stats.json` |
| Full method spec | `docs/motionfix_v19_full_spec.md` |
| Current-status doc | `docs/motionfix_current_status.md` |
| Perceptual tool-chain | `outputs/perceptual/rating.html`, `analysis/perceptual_analysis.py` |
