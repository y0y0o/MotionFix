# 论文修改清单 / Thesis Revision Notes

> 对象:`Generator_Agnostic_Post_Hoc_Correction_of_Foot_Skating_in_Text_to_Motion_Generation.pdf`
> 日期:2026-08-31。所有"实测值"来自磁盘 `analysis/v19/*.json`,已用 088a10 交付点核对。
> 结论先说:**方法、framing、两张主消融表(Table 1/2)和 FID 表(Table 3)全部逐格吻合,不用改。
> 只有两处派生结论与磁盘数据冲突,必须改(§5.E 尖峰、§5.C 分类别);另有几处小建议。**

---

## A. 必须改(与磁盘数据冲突)

### A1. §5.E Jitter-spike analysis —— "0.91×/0.94× on all three generators" 是错的

**原文(第 10 页)**
> Relative to the original output, the delivery point reduces the 99th-percentile and
> maximum ankle acceleration to **0.91× and 0.94×** respectively, **on all three generators**.

**问题**
- 旧 `jitter_stats.json` 是**错交付点 v19_045**(脚本曾硬编码该 checkpoint)。已改到 088a10 重跑三个生成器。
- 实测 v19/original 的 **p99 / max**(`analysis/v19/spike_ratios_088a10.json`):

  | 生成器 | p99 | max |
  |---|---|---|
  | T2M-GPT (n=200) | 0.93× | 0.98× |
  | MoMask (n=200) | **1.03×** | **1.07×** |
  | MDM (n=50) | 0.90× | 0.83× |

- 即:T2M-GPT/MDM 降尖峰,**MoMask 的尖峰反而略升**。"on all three generators"不成立。

**建议替换文本**
> Relative to the original output, the delivery point reduces the 99th-percentile and
> maximum ankle acceleration on T2M-GPT (0.93×/0.98×) and MDM (0.90×/0.83×), while on
> MoMask the spikes are essentially unchanged to slightly higher (1.03×/1.07×). The
> spike behaviour is therefore generator-dependent: the pipeline removes worst-case
> twitches on two of the three generators and holds them roughly constant on the third.
> This does not affect the FSR and RMS-jitter results, which improve on all three
> generators (Tables 1–2); it refines only the worst-case-spike claim.

**证据**:`analysis/v19/spike_ratios_088a10.json`(脚本 `analysis/v19_spikes_3gen.py`)

---

### A2. §5.C Per-category generalisation —— "no category regresses" 和 "rotation largest residual" 都错

**原文(第 9 页)**
> First, **every category improves** on the original output, and **no category regresses** …
> Second, the **rotation** category retains the **largest residual FSR**.

**问题**(实测 `analysis/v19/by_category_088a10.json`,与旧 json 逐位一致 → 本来就是 088a10):

| 类别 | orig | deskate | v19 | |
|---|---|---|---|---|
| rotation (n=18) | 0.2199 | 0.1251 | 0.1752 | |
| walking (n=21) | 0.2076 | 0.0807 | 0.1121 | |
| **backward (n=9)** | 0.1897 | 0.1319 | **0.1964** | ← v19 **高于**原始(回归) |
| turning (n=21) | 0.1384 | 0.0980 | 0.1190 | |
| complex (n=30) | 0.1174 | 0.0738 | 0.0983 | |
| dance (n=30) | 0.0599 | 0.0312 | 0.0445 | |
| **jumping (n=21)** | 0.0334 | 0.0296 | **0.0369** | ← v19 **高于**原始(回归) |

- **backward、jumping 两类 v19 回归**(高于原始)——"no category regresses"不成立。
- **最大残留是 backward(0.196),不是 rotation(0.175)**——"rotation largest residual"不成立。
- (注:backward n=9 很小,可在正文标注样本量。)

**建议替换文本**
> Most categories improve on the original output; the two exceptions are *backward*
> (0.190 → 0.196, n = 9) and *jumping* (0.033 → 0.037), where the smoother gives back
> enough FSR to slightly exceed the original — a category-level echo of the frontier
> trade-off in Section 5.2. The largest residual FSR is in the *backward* category
> (0.196), followed by *rotation* (0.175). Rotation nonetheless shows the largest
> *absolute reduction* from the physics de-skate stage (0.220 → 0.125), consistent
> with the plant-at-mean-XZ model being most effective on translationally planted
> motions; the reach-clamp triggers most often on rotating and backward motions,
> which is where the residual concentrates.

**同时检查**:§5.C 若其他地方还写了"rotation 最难/最大残留",一并按上表改。

**证据**:`analysis/v19/by_category_088a10.json`(脚本 `analysis/v19_by_category.py`)

---

## B. 建议核对(不一定错,但写正文前确认)

### B1. §5.B FSR–jitter frontier 的样本量
- 前沿数据 `analysis/v19/frontier.json` 的 key 是 **T2M-GPT (n=50)** / MoMask (held-out, n=10) / MDM (n=50)。
- 主表 Table 1/2 是 **n=200**。前沿结论(学习≈高斯)本身没问题,但正文若让读者以为前沿也是 n=200,需注明前沿扫描用的是较小样本,或重跑前沿到 n=200。
- **建议**:在 §5.B 加一句"the frontier sweep uses n = 50 (T2M-GPT) / n = 10 (MoMask held-out) / n = 50 (MDM)",避免与主表 n 混淆。

### B2. Abstract / §5.4 里的 "0.48 and 0.96"
- 实测 FID(`semantic_largeref.json`):T2M-GPT original **0.483**,MoMask original **0.958**。
- 摘要写的 "0.48 and 0.96" 对(四舍五入)。✅ 无需改,仅确认。

### B3. §5.E 里 RMS 与 Table 里 Jitter 的定义不同(不是错,避免读者混淆)
- §5.E 尖峰脚本的 RMS(0.015 量级,ankle 3D accel)与 Table 1/2 的 Jitter(0.009 量级)**是两套不同定义**(不同关节集/归一化)。两者都自洽,但如果正文同时出现两个"jitter/RMS"数值,建议一句话说明尖峰分析用的是独立的 ankle-acceleration 轨迹度量,只看相对比值。

### B4. V8 基线数字冲突(如果论文正文/附录引用了 V8)
- `CLAUDE.md` 说 V8 降 2.9%,`docs/research_log_20260624.md` 说 14.1%→15.6%(变差)。**方向相反**。
- 若论文任何地方引了 V8 数字,写之前必须确认哪个对。当前 PDF 我没看到 V8,若确实没引可忽略。

---

## C. 已核对无误(不用改,列出以放心)

| 论文处 | 磁盘来源 | 状态 |
|---|---|---|
| Table 1 (T2M-GPT n=200) 全 5 行 FSR/Jitter | `v19_088a10_expanded_results.json` → t2mgpt | ✅ 逐格吻合 |
| Table 2 (MoMask n=200) 全 5 行 | `momask_pool_results.json` | ✅ 逐格吻合 |
| Table 3 (FID n=1215) 3×5 全格 | `semantic_largeref.json`(refN=1215,dim=512) | ✅ 逐格吻合 |
| 去滑相对降幅 38%(T2M-GPT)/40%(MoMask) | 反算 | ✅ |
| FID 0.96→1.61(MoMask de-skate) | `semantic_largeref.json` | ✅ |
| 46,276 参数、IK 2.2e-16、088a10(jit 0.88/anch 0.10) | 方法文件 | ✅ |
| MDM n=50、FID≈16、不用于方法结论 | `semantic_largeref.json` | ✅ 诚实 |
| §4.4 FID 参考 26→1215 满秩修复 | `testing/v19_fid_ref.py` | ✅ |
| §4.2 四指标 invariant-by-construction | `testing/v19_eval.py` | ✅ |
| §4.6 感知实验只搭工具、列为 future work | 实际一致 | ✅ 诚实 |

---

## D. 修改顺序建议

1. 先改 **A1、A2**(硬冲突,评审一查数据就穿)。用上面的"建议替换文本"直接替。
2. 再顺手补 **B1、B3** 各一句(消除 n 与 RMS 定义的歧义)。
3. **B4** 仅当正文引了 V8 时才处理。
4. 改完这两处后,论文与磁盘数据完全一致。

**新增/改动的可复现脚本(已提交到仓库)**
- `analysis/v19_jitter_trace.py` —— checkpoint 修正为 v19_088a10
- `analysis/v19_spikes_3gen.py` —— 三生成器尖峰比值(新)
- `analysis/v19_by_category.py` —— 分类别 FSR 重建(新,补回丢失的脚本)
- 输出:`analysis/v19/spike_ratios_088a10.json`、`analysis/v19/by_category_088a10.json`
- 旧文件备份:`analysis/v19/jitter_stats_v045.json.bak`
