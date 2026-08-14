# MotionFix 当前版本 —— 我做了哪些东西

> 生成日期:2026-08-13。所有数字来自 `analysis/v19/*.json` 的实际运行输出,不是估计。
> 目的:一页看清「当前最新版本是什么、消融做了没有、还有什么没做」。

---

## 0. 一句话

**当前交付版本 = `checkpoints/v19_088a10`。消融实验已完成**,是四层的:
①五路主消融(隔离每一级)、②FSR–Jitter 前沿扫描(学习 vs 高斯)、③分动作类别、④语义保真。
原始结果分别在 `v19_088a10_results.json` / `frontier.json` / `by_category.json` / `semantic.json`。

---

## 0.5 评估样本扩充(2026-08-13)

为增强评估的统计强度,把 **T2M-GPT 从 50 条扩到 200 条**(用 VQVAE+Transformer 对
HumanML3D 测试集另选 150 个 prompt 重新生成,链路与原 50 条完全一致:去归一化 + `recover_from_ric`;
逐条验证脚速/形状正常,footV_mean 0.016 甚至比旧样本更平滑,0 条异常)。

- **MDM 尝试扩充但失败,已回退到原始 50。** 排查过程(记录以备将来):
  1. 第一次(200k,无 `--use_ema`)→ 加载了非 EMA 原始权重 → 脚速 ~10×、飞脚 2.5m、穿地(垃圾)。
  2. 加 `--use_ema` 后飞脚消失,但 **batch=50 仍批级数值崩坏**(NaN/飞脚);**batch=10 才位置正常**。
  3. 但 batch=10 的样本**仍有严重脚滑**:FSR **~94%** vs 原始 50 条的 ~12%——脚看着贴地却在接触时
     持续打滑,与原始不同质,**不可用**。仍有未查明的生成配方差异(guidance / 接触后处理 / MDM 变体)。
  4. `model000600000 + --use_ema` 段错误硬崩溃(无输出);gpu3 节点加载 t2m 数据集时频繁段错误/挂起。
  **结论:MDM 在本环境无法复现原始质量,保持原始 50 条。**(细节见项目记忆 `mdm-generation-gotchas`。)
- **MoMask 保持 50**(本地无 MoMask 仓库,无法再生成)。

**净结果:只有 T2M-GPT 成功扩充(50→200);MDM、MoMask 维持 50。**

**当前评估样本:MoMask 50 / MDM 50 / T2M-GPT 200。** 扩充结果见 `v19_088a10_expanded_results.json`。

**T2M-GPT n=50 → n=200 主表(五路消融,FSR↓ / Jitter↓):**

```
                FSR(n=50)  FSR(n=200) | Jit(n=50)  Jit(n=200)
original         11.99%      6.43%    |  0.01388    0.00916
deskate_ik        6.65%      3.98%    |  0.02703    0.01655
gauss_ik          8.83%      4.92%    |  0.01217    0.00778
learn_ik(V18)    10.99%      6.00%    |  0.01070    0.00704
v19_ik            8.93%      5.08%    |  0.01283    0.00827
```

**结论在 4× 样本下全部保住**:
1. **物理去滑仍是降 FSR 的主力**:original 6.43% → deskate 3.98%(降幅最大的一步)。
2. **平滑器仍对 FSR 做负功**:相对 deskate,gauss +0.94pp、learn +2.02pp、v19 +1.10pp。
3. **v19(5.08%)≈ 调好的高斯(4.92%)**——学习平滑器无稳定优势,与主结论一致。

**注意样本组成变了**:绝对 FSR 从 ~12% 降到 ~6% 是因为新 150 个 prompt 更"平"(原 50 条像是
偏脚滑高发的走/转/旋转类,新集更广更杂)。**相对排序与核心结论不变**——写论文时应说明
n=200 是在更广、更有代表性的样本上复核了结论,而非在同一分布上收紧估计。

---

## 1. 当前版本是什么

三段式后处理管线,输入是任意文本→动作生成器的输出,不重训生成器:

```
input → 去滑(物理) → 学习平滑器(学习) → 2骨IK(物理) → output
        plant-at-mean XZ   46.3K 1D-CNN,只改踝    髋→膝→踝 + 刚性脚趾
        (无漂移,低FSR)     (IK 在训练回路里)      (骨长刚性,接触保证)
```

- 三段里**只有中间那段是神经网络**;去滑和 IK 是纯物理、零参数。
- 交付点 `v19_088a10`:`--jit-share 0.88 --anch-share 0.10`,是 FSR–Jitter 前沿上偏低抖动的一个操作点。
- 代码:`models/v19.py`(V19Smoother / V19Loss / torch_ankle_ik)、`training/v19.py`、`data/prep/v19.py`。

---

## 2. 消融实验(你问的核心)——做了,四层

### 2.1 主消融:五路 × 3 生成器 × 7 指标

文件 `analysis/v19/v19_088a10_results.json`。每一路各去掉/替换一个组件,用来回答**每一级到底起不起作用**(直接对应导师第 2 点):

| 路 | 组成 | 含义 |
|---|---|---|
| `original` | 不修 | 基线 |
| `deskate_ik` | 去滑 + IK,**无学习** | 纯物理能到什么程度 |
| `gauss_ik` | 去滑 + **高斯平滑** + IK | 把学习换成非学习平滑器 |
| `learn_ik` | 去滑 + **V18 学习平滑** + IK | 旧学习器 |
| `v19_ik` | 去滑 + **V19 学习平滑** + IK | 当前方法 |

实测(FSR ↓ / Jitter-RMS ↓ / FootErr ↓):

```
                FSR      Jitter    FootErr
── MoMask held-out (n=10) ──
original      16.29%   0.01403    0.0000
deskate_ik     8.18%   0.02948    0.0424
gauss_ik      11.12%   0.01311    0.0481
learn_ik      12.85%   0.01143    0.0522
v19_ik        11.63%   0.01343    0.0432
── MDM (n=50) ──
original      11.86%   0.01417    0.0000
deskate_ik     7.04%   0.02850    0.0252
gauss_ik       9.06%   0.01363    0.0313
learn_ik      11.18%   0.01146    0.0367
v19_ik         9.24%   0.01363    0.0250
── T2M-GPT (n=50) ──
original      11.99%   0.01388    0.0000
deskate_ik     6.65%   0.02703    0.0293
gauss_ik       8.83%   0.01217    0.0350
learn_ik      10.99%   0.01070    0.0399
v19_ik         8.93%   0.01283    0.0292
```

**这张表消融出的事实**:
- **FSR 主要由物理(去滑)完成**:`deskate_ik` 就把 FSR 从 ~12% 压到 6.65–8.18%,是降幅最大的一步。
- **纯去滑抖动最大**(Jitter 0.027–0.029,约 2× 原始):平滑器/IK 之后的存在就是为了补这个。
- **平滑器降的是抖动、代价是 FSR 略回升**:`v19_ik` 相对 `deskate_ik`,Jitter 回落一半,FSR 回升 2–3pp。
- **学习平滑器 vs 高斯**:两者非常接近(如 T2M-GPT 8.93% vs 8.83%),学习没有稳定优势——这是诚实结论,不是 bug。

### 2.2 前沿扫描:学习 vs 高斯,不是单点比

文件 `analysis/v19/frontier.json` + `frontier.png`,三个生成器各一条 FSR–Jitter 前沿。
把 `--jit-share` 扫成一串操作点,证明"学习≈调好的高斯"不是靠挑一个点得出的,而是**整条前沿基本重叠**。脚本 `analysis/v19_frontier.py`。

### 2.3 分动作类别消融

文件 `analysis/v19/by_category.json`,7 类:rotation / walking / backward / turning / complex / dance / jumping,每类比 `original` vs `deskate_ik` vs `v19_ik`(对应导师第 4 点"跨类别泛化")。结论:**每一类都改善,无一类变差**;旋转类残留最高(踝持续移动,reach-clamp 触发最频繁)。

### 2.4 语义保真消融

文件 `analysis/v19/semantic.json`,HumanML3D `text_mot_match` 评估器(FID / R-precision / MM-Dist / Diversity),五路 × 3 生成器,配对 bootstrap(对应导师第 7 点的语义那一半)。MM-Dist(越低越贴合文本,原始 ≈ 2.6–3.1):

```
              MoMask   MDM    T2M-GPT
deskate_ik    2.812   2.537   3.103
gauss_ik      2.649   2.535   2.994
v19_ik        2.636   2.533   2.959
```

**修脚基本不动语义**,三个生成器 MM-Dist 变化均不显著;`deskate_ik`(纯去滑)反而是最差的一路 → 平滑器也顺带修了一点语义漂移。脚本 `testing/v19_semantic.py`。

---

## 3. 除消融外,当前版本还做了哪些

| 项 | 做了什么 | 位置 |
|---|---|---|
| **数据** | 从**原始** HumanML3D `new_joint_vecs` 重建语料(修掉 v14 的反归一化 bug);按接触占比 [0.2,0.9] 过滤,保留 ~3.4k 条 | `data/prep/v19.py` |
| **无泄漏** | 训练用 4k HumanML3D → MoMask/MDM/T2M-GPT **全部 out-of-distribution**,结构上不可能泄漏 | `training/v19.py` B7 |
| **IK 进训练回路** | 可微踝钳制放进前向,loss 算在真正被评估的轨迹上(与 numpy IK 精度 2.2e-16) | `models/v19.py::torch_ankle_ik` |
| **损失对齐指标** | 反脚滑项改成阈值对齐的 soft-count `sigmoid((v-0.03)/τ)`,不再是均值代理 | `models/v19.py::V19Loss` |
| **λ 自动标定** | 按每项梯度范数标定权重,消除量纲问题(acc²~1e-6 vs 软计数~1e-1) | `training/v19.py::calibrate_lambdas` |
| **验证集选点** | 按验证 loss 存 checkpoint(V18 是按训练 loss、无验证集) | `training/v19.py` B3 |
| **尖峰分析** | 加 p99/max/尖峰率 + 逐帧曲线(Jitter-RMS 会掩盖抽搐) | `analysis/v19_jitter_trace.py` |
| **7 指标评估** | FSR/Jitter/Floating/FootErr/ContactAcc/BoneCV/Penetration,无泄漏协议 | `testing/v19_eval.py` |

---

## 4. 还没做的(缺口)

1. **视觉确认**(只有你能做):`outputs/videos/v19/*.mp4` 现在渲染的是**旧交付点 v19_045**,不是 088a10。要看当前点得把 `utils/render_v19.py` 的 checkpoint 改成 `v19_088a10` 重渲。
2. **尖峰统计需对 088a10 重跑**:磁盘上 `jitter_stats.json` 记录的是 v19_045(v19_ik p99=0.101,约 1.54× 原始);CLAUDE.md 记 088a10 为 0.91×/0.94×,但该 json **未被覆盖**。要引用 088a10 的尖峰数,得重跑 `analysis/v19_jitter_trace.py`。
3. **感知实验**:没做。"Jitter 低 = 看起来更自然"是整条权衡叙事底下未验证的假设。
4. **四个恒定指标**:Floating/ContactAcc/Pen/BoneCV 在所有路里都不变(构造保证),7 指标实际只有 3 个在动——应写成"invariant by construction",别当成绩。
5. **样本量**:语义 n=50、FID 参考集 n=26(本地 HumanML3D 只有 8177/14616 条)。
6. **V8 基线数字冲突**:`CLAUDE.md` 说降 2.9%,`research_log` 说 14.1%→15.6%(变差)。写进正文前需确认。

---

## 5. 一句话回答"消融做了没"

**做了。** 五路主消融 + 前沿扫描 + 分类别 + 语义,四层都在磁盘上有原始 json。
它们共同证明的事实是:**FSR 的降幅主要来自物理去滑;学习平滑器只负责补抖动,且其效果与调好的高斯基本持平。**
