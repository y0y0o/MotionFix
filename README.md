# MotionFix: Post-Hoc Foot-Skating Correction for VQ-Based Motion Generation

**MotionFix** is a lightweight, generator-agnostic **post-processing** pipeline that removes **foot-skating** artifacts from text-to-motion generators (MoMask, T2M-GPT, MDM) **without retraining** them. It reduces both *foot-skating* and *jitter* while keeping bone lengths rigid (no leg-tear, no foot-flip) and feet in contact.

<p align="center">
  <b>English</b> | <a href="#motionfix-基于-vq-的动作生成脚步滑动后处理修正">中文</a>
</p>

---

## Overview

VQ-VAE motion generators (MoMask, T2M-GPT) are **15–20× faster** than diffusion models but suffer from **foot-skating** — feet slide when they should stay planted. Naive fixes trade one artifact for another: hard-planting the foot kills skating but injects **jitter**; global smoothing kills jitter but re-introduces skating. MotionFix treats correction as a **constraint-satisfaction problem** and combines physics with a small learned model:

```
        ┌─────────────┐     ┌────────────────────┐     ┌──────────────┐
input → │  De-skate   │  →  │  Learned adaptive   │  →  │  2-bone IK   │ → output
(T,22,3)│  (physics)  │     │  smoother (learning)│     │  (physics)   │  (T,22,3)
        └─────────────┘     └────────────────────┘     └──────────────┘
         plant-at-mean        rounds plant→air           hip→knee→ankle
         (no drift, low FSR)  boundaries (low jitter)     + rigid toe
                                                          (rigid bones)
```

- **De-skate (physics):** plant the foot at its per-contact-segment mean XZ — removes skating with *no integration drift* (target stays reachable).
- **Learned smoother (learning):** a 48.8K-param 1D-CNN that adaptively rounds the sharp plant→air velocity boundaries to cut jitter, without re-introducing skating. Trained on MoMask held-out; **generalizes to MDM and T2M-GPT unchanged**.
- **2-bone IK (physics):** analytic cosine-rule solve on the leg chain (hip→knee→ankle) with a rigid toe. Clamps the ankle target to leg reach — this is what **fixes leg-tear and foot-flip** (bones stay rigid).

## Results (n = 50 per generator)

The **same** pipeline (learned smoother trained only on MoMask) applied post-hoc to three generators:

| Generator | FSR ↓ | Jitter ↓ | BoneCV (bones) | ContactAcc |
|-----------|:-----:|:--------:|:--------------:|:----------:|
| **MoMask** | 14.1% → **12.7%** | 0.0128 → **0.0107** | 0.0037 → 0.0037 | 100% |
| **MDM** | 11.9% → **11.2%** | 0.0142 → **0.0115** | 0.0149 → 0.0149 | 100% |
| **T2M-GPT** | 12.0% → **11.0%** | 0.0139 → **0.0107** | 0.0294 → 0.0294 | 100% |

Every generator: **FSR and Jitter both drop, bone lengths unchanged (no leg-tear / foot-flip), 100% contact preserved.**

**Honest finding.** Once the skeleton is IK-constrained, FSR ↔ Jitter is an *irreducible* trade-off: the learned smoother traces the **same FSR–Jitter frontier** as a tuned Gaussian (it lets you pick the operating point and beats the original on both axes, but does not Pareto-dominate the analytical filter). This is consistent with the framing that post-hoc foot-skating correction is a constraint-satisfaction problem whose ceiling is set by skeletal reachability, not the generator.

See `analysis/v18_ik_scale/results.png` for the 4-panel figure and `docs/v18_devlog.md` for the full derivation (including three learned-smoother training failures and their fix).

## Evaluation Metrics (7)

`utils/metrics.py`: **FSR** (foot-skating ratio), **Jitter** (foot-accel RMS), **Floating**, **FootErr** (deviation vs original), **ContactAcc**, **BoneCV** (bone-length consistency), **Penetration**.

## Project Structure

```
motionfix/
├── models/            # Model definitions (v8–v18, v18_ik = the IK core)
├── data/
│   ├── prep/          # Training-pair generation (v1–v15)
│   ├── datasets/      # PyTorch Dataset classes
│   ├── training/      # Training data (.npy — git-ignored)
│   └── test_inputs/   # Generator outputs: momask_50 / mdm / t2mgpt (git-ignored)
├── training/          # Training entry-points (training/v18_ik.py = final)
├── testing/           # Eval scripts (testing/v18ik_scale.py = cross-generator)
├── analysis/          # Analysis scripts + result figures (*.png tracked)
├── utils/             # metrics.py, physics_fix.py, render_v18ik.py
├── checkpoints/       # Weights (.pth — git-ignored)
├── outputs/           # Fixed motions / videos (git-ignored)
├── docs/              # v18_devlog.md, research logs
└── CLAUDE.md          # Detailed dev guidance
```

## Quick Start

Environment: `conda activate t2mgpt` (PyTorch 2.1.0+cu121). Run all scripts from the repo root.

```python
import torch, numpy as np
from models.v18 import FootRefiner, smooth_fix
from models.v18_ik import apply_ik

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FootRefiner().to(device)
model.load_state_dict(torch.load("checkpoints/v18_ik/best.pth", map_location=device)['model_state_dict'])
model.eval()

motion = np.load("your_motion.npy").astype(np.float32)   # (T, 22, 3)
fixed  = apply_ik(motion, smooth_fix(motion, model, device))   # de-skate → smooth → IK
np.save("your_motion_fixed.npy", fixed)
```

### Train / Evaluate / Render

```bash
python training/v18_ik.py       # train the adaptive smoother → checkpoints/v18_ik/best.pth
python testing/v18ik_scale.py   # cross-generator eval (MoMask/MDM/T2M-GPT, n=50) → analysis/v18_ik_scale/
python analysis/make_results_chart.py   # 4-panel results figure
python utils/render_v18ik.py    # side-by-side Original vs Learned+IK videos → outputs/videos/v18_ik/
```

## Method Evolution

| Stage | Versions | Lesson |
|-------|----------|--------|
| Diagnosis | V8–V17 | Why learning fails: FSR↔Jitter antagonism, metric gaming, jitter blow-up |
| Break the antagonism | V18 | Velocity-space contact mask → low FSR *and* low jitter, but `cumsum` integration **drift** (FootErr 0.39 m) |
| **Final** | **V18 + 2-bone IK** | De-skate (no drift) + IK (rigid bones) + direct-objective learned smoother → FSR↓, Jitter↓, bones intact |

## Data

Training pairs generated from [HumanML3D](https://github.com/EricGuo5513/HumanML3D). All motions are **22-joint, 3D world-coordinate** positions, shape `(T, 22, 3)`. Foot joints: **7, 8 (ankles), 10, 11 (toes)**.

## Citation

Part of ongoing MSc research at Durham University.

```bibtex
@misc{wan2026motionfix,
  title={MotionFix: Post-Hoc Foot-Skating Correction for VQ-Based Motion Generation},
  author={Wan, Xin},
  year={2026},
  note={Durham University}
}
```

---

# MotionFix: 基于 VQ 的动作生成脚步滑动后处理修正

**MotionFix** 是一个轻量、与生成器无关的**后处理**管线,用于消除文本到动作生成器(MoMask、T2M-GPT、MDM)的**脚步滑动**伪影,**无需重新训练**生成器。它在同时降低*脚滑*与*抖动*的同时,保持骨长刚性(不扯腿、不翻脚)、脚部接触不变。

## 概述

基于 VQ 的生成器比扩散模型快 **15–20 倍**,但存在**脚步滑动**——脚该踩住时却在滑。朴素修法会拆东补西:硬钉脚消除滑动却引入**抖动**;全局平滑消除抖动却重新引入滑动。MotionFix 把修正视为**约束满足问题**,用物理 + 小型学习模型结合:

```
输入 → 去滑(物理) → 学习自适应平滑(学习) → 2-bone IK(物理) → 输出
       plant-at-mean   圆滑 踩→抬 边界        髋→膝→踝 + 刚性脚趾
       (无漂移,低FSR)  (低抖动)              (骨骼刚性)
```

- **去滑(物理):** 把脚钉在每个接触段的 XZ 均值——消除滑动且**无积分漂移**(目标可达)。
- **学习平滑(学习):** 4.88 万参数的 1D-CNN,自适应圆滑"踩→抬"的速度突变以降抖动,又不重新引入滑动。**仅在 MoMask 上训练,直接泛化到 MDM 与 T2M-GPT**。
- **2-bone IK(物理):** 腿链(髋→膝→踝)解析余弦定理求解 + 刚性脚趾,将踝目标钳制到腿长可达范围——这是**修复扯腿与翻脚**的关键(骨长保持刚性)。

## 结果(每个生成器 n = 50)

**同一个**管线(平滑器仅在 MoMask 上训练)后处理三个生成器:

| 生成器 | FSR ↓ | Jitter ↓ | BoneCV(骨长) | ContactAcc |
|--------|:-----:|:--------:|:------------:|:----------:|
| **MoMask** | 14.1% → **12.7%** | 0.0128 → **0.0107** | 0.0037 → 0.0037 | 100% |
| **MDM** | 11.9% → **11.2%** | 0.0142 → **0.0115** | 0.0149 → 0.0149 | 100% |
| **T2M-GPT** | 12.0% → **11.0%** | 0.0139 → **0.0107** | 0.0294 → 0.0294 | 100% |

每个生成器:**FSR 与 Jitter 同时下降,骨长不变(不扯腿/翻脚),接触 100% 保持。**

**诚实的发现。** 骨架被 IK 约束后,FSR ↔ Jitter 是**不可消除**的权衡:学习平滑器与调好的高斯滤波落在**同一条 FSR–Jitter 前沿**上(它让你自选工作点、双指标击败原始,但不 Pareto 碾压解析滤波)。这印证了"后处理脚滑修正是约束满足问题,天花板由骨架可达性决定,而非生成器"的论断。

四面板图见 `analysis/v18_ik_scale/results.png`,完整推导(含学习平滑器三次训练失败与修复)见 `docs/v18_devlog.md`。

## 评测指标(7 个)

`utils/metrics.py`:**FSR**(脚滑率)、**Jitter**(脚部加速度 RMS)、**Floating**(浮空)、**FootErr**(相对原始的偏移)、**ContactAcc**(接触准确率)、**BoneCV**(骨长一致性)、**Penetration**(穿地)。

## 项目结构

```
motionfix/
├── models/            # 模型定义(v8–v18,v18_ik = IK 核心)
├── data/
│   ├── prep/          # 训练数据对生成(v1–v15)
│   ├── datasets/      # PyTorch 数据集类
│   ├── training/      # 训练数据(.npy,git 忽略)
│   └── test_inputs/   # 生成器输出:momask_50 / mdm / t2mgpt(git 忽略)
├── training/          # 训练入口(training/v18_ik.py = 最终)
├── testing/           # 评测脚本(testing/v18ik_scale.py = 跨生成器)
├── analysis/          # 分析脚本 + 结果图(*.png 跟踪)
├── utils/             # metrics.py, physics_fix.py, render_v18ik.py
├── checkpoints/       # 权重(.pth,git 忽略)
├── outputs/           # 修正动作 / 视频(git 忽略)
├── docs/              # v18_devlog.md, 研究日志
└── CLAUDE.md          # 详细开发指南
```

## 快速开始

环境:`conda activate t2mgpt`(PyTorch 2.1.0+cu121)。所有脚本从仓库根目录运行。

```python
import torch, numpy as np
from models.v18 import FootRefiner, smooth_fix
from models.v18_ik import apply_ik

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FootRefiner().to(device)
model.load_state_dict(torch.load("checkpoints/v18_ik/best.pth", map_location=device)['model_state_dict'])
model.eval()

motion = np.load("your_motion.npy").astype(np.float32)   # (T, 22, 3)
fixed  = apply_ik(motion, smooth_fix(motion, model, device))   # 去滑 → 平滑 → IK
np.save("your_motion_fixed.npy", fixed)
```

### 训练 / 评测 / 渲染

```bash
python training/v18_ik.py       # 训练自适应平滑器 → checkpoints/v18_ik/best.pth
python testing/v18ik_scale.py   # 跨生成器评测(MoMask/MDM/T2M-GPT,n=50)→ analysis/v18_ik_scale/
python analysis/make_results_chart.py   # 四面板结果图
python utils/render_v18ik.py    # 并排 Original vs Learned+IK 视频 → outputs/videos/v18_ik/
```

## 方法演进

| 阶段 | 版本 | 教训 |
|------|------|------|
| 诊断 | V8–V17 | 学习为何失败:FSR↔Jitter 对抗、指标作弊、抖动爆炸 |
| 打破对抗 | V18 | 速度空间接触掩码 → FSR 与抖动同时低,但 `cumsum` 积分**漂移**(FootErr 0.39 m) |
| **最终** | **V18 + 2-bone IK** | 去滑(无漂移)+ IK(骨骼刚性)+ 直接目标学习平滑器 → FSR↓、Jitter↓、骨骼完好 |

## 引用

杜伦大学硕士研究的一部分。

```bibtex
@misc{wan2026motionfix,
  title={MotionFix: Post-Hoc Foot-Skating Correction for VQ-Based Motion Generation},
  author={Wan, Xin},
  year={2026},
  note={Durham University}
}
```
