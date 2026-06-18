# MotionFix: Foot Skating Correction for VQ-Based Motion Generation

**MotionFix** is a lightweight post-processing neural network that corrects **foot skating artifacts** in VQ-based text-to-motion generation models (MoMask, T2M-GPT). It operates as a plug-and-play module — no retraining of the underlying generation models is required.

<p align="center">
  <b>English</b> | <a href="#motionfix-基于-vq-的动作生成脚步滑动修正">中文</a>
</p>

---

## Overview

VQ-VAE-based motion generation models (MoMask, T2M-GPT) offer **15–20× faster inference** than diffusion-based approaches, making them ideal for real-time applications. However, they suffer from **foot skating** — feet slide along the ground when they should stay planted during contact phases. MotionFix addresses this with a Transformer-based correction network.

### Key Features

- **🔥 Selective Correction** — Only modifies foot joints on skating frames; leaves other body parts untouched
- **⚡ Lightweight** — 19.1M parameters, ~100ms per sequence on GPU
- **🔌 Plug-and-Play** — Works with any VQ-based motion generator; no retraining needed
- **🧠 Soft Gating (V9)** — Learned gate network + heuristic cues (height, velocity) for precise artifact localization
- **🦿 Kinematic IK (V9)** — Leg-chain inverse kinematics ensures knee positions are consistent with corrected ankles
- **📐 Bone Length Preservation** — Auxiliary loss maintains anatomical consistency

## Architecture

```
Input (T, 22, 3)
     │
     ▼
Flatten (T, 66)
     │
     ▼
Linear Proj → 512d
     │
     ▼
Positional Encoding (sinusoidal)
     │
     ▼
Transformer Encoder ×6
  (d_model=512, nhead=8, FFN=2048)
     │
     ▼
Output Proj → 66d
     │
     ▼
Soft Gating + Selective Blend (V9)
     │
     ▼
Leg-Chain IK (inference only)
     │
     ▼
Corrected Motion (T, 22, 3)
```

### Skating Ratio

| Model | Type | Avg Skating Ratio |
|-------|------|-------------------|
| MDM | Diffusion | Lowest (baseline) |
| MoMask (no IK) | VQ-based | ~13.6% |
| MoMask (IK) | VQ-based | ~12.7% |
| **MoMask + MotionFix** | **VQ + Post-process** | **Significantly reduced** |

## Project Structure

```
motionfix/
├── motionfix_model.py       # V8: Selective foot replacement (baseline)
├── motionfix_model_v9.py    # V9: Soft gating + temporal smooth + kinematic IK
├── motionfix_v6.py          # V6: Earlier iteration
├── train.py                 # V8 training script
├── train_v9.py              # V9 training script
├── dataset.py               # PyTorch Dataset for motion pairs
├── prepare_data.py          # Generate training pairs from HumanML3D
├── prepare_data_v2.py       # V2 data preparation
├── test.py                  # V8 evaluation on MoMask outputs
├── test_v3.py / test_v5.py / test_v9.py  # Version-specific test scripts
├── test_mdm.py / test_momask.py / test_t2mgpt.py  # Multi-model benchmarks
├── diagnose.py              # Debug/diagnostic tool
├── visualize_comparison.py  # Motion visualization
├── autosync.py              # Auto-commit & push watchdog
├── sync.sh                  # Launcher script for autosync
│
├── motionfix_v10/           # V10 experiments (latest iteration)
├── motionfix_v11/           # V11 experiments (work in progress)
│
├── fixed_outputs*/          # Corrected motion outputs (various versions)
├── momask_results/          # Raw MoMask outputs for comparison
├── t2mgpt_raw_joints/       # Raw T2M-GPT outputs
├── mdm_raw_joints/          # Raw MDM outputs
│
└── checkpoints*/            # Model weights (*.pth, excluded from git)
```

## Quick Start

### Requirements

```bash
pip install torch numpy scipy
```

### Fix a Motion

```python
import torch
import numpy as np
from motionfix_model_v9 import MotionFixNetworkV9

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MotionFixNetworkV9(blend_alpha=0.5, temperature=0.01).to(device)
ckpt = torch.load("checkpoints_v9/best.pth", map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load a motion: shape (T, 22, 3)
motion = np.load("your_motion.npy")  # (T, 22, 3)

# Fix it
T = motion.shape[0]
motion_flat = motion.reshape(T, -1)  # (T, 66)
motion_tensor = torch.FloatTensor(motion_flat).unsqueeze(0).to(device)

with torch.no_grad():
    fixed = model(motion_tensor, foot_only=True)

fixed_motion = fixed.squeeze(0).cpu().numpy().reshape(T, 22, 3)
np.save("your_motion_fixed.npy", fixed_motion)
```

### Evaluate Skating Ratio

```python
from test import detect_skating
sr_before = detect_skating(motion)
sr_after = detect_skating(fixed_motion)
print(f"Skating: {sr_before:.1%} → {sr_after:.1%}")
```

### Train from Scratch

```bash
# Step 1: Prepare training data
python prepare_data_v2.py

# Step 2: Train
python train_v9.py

# Step 3: Evaluate
python test_v9.py
```

## Model Versions

| Version | Key Innovation | Status |
|---------|---------------|--------|
| V3 | Basic Transformer encoder, L1 + velocity + foot loss | Baseline |
| V5 | Added bone length preservation loss | Improved |
| V6 | Architecture refinements | Iteration |
| V7 | Hyperparameter tuning | Iteration |
| **V8** | **Selective foot replacement (train full, infer partial)** | **Stable** |
| **V9** | **Soft gating + temporal smoothing + kinematic IK** | **Current best** |
| V10 | Further experiments | In progress |
| V11 | Iterative refinement (3-pass) | In progress |

## Training Data

Training pairs are generated from the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset. Three distortion types are applied to create (distorted, clean) pairs:

1. **Temporal Smoothing** — Gaussian filter on joint trajectories (window 3–7)
2. **Y-Shift** — Random vertical translation (±5cm)
3. **Spatial Noise** — Gaussian noise (σ = 0.005–0.02)

## Loss Function (V9)

$$\mathcal{L} = \mathcal{L}_{L1} + \lambda_{vel}\mathcal{L}_{vel} + \lambda_{foot}(\mathcal{L}_{foot} + \mathcal{L}_{foot\_vel}) + \lambda_{bone}\mathcal{L}_{bone} + 0.5\mathcal{L}_{contact\_vel}$$

| Term | Purpose |
|------|---------|
| $\mathcal{L}_{L1}$ | Overall reconstruction |
| $\mathcal{L}_{vel}$ | Temporal smoothness |
| $\mathcal{L}_{foot}$ | Foot position accuracy |
| $\mathcal{L}_{foot\_vel}$ | Foot velocity consistency |
| $\mathcal{L}_{bone}$ | Bone length preservation (22 pairs) |
| $\mathcal{L}_{contact\_vel}$ | Zero-velocity during ground contact |

## License & Citation

This project is part of ongoing research at Durham University. If you use MotionFix in your work, please cite:

```bibtex
@misc{wan2026motionfix,
  title={MotionFix: Foot Skating Correction for VQ-Based Motion Generation},
  author={Wan, Xin},
  year={2026},
  note={Durham University}
}
```

---

# MotionFix: 基于 VQ 的动作生成脚步滑动修正

**MotionFix** 是一个轻量级后处理神经网络，用于修正基于 VQ 的文本到动作生成模型（MoMask、T2M-GPT）中的**脚步滑动伪影**。它作为即插即用模块工作——无需重新训练底层的生成模型。

## 概述

基于 VQ-VAE 的动作生成模型推理速度比扩散模型快 **15–20 倍**，但在脚部接触地面时会出现滑动现象。MotionFix 通过一个 Transformer 修正网络来解决这个问题。

### 核心特性

- **🔥 选择性修正** — 仅修改滑动帧上的脚部关节；其他身体部位保持不变
- **⚡ 轻量化** — 1910 万参数，GPU 上每序列约 100ms
- **🔌 即插即用** — 适用于任何 VQ 动作生成器；无需重新训练
- **🧠 软门控 (V9)** — 可学习门控网络 + 启发式线索（高度、速度）精准定位伪影
- **🦿 运动学 IK (V9)** — 腿部骨骼链逆运动学确保膝盖位置与修正后的脚踝一致
- **📐 骨骼长度保持** — 辅助损失函数维持解剖学一致性

### 滑动率对比

| 模型 | 类型 | 平均滑动率 |
|------|------|-----------|
| MDM | 扩散模型 | 最低（基准） |
| MoMask（无 IK） | VQ | ~13.6% |
| MoMask（带 IK） | VQ | ~12.7% |
| **MoMask + MotionFix** | **VQ + 后处理** | **显著降低** |

## 项目结构

```
motionfix/
├── motionfix_model.py       # V8：选择性脚部替换（基线版本）
├── motionfix_model_v9.py    # V9：软门控 + 时间平滑 + 运动学 IK
├── motionfix_v6.py          # V6：早期迭代
├── train.py / train_v9.py   # 训练脚本
├── dataset.py               # PyTorch 数据集类
├── prepare_data.py          # 从 HumanML3D 生成训练数据对
├── test.py                  # V8 评估脚本
├── test_mdm.py / test_momask.py / test_t2mgpt.py  # 多模型基准测试
├── diagnose.py              # 诊断调试工具
├── visualize_comparison.py  # 动作可视化
├── autosync.py              # 自动 commit + push 监听脚本
├── sync.sh                  # autosync 启动脚本
│
├── motionfix_v10/           # V10 实验（最新迭代）
├── motionfix_v11/           # V11 实验（进行中）
│
├── fixed_outputs*/          # 修正后的动作输出
├── momask_results/          # MoMask 原始输出
├── t2mgpt_raw_joints/       # T2M-GPT 原始输出
└── mdm_raw_joints/          # MDM 原始输出
```

## 快速开始

### 环境依赖

```bash
pip install torch numpy scipy
```

### 修正一个动作

```python
import torch
import numpy as np
from motionfix_model_v9 import MotionFixNetworkV9

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MotionFixNetworkV9(blend_alpha=0.5, temperature=0.01).to(device)
ckpt = torch.load("checkpoints_v9/best.pth", map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

motion = np.load("your_motion.npy")  # (T, 22, 3)
T = motion.shape[0]
motion_tensor = torch.FloatTensor(
    motion.reshape(T, -1)
).unsqueeze(0).to(device)

with torch.no_grad():
    fixed = model(motion_tensor, foot_only=True)

fixed_motion = fixed.squeeze(0).cpu().numpy().reshape(T, 22, 3)
np.save("your_motion_fixed.npy", fixed_motion)
```

### 从头训练

```bash
python prepare_data_v2.py   # 准备训练数据
python train_v9.py           # 训练模型
python test_v9.py            # 评估效果
```

## 模型版本演进

| 版本 | 关键创新 | 状态 |
|------|---------|------|
| V3 | 基础 Transformer，L1 + 速度 + 脚部损失 | 基线 |
| V5 | 增加骨骼长度保持损失 | 改进 |
| **V8** | **选择性脚部替换（全量训练，局部推理）** | **稳定** |
| **V9** | **软门控 + 时间平滑 + 运动学 IK** | **当前最优** |
| V10 | 进一步实验优化 | 进行中 |
| V11 | 迭代修正（3轮） | 进行中 |

## 训练数据

训练数据对从 [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 数据集生成，使用三种扰动方式：

1. **时序平滑** — 对关节轨迹应用高斯滤波（窗口 3–7）
2. **Y 轴偏移** — 随机垂直平移（±5cm）
3. **空间噪声** — 高斯噪声（σ = 0.005–0.02）

## 引用

本项目是杜伦大学（Durham University）正在进行的研究的一部分：

```bibtex
@misc{wan2026motionfix,
  title={MotionFix: Foot Skating Correction for VQ-Based Motion Generation},
  author={Wan, Xin},
  year={2026},
  note={Durham University}
}
```
