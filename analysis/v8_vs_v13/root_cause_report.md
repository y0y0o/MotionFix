# 训练损失下降但实际效果退化的根因分析

**问题:** 为什么 V8 训练 loss 收敛到 0.0114，FSR 指标也下降，但实际效果却是脚步闪现、Jitter 暴增 24 倍？

**结论: 训练数据与推理数据的坐标系不匹配（Coordinate System Mismatch）。**

---

## 1. 现象：完美的训练 vs 崩溃的推理

对 V8 模型分别在训练数据和 MoMask 测试数据上运行前向传播：

| 测试数据 | Full Body L1 | 脚步误差 | >1m 误差帧占比 | 模型行为 |
|---------|-------------|---------|---------------|---------|
| **训练数据 (V2)** | **0.047 m** | **0.010 m** | **0%** | ✅ 完美去噪 |
| **MoMask (p000021)** | **9.859 m** | **2.016 m** | **75-79%** | ❌ 完全崩溃 |

同样的模型，训练数据上脚步误差仅 1cm，MoMask 上脚步误差 2m——差距 **200 倍**。

### 训练数据上的表现（5个样本）

```
distorted_000000: Full body L1=0.047m, Foot error=0.010m  ← 近乎完美
distorted_000001: Full body L1=0.047m, Foot error=0.010m
distorted_000002: Full body L1=0.048m, Foot error=0.010m
distorted_000003: Full body L1=0.094m, Foot error=0.015m  ← 短序列，略差
distorted_000004: Full body L1=0.094m, Foot error=0.015m
```

**模型在训练数据上的表现是优秀的。** V8 确实学会了去除 V2 格式的噪声。

### MoMask 数据上的表现（5个样本）

```
p000021 (walking):     Full body L1=9.86m, Foot error=2.02m  ← 崩溃
p000818 (spinning):    Full body L1=3.29m, Foot error=0.50m  ← 也很差
p001120 (walking fwd): Full body L1=3.29m, Foot error=0.65m  ← 仍然差
p001168 (casual walk): Full body L1=3.54m, Foot error=0.66m
p001448 (turning):     Full body L1=2.48m, Foot error=0.53m
```

**所有 MoMask 样本都出现了不同程度的退化。** p000021 最严重（9.86m），因为它的角色移动范围最大（X=[-3.61, 0.26]）。

---

## 2. 根因：坐标系不匹配

### 2.1 训练数据的坐标空间

V2 训练数据（`data/training/v2/target_000000.npy`）的坐标范围：

```
X: [-0.34,  0.32]  ← 仅 0.66m 跨度
Y: [ 0.05,  1.68]  ← 1.63m (身高范围)
Z: [-0.06,  1.11]  ← 1.17m
```

这是 **root-relative（根关节相对）坐标**——角色被固定在原点附近，只有局部的肢体运动。

### 2.2 MoMask 输出的坐标空间

```
X: [-3.61,  0.26]  ← 3.87m 跨度 (角色在行走！)
Y: [ 0.02,  1.57]  ← 1.55m
Z: [-0.48,  2.07]  ← 2.55m
```

这是 **world coordinate（世界坐标）**——角色在物理空间中移动，X 轴跨了近 4 米。

### 2.3 模型学到了什么

运行模型后，输出的坐标范围：

| | 输入范围 X | 输出范围 X |
|---|----------|----------|
| 训练数据 | [-0.34, 0.32] | **[-0.31, 0.30]** ← 匹配 |
| MoMask 数据 | [-3.61, 0.26] | **[-0.35, 0.28]** ← 被压缩到训练范围！ |

**模型学到了：不管输入坐标范围多大，输出始终在训练数据的坐标范围内。**

模型在训练时只见过 X ∈ [-0.34, 0.32] 范围内的坐标。对于 MoMask 输入落在 X=-3.6 的帧（训练时从未见过），模型仍然输出 X≈0 附近的值——因为这是它在训练中学到的"安全输出"。这是神经网络的已知行为：**对分布外（Out-of-Distribution, OOD）输入的响应是回退到训练分布的均值。**

### 2.4 为什么 Loss 还能降低？

V8 Loss 函数：
```
L = L1(pred, target)           # 全身体
  + 0.5 × L1(vel_pred, vel_target)  # 速度
  + 2.0 × L1(foot_pred, foot_target)  # 脚步
  + 2.0 × L1(foot_vel_pred, foot_vel_target)  # 脚步速度
```

在**训练数据**上：
- 输入是 root-relative 坐标 → 模型输出也在 root-relative 范围
- L1 误差仅 ~5cm（噪声被正确去除）
- Loss 从 ~0.5 降到 0.011
- ✅ 模型确实学会了去噪

问题不在于 Loss 函数本身，而在于 **Loss 是在训练分布上计算的，无法反映分布外泛化能力。**

---

## 3. 为什么 selective_replace 不能拯救这个问题？

### 3.1 修正机制回顾

```python
if foot_height < ground + 5cm AND foot_velocity > 0.03:
    output = 0.5 × original + 0.5 × model_output
    # model_output 被压回原点附近 → output 也被拉向原点
```

### 3.2 为什么 FSR 仍然下降？

FSR 下降不是因为模型学会了修正脚滑，而是因为 `_selective_replace` 的**副作用**：

1. 检测到 skating → 应用 blend
2. Blend 把脚拉向原点 → 水平方向被"锚定"
3. 锚定减少了水平速度 → skating 检测的 vel > 0.03 条件更难满足
4. FSR 计算公式: `skating_frames / contact_frames`
   - 分子 (skating) 减少：速度被 blend 降低
   - 分母 (contact) 也减少：脚被拉离地面，接触帧减少
   - 但比例改善 → FSR 下降

**FSR 下降是 blend 的数学副作用，不是模型智能修正的结果。**

### 3.3 为什么产生闪烁？

```
修正组 (frames 70-75):    脚被拉向原点 ~1.5m
未修正组 (frames 76-102):  脚回到 MoMask 原始位置
修正组 (frames 103-108):   脚再次被拉向原点 ~1.5m
```

修正组边界处帧间位移 ≈ 1.5m（原始位置 ↔ 被拉回的位置），产生肉眼可见的"闪现"。

---

## 4. V8 和 V13 的比较：为什么几乎一样？

| | V8 | V13 | 差异原因 |
|---|-----|-----|---------|
| 训练噪声 | 1x | 4.3x | — |
| 训练数据坐标空间 | root-relative | root-relative | **相同** |
| 推理数据坐标空间 | world | world | **相同** |
| 模型架构 | Transformer Encoder ×6 | Transformer Encoder ×6 | **相同** |
| selective_replace | blend_alpha=0.5 | blend_alpha=0.5 | **相同** |

**两个模型面对的是完全相同的坐标系不匹配问题。** 噪声倍数不同只影响训练数据上的去噪难度，不影响推理时的分布外行为——无论噪声 1x 还是 4.3x，模型都只学会在 root-relative 空间内输出，遇到 world-coordinate 的 MoMask 输入时都会退化。

V13 的噪声更大导致 full_output 稍微更接近原点（误差高 ~9%），但这是次要因素。**核心问题是共通的。**

---

## 5. 因果链总结

```
HumanML3D 数据集使用 root-relative 坐标存储
        │
        ▼
V2 训练数据生成时保留 root-relative 格式
  (角色固定在原点，X 范围 [-0.34, 0.32])
        │
        ▼
模型只学习在 root-relative 坐标空间内去噪
  (输出范围 X ∈ [-0.35, 0.30])
        │
        ▼
MoMask 推理输出是 world-coordinate
  (角色行走，X 范围 [-3.61, 0.26])
        │
        ▼
模型从未见过 X=-3.6 这样的坐标 → OOD 输入
  → 回退到训练分布均值 → 输出 X≈0
        │
        ▼
full_output 与输入的 L1 误差达 2-10m
        │
        ▼
_selective_replace 在 skating 帧 blend α=0.5
  → 脚被拉向原点 ~1.5m
        │
        ▼
修正组 (16% 帧被修正) 与未修正组之间的边界
  → 帧间跳跃 ~1.5m
        │
        ▼
脚步闪现 + Jitter 暴增 24x
```

---

## 6. 解决方案

### 方案 A: 统一坐标空间（根本解决）

**在推理前将 MoMask 输入转换到 root-relative 空间，推理后再转回 world space。**

```python
# 推理前
pelvis_pos = input[:, 0, :]  # joint 0 = pelvis
input_root_relative = input - pelvis_pos[:, None, :]

# 模型推理
output_root_relative = model(input_root_relative)

# 推理后转回 world
output_world = output_root_relative + pelvis_pos[:, None, :]
```

优点：直接解决坐标不匹配  
缺点：需要修改推理流程

### 方案 B: 用 world-coordinate 数据重新训练

在生成训练数据时不使用 root-relative 格式，保留原始的 world coordinates。让模型见识完整的坐标范围。

优点：模型直接适配 world space  
缺点：需要重新生成训练数据和重新训练

### 方案 C: 在训练时做坐标归一化 + 推理时同样归一化

```python
# 训练时：对输入做 per-sample 归一化
input_mean = input.mean()  # 或只对 pelvis 做
input_std = input.std()
input_normalized = (input - input_mean) / input_std

# 推理时：同样归一化
input_normalized = (input - input_mean) / input_std
output_normalized = model(input_normalized)
output = output_normalized * input_std + input_mean
```

### 方案 D: 在 Loss 中加入 OOD 惩罚

在 Loss 中加入对大幅偏离输入的惩罚：
```python
L_anchor = ||pred - original_input||  # 惩罚偏离输入太远
```
但这只是缓解，不能根本解决坐标空间不匹配。

---

## 7. 为什么之前的评估没有发现这个问题？

1. **FSR 是一个有误导性的指标**——它测量的是 skating 帧比例，但不关心修正幅度是否合理。即使修正把脚拉到完全错误的位置，只要水平速度降低了，FSR 就下降。

2. **没有评估模型 raw output (full_output) 的质量**——之前的测试只检查了 selective_replace 后的 fixed output，没有直接对比 full_output 与原始输入。

3. **没有检查训练数据与推理数据的坐标空间差异**——假设训练和推理在相同的坐标空间，但实际上 V2 数据是 root-relative，MoMask 是 world coordinate。

4. **Loss 收敛被误认为模型学习成功**——Loss 只在训练分布上衡量性能，不能反映分布外泛化。

---

## 8. 关键指标

| 指标 | 训练数据 (V2) | 推理数据 (MoMask p000021) | 退化倍数 |
|------|-------------|--------------------------|---------|
| 输入 X 范围 | [-0.34, 0.32] | [-3.61, 0.26] | 5.9x |
| 输出 X 范围 | [-0.31, 0.30] | [-0.35, 0.28] | — |
| 脚步 L1 误差 | 0.010 m | 2.016 m | **200x** |
| >1m 误差帧占比 | 0% | 75-79% | — |
| Per-joint 变化量 | 0.011 m | 2.094 m | **190x** |

模型在训练数据上的输出范围与输入范围相匹配（[-0.31, 0.30] ≈ [-0.34, 0.32]），但在 MoMask 上输出范围被强制限制在训练范围内（[-0.35, 0.28]），完全忽略了输入的 4m 跨度。
