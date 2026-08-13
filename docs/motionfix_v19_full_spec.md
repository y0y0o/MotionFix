# MotionFix V19 —— 完整技术文档(结构 · 方法 · 全流程细节)

> 生成日期:2026-08-13。本文档所有结构、公式、超参数均**逐行核对自源码**:
> `models/v19.py`、`models/v18.py`、`models/v18_ik.py`、`data/prep/v19.py`、
> `training/v19.py`、`testing/v19_eval.py`。凡涉及数字的地方标注了来源。
> 目标:任何人拿这一份就能从零复现 V19,不需要再翻别的文件。

---

## 目录

- [0. 总览:V19 是什么](#0-总览v19-是什么)
- [1. 数据表示与坐标系](#1-数据表示与坐标系)
- [2. 数据来源与语料构建](#2-数据来源与语料构建)
- [3. 接触检测与去滑(物理,第一段)](#3-接触检测与去滑物理第一段)
- [4. 学习平滑器(神经网络,第二段)](#4-学习平滑器神经网络第二段)
- [5. 2骨逆运动学(物理,第三段)](#5-2骨逆运动学物理第三段)
- [6. 损失函数](#6-损失函数)
- [7. 训练流程](#7-训练流程)
- [8. 推理流程](#8-推理流程)
- [9. 评估协议与 7 指标](#9-评估协议与-7-指标)
- [10. 关键设计决策(为什么这样做)](#10-关键设计决策为什么这样做)
- [11. 从零复现命令](#11-从零复现命令)

---

## 0. 总览:V19 是什么

MotionFix 是一个**生成器无关、无需重训生成器的后处理管线**,专门去除文本→动作生成器
(MoMask / T2M-GPT / MDM)输出里的**脚滑(foot-skating)**,同时保持骨长刚性、脚不穿地。

**三段式管线**(输入输出都是 `(T, 22, 3)` 世界坐标关节):

```
 原始动作            第一段:去滑            第二段:学习平滑         第三段:2骨IK          输出
(T,22,3)  ──────►  plant-at-mean XZ ──►  46.3K 1D-CNN,只改踝 ──►  髋→膝→踝+刚性脚趾 ──►  (T,22,3)
                    纯物理,零参数          IK 在训练回路里          纯物理,零参数
                    (无漂移,低FSR)        (补抖动)               (骨长刚性,接触保证)
```

**核心事实**:三段里**只有中间那段是神经网络**。第一段和第三段是纯物理、无参数。
V19 相对 V18 **没有改变管线形状**,只修了平滑器的**训练方式**(见 §10)。

- 当前交付点:`checkpoints/v19_088a10`(`--jit-share 0.88 --anch-share 0.10`)。
- 平滑器参数量:**46,276 ≈ 46.3K**(见 §4.2 逐层计算)。

---

## 1. 数据表示与坐标系

### 1.1 骨架:HumanML3D 22 关节

所有模型都在 **22 关节、3D 世界坐标**上操作。一条动作 `(T, 22, 3)`,展平成 `(T, 66)` 作为输入。

关节索引(`models/v18.py:30-35`、`models/v18_ik.py:28-34`):

| 索引 | 名称 | 索引 | 名称 |
|---|---|---|---|
| 0 | Pelvis 骨盆 | 7 | **Left Ankle 左踝** |
| 1 | Left Hip 左髋 | 8 | **Right Ankle 右踝** |
| 2 | Right Hip 右髋 | 9 | Spine2 |
| 3 | Spine1 | 10 | **Left Foot 左脚(趾)** |
| 4 | Left Knee 左膝 | 11 | **Right Foot 右脚(趾)** |
| 5 | Right Knee 右膝 | 12 | Neck |
| 6 | Spine3 | 13–21 | 肩/臂/头 |

**腿链**(`LEGS`,两处定义完全一致):
- 左腿:hip 1 → knee 4 → ankle 7 → toe 10
- 右腿:hip 2 → knee 5 → ankle 8 → toe 11

脚关节:**踝 7、8;趾 10、11**。
- **FSR(脚滑)只定义在踝(7,8)上**——这决定了 V19 只需踝可微(见 §5.2)。
- 去滑对全部 4 个脚关节的 XZ(8 维)操作;学习平滑器只对 2 个踝的 XZ(4 维)操作。

维度索引(`models/v18.py:31-34`):
- `FOOT_XZ_DIMS = [21,23, 24,26, 30,32, 33,35]` = 4 脚关节 × (X,Z),共 8 维。顺序 `[7x,7z, 8x,8z, 10x,10z, 11x,11z]`,**踝在前 4 维**。
- `FOOT_Y_DIMS = [22,25,31,34]` = 4 脚关节的 Y(高度),共 4 维。

### 1.2 从 263 维特征到世界坐标

原始 HumanML3D 文件是 **263 维特征向量**(`new_joint_vecs`),不是关节坐标。
转换靠 T2M-GPT 的 `recover_from_ric`:

```
263维特征 (T,263) ──recover_from_ric──► 世界坐标关节 (T,22,3)
```

代码位置:`/home3/nxkh91/projects/T2M-GPT/utils/motion_process.py::recover_from_ric`
(`data/prep/v19.py:52-55` 通过临时改 `sys.path` 导入,避开 T2M-GPT 自带的 `utils` 包冲突)。

---

## 2. 数据来源与语料构建

脚本:`data/prep/v19.py`。输出:`data/training/v19_cache.pt`。

### 2.1 数据源

```
SRC = /home3/nxkh91/projects/HumanML3D/HumanML3D/new_joint_vecs   （原始 263 维特征）
```

**关键:读取原始 `new_joint_vecs`,不做任何反归一化。**

> ⚠️ **v14 反归一化 bug(2026-07-20 发现)**:旧脚本 `data/prep/v14.py::convert_to_joints`
> 对已经是原始特征的 `new_joint_vecs` 又做了 `motion = motion_263 * std + mean`,把数据搞坏。
> 实测 146 条动作:脚 Y 动态范围从正确的 0.269m 压成 0.090m(9cm 带),
> 接触启发式把 ~97% 帧误判为接触,`deskated_target` 把整条走路轨迹钉到一个均值点
> (偏离原始 0.65m,而 MoMask 参考只有 0.042m)。**V14/V15 都在这份坏数据上训练**,
> 所以那两个负结果是被数据污染混淆的,不能当作"学习不行"的干净证据。V19 新建 prep 修掉。

### 2.2 语料过滤

只保留"接触/腾空混合合理"的动作(`data/prep/v19.py:63-66, 94-118`):

| 过滤条件 | 阈值 | 理由 |
|---|---|---|
| 最短长度 | ≥ 40 帧 | 太短没意义 |
| 最长截断 | 196 帧(`MAX_LEN`) | 统一长度 |
| **接触占比** | `[0.20, 0.90]`(`MIN_CF, MAX_CF`) | 落在这个带外,接触启发式(地面=5分位)不可信,去滑参考会是垃圾 |
| **去滑偏离** | ≤ 0.30m(`MAX_DESKATE_DEV`) | 偏离过大说明去滑退化,剔除 |

最终保留 **~3.4k 条 HumanML3D 动作**(`--n 4000` 上限)。

### 2.3 缓存内容

`v19_cache.pt` 里预先算好所有**不依赖模型**的量(`data/prep/v19.py:84, 126-138`),
每条 padding 到 196 帧(edge 模式):

| key | 形状 | 含义 |
|---|---|---|
| `des4` | (N,196,4) | 去滑后的**踝** XZ(平滑器要拟合/超越的参考) |
| `orig4` | (N,196,4) | 原始**踝** XZ |
| `w4` | (N,196,4) | 接触权重(踝,每 XZ 维复制) |
| `hip` | (N,196,2,3) | 两腿髋的世界坐标(IK 用) |
| `L1` | (N,196,2) | 大腿长 `|膝-髋|`,逐帧取自原始 |
| `L2` | (N,196,2) | 小腿长 `|踝-膝|`,逐帧取自原始 |
| `ankY` | (N,196,2) | 踝的 Y(高度,IK 保持不变) |
| `toeoff` | (N,196,2,3) | 趾相对踝的向量 `趾-踝`(IK 刚性跟随) |
| `mask` | (N,196) | 1=真实帧,0=padding |

> **无泄漏保证**:训练用 4k HumanML3D,而评估用 MoMask/MDM/T2M-GPT →
> 三个生成器**全部 out-of-distribution**,结构上不可能泄漏(`training/v19.py` B7)。

---

## 3. 接触检测与去滑(物理,第一段)

### 3.1 软接触权重

`models/v18.py:38-48::compute_contact_weight_np`。对每个脚关节按高度算一个 [0,1] 的软权重,
接触时 ≈1:

```
ground_j = percentile(foot_y[:, j], 5)          # 地面 = 该关节高度的第 5 分位
w[:, j]  = sigmoid( -(foot_y[:, j] - (ground_j + h_thresh)) / temp )
```

超参(`models/v19.py:85-87`,**必须与 utils.metrics 完全一致**):
- `H_THRESH = 0.05`(离地 5cm 内算接触)
- `TEMP = 0.02`(sigmoid 温度)

得到 `(T,4)` 权重,再 `np.repeat(..., 2)` 复制成 `(T,8)`(每个 XZ 维一份)。

### 3.2 去滑:plant-at-segment-mean

`models/v18.py:51-80::deskated_target`。对每个脚 XZ 通道逐段处理:

- **接触段**(`w > 0.5`):把整段位置**钉到该段的均值**(`anchor = pos[t:e].mean()`)
  → 脚不滑、不漂移。用均值(而非段首)是为了**最小化对原始的偏离**。
- **腾空段**:保留原始位置 → 不动摆动形状。

**为什么用 plant-at-mean 而不是速度积分**:早期 V18 用"速度 mask + cumsum 积分"
的路径会累积漂移(FootErr 达 0.39m)。plant-at-mean **没有积分,天然无漂移**,
且钉的目标是可达的(偏离原始只几 cm,不是 40cm 的积分漂移)。

> 注意:`models/v18.py` 里同时存在**已废弃的速度空间 cumsum 路径**(`refine`/`v18_fix`)
> 和**在用的位置空间路径**(`deskate_xz`/`smooth_fix`)。V19 只用后者。别混。

去滑入口 `models/v18.py:189-197::deskate_xz` 返回 `(tgt8, w_xz8)`,踝在前 4 维。

---

## 4. 学习平滑器(神经网络,第二段)

### 4.1 职责

去滑之后,接触段被钉成常数、接触↔腾空边界是硬拐点 → **抖动(Jitter)偏高**。
平滑器唯一的活:**在不重新引入脚滑的前提下,磨平这些拐点**。它预测的是**踝 XZ 的位置残差**,
叠加在去滑参考上:`out = deskated + residual`。

它比全局高斯**唯一的结构性优势**:它能看到接触权重 `w`,所以能按接触相位自适应
(接触时别动、腾空时使劲平滑)。全局高斯做不到——这正是损失函数要榨出来的东西(§6)。

### 4.2 网络结构

`models/v19.py:132-160::V19Smoother`。一个**轻量 1D 时序 CNN**:

```
输入 (B, T, 12): [去滑踝XZ(4) | 原始踝XZ(4) | 接触权重w(4)]
   （同时给去滑和原始,让它看到"滑被去掉了多少",即有多少平滑余量）
      │  permute → (B, 12, T)
      ▼
  Conv1d(12 → 64, kernel=5, pad=2) → ReLU
  Conv1d(64 → 64, kernel=5, pad=2) → ReLU
  Conv1d(64 → 64, kernel=5, pad=2) → ReLU
  Conv1d(64 →  4, kernel=5, pad=2)          ← 末层零初始化
      │  permute → (B, T, 4)
      ▼
输出 (B, T, 4): 踝 XZ 位置残差
```

超参:`hidden=64, kernel=5, n_layers=4, in_dim=12, out_dim=4`。

**逐层参数量**(kernel=5):
- Conv1: 12×64×5 + 64 = 3,904
- Conv2: 64×64×5 + 64 = 20,544
- Conv3: 64×64×5 + 64 = 20,544
- Conv4: 64×4×5 + 4 = 1,284
- **合计 = 46,276 ≈ 46.3K**

**两个关键设计**:
1. **末层零初始化**(`nn.init.zeros_`,`models/v19.py:152-153`):初始残差恒为 0
   → 模型在初始时**精确等于去滑基线**,是个安全起点。
2. **只输出 4 维(踝 XZ),不是 8 维**:V18 输出 8 维,但 IK 会把趾刚性重建、
   覆盖掉一半输出 → V18 一半容量和一半梯度浪费在到不了指标的输出上。V19 只输出踝。

### 4.3 关键:IK 在训练回路里(A1)

V18 在**平滑器的原始输出**上训练,却在**过 IK 之后**评估。实测这个错配的代价(10 条 held-out MoMask):
```
FSR    12.0% ──IK──► 12.9%   (+0.8pp)
Jitter 0.00970 ──IK──► 0.01143  (×1.18)
```
平滑器输给调好的高斯 1.8pp FSR,其中约一半的差距是**它从没被训练穿过的那一级**引入的。
V19 把 IK 的踝钳制做成可微(`torch_ankle_ik`)放进前向,**损失算在真正被评估的轨迹上**。

---

## 5. 2骨逆运动学(物理,第三段)

去滑/平滑只改脚 XZ,会**拉断小腿**(旋转类动作里小腿可被拉到 2m)、**翻转脚**(趾被独立移动)。
IK 段在保持平滑后踝目标的同时**修复骨架**。

### 5.1 numpy 版(推理用,`models/v18_ik.py`)

`two_bone_ik`(`:42-109`)对每条腿 `髋→膝→踝(+刚性趾)`:

1. 取平滑后的踝 XZ 作 IK 目标(**Y 保持原始**,去滑/平滑只动 XZ)。
2. **钳制到腿的可达范围**(`|髋-踝| ≤ 大腿+小腿`)——这是**阻止无界拉伸/漂移**的关键。
   - 先钳水平 reach(保持脚高 Y 不变),再把 `髋→踝` 距离钳进 `[|L1-L2|, L1+L2]`。
3. 用**余弦定理**解膝,弯曲平面由**原始膝**确定(膝按自然方向弯);原始膝共线时用 fallback 方向。
4. **趾刚性跟随踝**:`趾_new = 踝_new + (趾-踝)_原始` → 骨长+朝向保持,不翻转。

**骨长逐帧取自原始动作**(`L1=|膝-髋|`,`L2=|踝-膝|`),所以修正后骨架精确复现源骨长
(BoneCV ≈ 原始)。骨盆/髋/脊柱/臂/头**完全不动**,只改膝/踝/趾。

`apply_ik`(`:112-148`)是整段的入口:输入(原始动作,平滑后动作)→ 输出 IK 修复后动作。

### 5.2 可微版(训练用,`models/v19.py:95-129::torch_ankle_ik`)

**只有 IK 的踝钳制路径需要可微**,因为:
- FSR 只定义在踝上;
- `apply_ik` 让趾**刚性**跟随踝(趾不碰任何指标);
- **余弦定理解膝不碰任何指标**,所以从训练图里省掉(推理时照常由 numpy `apply_ik` 解)。

`torch_ankle_ik` **逐行镜像** numpy 版的踝路径(水平 reach 钳制 → 半径钳进 `[|L1-L2|, L1+L2]`)。
正确性由 `models/v19.py:262-291` 的自检保证:
- **与 numpy IK 的踝最大绝对误差 < 1e-6**(实测 2.2e-16 量级);
- 梯度有限(用 `_safe_norm`,`sqrt(clamp(min=1e-6))`,避免在 0 处 NaN)。

---

## 6. 损失函数

`models/v19.py:163-220::V19Loss`。**全部算在过 IK 之后的踝轨迹 + 刚性趾上**,即 `utils.metrics` 真正会看到的量。**按接触相位分区**,这是单一全局高斯**可证明做不到**的行为。

四项:

```
L = λ_air   · jit_air              腾空使劲平滑
  + λ_all   · jit_all              保持边界连续
  + λ_skate · skate                接触帧过阈值的软计数(阈值对齐)
  + λ_anch  · anch                 锚定去滑参考(抗漂移/抗作弊)
```

各项定义(`models/v19.py:183-214`,`_wmean` = 掩码归一化加权均值):

| 项 | 公式 | 说明 |
|---|---|---|
| `jit_all` | `wmean(acc(踝)², m) + wmean(acc(趾)², m)` | 二阶差分 acc = `x[2:]-2x[1:-1]+x[:-2]`;全帧,保边界连续 |
| `jit_air` | `wmean(acc(踝)², m·(1-w)) + wmean(acc(趾)², m·(1-w))` | 用 `(1-w)` 加权 → **只在腾空使劲平滑** |
| `skate` | `wmean( sigmoid((speed-0.03)/τ), m·w³ )` | **阈值对齐的软计数**;`speed=|踝相邻帧位移|`;`w³` 加权到接触帧;**只算踝** |
| `anch` | `wmean(|踝 - 去滑参考|, m)` | 抗漂移、抗代理作弊 |

超参:`τ = 0.008`(skate 软计数温度);`SKATE_THRESH = 0.03`(**必须与 FSR 阈值一致**)。

**为什么 skate 项要"阈值对齐的软计数"而不是均值(A3)**:
- V18 的反脚滑项是 `mean(|v|·w³)`——一个**速度均值**;
- 而 FSR 是 `接触帧里 |v|>0.03 的比例`——一个**过阈值的计数**。
- 大多数去滑后的接触帧远低于 0.03,它们主导了均值却对 FSR 毫无贡献 →
  模型把预算花在**本来就已通过**的帧上。V19 改成 `sigmoid((speed-0.03)/τ)`,**直接对齐 FSR 的判定边界**。

> ⚠️ **V16 教训**:V16 试过可微 soft-FSR 惨败(FSR 35.8%,模型把脚甩到 12-25cm 外钻代理空子)。
> V19 之所以安全,是因为 V16 的逃逸路线现在被物理堵死了:
> (1) 残差叠加在**去滑参考**上(不是自由轨迹);(2) `torch_ankle_ik` 在损失内把踝钳到腿的可达范围。
> **`anch` 项 + 每 epoch 记录的 FootErr 是"绊线"**——FootErr 一涨,就是代理又被钻了(`training/v19.py:94-99`)。

---

## 7. 训练流程

脚本:`training/v19.py`。

### 7.1 超参

| 项 | 值 | 备注 |
|---|---|---|
| epochs | 300(默认;实跑常用 120) | |
| batch | 16 | |
| **LR** | **2e-4** | ⚠️ 2e-3 会让模型坍回残差=0(Adam 首步过冲、梯度死掉),见 devlog §6 |
| scheduler | CosineAnnealingLR | |
| grad clip | 1.0 | |
| val 比例 | 0.12 | 按固定 seed 随机切分 |
| seed | 0 | |

### 7.2 前向管线(`training/v19.py:73-91::forward_pipeline`)

```
res = model(des4, orig4, w4)             平滑器输出踝残差
ank_xz = des4 + res                       叠加到去滑参考 → PRE-IK 踝 XZ
对每条腿:
    tgt = [ank_x, ankY_原始, ank_z]        Y 用原始
    a = torch_ankle_ik(hip, tgt, L1, L2)   可微踝钳制 → POST-IK 踝
    toe = a + toeoff                       刚性趾跟随
返回 (POST-IK 踝 XZ, 刚性趾 XZ)            ← 损失就算在这两个上
```

### 7.3 λ 自动标定(B5,`training/v19.py:102-128::calibrate_lambdas`)

四项的原始量纲差 ~3 个数量级(acc²~1e-6 vs 软计数~1e-1),手调权重没有意义。
V19 在**第 0 步**按**每项对模型参数的梯度范数**标定 λ,使每项占总梯度范数的**目标份额**:

```
GRAD_TARGET = {jit_air: 0.30, jit_all: 0.15, skate: 0.35, anch: 0.20}   （默认)
λ_k = target_k / grad_norm_k，再整体归一化到 O(1)
```

- `--jit-share`:给两个 jitter 项的总梯度份额(默认 0.45,按 2:1 分给 air/all)。
  **这是选择 FSR-Jitter 前沿上操作点的旋钮——要扫,不要只调一次。**
- `--anch-share`:锚点份额(默认 0.20);调低才能到前沿的低抖动端。
- skate 份额 = `1 - anch - jit`(自动补足)。
- 交付点 `v19_088a10` = `--jit-share 0.88 --anch-share 0.10`。

### 7.4 选点与保存(B3)

- 每 epoch 在**验证集**上算 val loss(V18 是按训练 loss、且根本没验证集);
- val loss 最优时存 `best.pth`,同时每 epoch 存 `latest.pth`;
- 每 epoch 记录 `FootErr`(抗作弊绊线);
- 训练历史 + λ + 梯度范数写 `history.json`。

### 7.5 掩码归一化(B6)

所有损失项都按**真实序列长度**归一化(`_wmean` 的分母是掩码和)。
V18 edge-pad 到 196 帧,常数尾部对分子贡献 0 但被算进分母 → 每条动作有效学习率不同。V19 修掉。

---

## 8. 推理流程

`models/v19.py:227-259`。

```python
# 单条动作
smooth_fix_v19(motion, model, device)   # 去滑 → V19平滑器 → 返回 PRE-IK (T,22,3)
pipeline_fix_v19(motion, model, device) # 完整:去滑 → V19平滑器 → 2骨IK → 返回 (T,22,3)
```

`smooth_fix_v19`(`:227-253`)细节:
1. `deskate_xz(motion)` → 去滑踝 `des4` + 权重 `w4`,原始踝 `orig4`;
2. `model(des4, orig4, w4)` → 残差(`torch.no_grad`);
3. `out4 = des4 + res`,写回**踝** XZ(趾不写,留给 `apply_ik` 重建);
4. `pipeline_fix_v19` 再调 `apply_ik(原始, 平滑结果)` 走 numpy IK(**含余弦解膝**)。

**注意**:训练用可微 `torch_ankle_ik`(只钳踝),推理用 numpy `apply_ik`(钳踝 + 余弦解膝 + 刚性趾)。
两者的踝路径数值一致(2.2e-16),膝只在推理解——不影响任何指标。

---

## 9. 评估协议与 7 指标

脚本:`testing/v19_eval.py`。输出:`analysis/v19/<tag>_results.json`。

### 9.1 无泄漏切分(B1)

旧脚本 `v18ik_scale.py` 在全部 50 条 MoMask 上评估,但平滑器训练用了其中 40 条 → "同分布"列 80% 是训练数据。V19 显式拆开:

| 切分 | n | 说明 |
|---|---|---|
| `momask_heldout` | 10 | **唯一诚实的同分布数字** |
| `momask_train` | 40 | 单列出来**展示泄漏差距** |
| `mdm` | 50 | 一直干净 |
| `t2mgpt` | 50 | 一直干净 |

(注:V19 本身训练在 4k HumanML3D 上,连 MoMask 的 40 条也 OOD;此切分是为了对比旧 V18 的泄漏。)

### 9.2 消融的五路(B2)

每一路都以**同一个 2骨IK 收尾**,只换中间(`testing/v19_eval.py:18-23, 138`):

| 路 | 组成 | 含义 |
|---|---|---|
| `original` | 不修 | 基线 |
| `deskate_ik` | 去滑 → IK | 纯物理,无学习 |
| `gauss_ik` | 去滑 → 高斯σ=1.5 → IK | 非学习平滑器(`GAUSS_SIGMA=1.5`) |
| `learn_ik` | 去滑 → V18平滑 → IK | 旧交付 |
| `v19_ik` | 去滑 → V19平滑 → IK | 当前方法(`--v19 <ckpt>`) |

### 9.3 七指标(`utils/metrics.py`,`testing/v19_eval.py:68-78`)

| 指标 | 含义 | 会不会动 |
|---|---|---|
| **FSR** | 脚滑率:接触帧里踝水平速度 >0.03 的**比例**(只算踝) | ✅ 主指标 |
| **Jitter** | 脚加速度 RMS(二阶差分) | ✅ 主指标 |
| **FootErr** | 相对原始的踝偏离量 | ✅ 会动 |
| Floating | 悬空 | ❌ 恒定 |
| ContactAcc | 接触准确率(IK 保证 100%) | ❌ 恒定 |
| BoneCV | 骨长一致性(IK 保证=原始) | ❌ 恒定 |
| Penetration | 穿地 | ❌ 恒定 |

> 7 指标里**实际只有 3 个在动**(FSR/Jitter/FootErr),另 4 个是**构造保证的不变量**。
> 写论文时应写成 "invariant by construction",别当成绩(否则审稿人一眼看穿)。

### 9.4 两个诊断(`testing/v19_eval.py:167-187`)

- **诊断 1 泄漏差距**:`learn_ik` 在 momask_train vs held-out 的 FSR 差。
- **诊断 2 平滑是否回退 FSR**:对每个切分打印 `deskate_ik / gauss_ik / learn_ik / v19_ik` 的 FSR
  及相对 `deskate_ik` 的增量 → 直接看到**平滑器在 FSR 上做负功多少**。

### 9.5 配套分析

- `analysis/v19_frontier.py` → FSR-Jitter 前沿(学习 vs 高斯),`frontier.json/.png`
- `analysis/v19_jitter_trace.py` → 尖峰 p99/max + 逐帧曲线(RMS 会掩盖抽搐),`jitter_stats.json`
- `testing/v19_semantic.py` → FID / R-precision / MM-Dist / Diversity + 配对 bootstrap,`semantic.json`

(具体消融数字见 `docs/motionfix_current_status.md` §2 与 `docs/v19_results.md`。)

---

## 10. 关键设计决策(为什么这样做)

V19 相对 V18 **不改管线形状**,只修训练方式。逐条对照(`models/v19.py:1-69` 文件头 + `training/v19.py:4-19`):

| 编号 | V18 的问题 | V19 的修法 |
|---|---|---|
| **A1** | IK 不在训练回路(训练在 raw 输出、评估在过 IK 后) | 可微 `torch_ankle_ik` 进前向,损失算在 POST-IK 轨迹上(精度 2.2e-16) |
| **A3** | 反脚滑项是**速度均值**,与 FSR 的**过阈值计数**错配 | 改成阈值对齐 `sigmoid((speed-0.03)/τ)`,且按接触相位分区 |
| — | 输出 8 维,一半被 IK 覆盖 | **只输出踝 4 维** |
| **B3** | 按训练 loss 选点、无验证集 | **验证集选点** |
| **B5** | 权重 250:12:1 跨 3 数量级、无意义 | **按梯度范数自动标定 λ** |
| **B6** | padding 稀释梯度 | 全项**掩码归一化** |
| **B7** | 训练用 40 条 MoMask(泄漏) | **4k HumanML3D**,三生成器全 OOD |
| — | `prep/v14.py` 反归一化 bug | 新建 `prep/v19.py` 读原始特征 |
| — | LR 2e-3 导致坍缩 | 改 **2e-4** |

**为什么整体是"物理做重活、学习只补抖动"的形状**(这是消融证据支持的结构,不是主张):
- FSR 主要由**去滑**(物理)完成;
- 纯去滑抖动大,**平滑器**只负责把抖动补回去,代价是 FSR 略回升;
- IK(物理)保证骨长与接触。
- 消融显示**学习平滑器 ≈ 调好的高斯**——这是诚实结论。

---

## 11. 从零复现命令

从仓库根目录 `/home3/nxkh91/projects/motionfix` 运行,环境 `conda activate t2mgpt`:

```bash
# 1. 构建训练语料(从原始 HumanML3D,一次性)
python data/prep/v19.py --n 4000              # → data/training/v19_cache.pt

# 2. 训练一个操作点（交付点 088a10）
python training/v19.py --jit-share 0.88 --anch-share 0.10 --tag 088a10
                                               # → checkpoints/v19_088a10/best.pth

# 3. 7 指标评估（无泄漏 + 五路消融）
python testing/v19_eval.py --v19 checkpoints/v19_088a10/best.pth --tag v19_088a10
                                               # → analysis/v19/v19_088a10_results.json

# 4. 前沿扫描（学习 vs 高斯）
python analysis/v19_frontier.py                # → analysis/v19/frontier.png

# 5. 尖峰分析（p99/max）
python analysis/v19_jitter_trace.py            # → analysis/v19/jitter_trace.png

# 6. 语义保真（FID/R-prec/MM-Dist）
python testing/v19_semantic.py --v19 checkpoints/v19_088a10/best.pth

# 7. 渲染对比视频（四联画）—— 注意先把 checkpoint 改成 088a10
python utils/render_v19.py                     # → outputs/videos/v19/

# 组件自检
python models/v19.py                           # 校验可微IK vs numpy IK + 参数量
python models/v18_ik.py                        # 校验骨长保持
```

---

## 附:一句话总结

**V19 = 去滑(物理,plant-at-mean)+ 46.3K 1D-CNN 平滑器(只改踝、IK 在训练回路里、
损失阈值对齐 FSR 并按接触相位分区)+ 2骨IK(物理,可达性钳制 + 刚性趾)。**
管线形状与 V18 相同,V19 只修了平滑器的训练方式(A1/A3 + 六项工程修正),
训练在 4k HumanML3D 上,三个生成器全部 out-of-distribution。
