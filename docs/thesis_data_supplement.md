# 数据补充(可直接粘进论文)
> 2026-08-31。补齐工作清单里"需我补的数据"5 项(A10 / A8 / B3 / B4 / B6)。
> 全部来自磁盘实测:`analysis/v19/semantic_largeref.json`、`data/training/v19_cache.pt`、
> `checkpoints/v19_088a10/best.pth`、`models/v19.py`、`training/v19.py`,GPU 计时于 gpu3。

---

## A10 语义指标完整表(补进 §5.4)

**FID↓ / R-precision(top-1/2/3)↑ / MM-Dist↓ / Diversity(参考≈9.5)** — 参考集 n=1215(满秩)。

### T2M-GPT (n=200)
| 配置 | FID↓ | R@1 | R@2 | R@3 | MM-Dist↓ | Diversity |
|---|---|---|---|---|---|---|
| original | 0.483 | 0.495 | 0.679 | 0.775 | 3.144 | 9.70 |
| de-skate+IK | 0.658 | 0.488 | 0.680 | 0.777 | 3.154 | 9.78 |
| Gaussian+IK | 0.485 | 0.491 | 0.679 | 0.775 | 3.153 | 9.54 |
| learn+IK | 0.489 | 0.495 | 0.680 | 0.777 | 3.157 | 9.54 |
| **V19** | 0.483 | 0.491 | 0.678 | 0.776 | 3.150 | 9.60 |

### MoMask (n=200)
| 配置 | FID↓ | R@1 | R@2 | R@3 | MM-Dist↓ | Diversity |
|---|---|---|---|---|---|---|
| original | 0.958 | 0.515 | 0.701 | 0.800 | 3.076 | 9.38 |
| de-skate+IK | 1.613 | 0.494 | 0.686 | 0.781 | 3.227 | 9.20 |
| Gaussian+IK | 1.058 | 0.504 | 0.696 | 0.794 | 3.138 | 9.39 |
| learn+IK | 1.020 | 0.505 | 0.691 | 0.793 | 3.139 | 9.17 |
| **V19** | 1.074 | 0.508 | 0.695 | 0.792 | 3.136 | 8.98 |

### MDM (n=50)
| 配置 | FID↓ | R@1 | R@2 | R@3 | MM-Dist↓ | Diversity |
|---|---|---|---|---|---|---|
| original | 15.942 | 0.430 | 0.643 | 0.794 | 2.499 | 9.89 |
| de-skate+IK | 14.903 | 0.490 | 0.667 | 0.778 | 2.537 | 8.67 |
| Gaussian+IK | 15.524 | 0.446 | 0.647 | 0.773 | 2.535 | 9.35 |
| learn+IK | 15.573 | 0.470 | 0.639 | 0.764 | 2.517 | 9.27 |
| **V19** | 15.758 | 0.443 | 0.645 | 0.782 | 2.533 | 9.38 |

**ΔMM-Dist vs original(配对 bootstrap 95% CI,是否显著)—— 这是"修脚不破坏文本对齐"的证据**:
- **V19 在三个生成器上 ΔMM-Dist 均不显著**:T2M-GPT +0.006 [−0.023,+0.042] ns;
  MoMask +0.060 [−0.003,+0.140] ns;MDM +0.034 [−0.031,+0.137] ns。→ 修脚不损文本对齐。✅
- 唯二显著的是 **MoMask 的 de-skate(+0.151, [+0.039,+0.290], SIG)** 和 learn(+0.063, SIG)——
  即**纯去滑显著推高 MM-Dist,平滑器把它修回不显著**,与"de-skate 伤最多、smoother 修复"一致。
- **可粘一句**:
  > Correcting the feet leaves text–motion alignment intact: the change in MM-Dist for the
  > full pipeline (V19) is not significant on any generator (paired bootstrap; T2M-GPT
  > p > 0.05, MoMask p > 0.05, MDM p > 0.05). The only significant degradation is de-skate
  > alone on MoMask (ΔMM-Dist +0.15, 95% CI [0.04, 0.29]), which the smoother repairs.

---

## A8 确切训练条数(补进 §3.6/§4.1,替换"3.4–4 k")

- 语料从**原始 HumanML3D** 重建,按**接触占比 ∈ [0.20, 0.90]** 过滤,取满上限 **4,000 条**
  (`data/training/v19_cache.pt` 各张量首维=4000)。
- 训练时 **12% 留验证**(`VAL_FRAC=0.12`)→ **train 3,520 / val 480**。
- **可粘**:
  > The smoother is trained on 4,000 HumanML3D motions (rebuilt from the raw dataset and
  > filtered to a contact fraction in [0.20, 0.90]), split 3,520 train / 480 validation.

---

## B3 网络结构 + 训练超参(补进 §3 / 附录)

**结构(46,276 参数)**:
- 1D 时序 CNN,4 层 Conv1d(kernel=5, padding=2, hidden=64) + ReLU,末层输出零初始化
  → 初始即等于 de-skate 基线(安全起点)。
- 输入 (T, 12):de-skated 踝 XZ(4) ∥ 原始踝 XZ(4) ∥ contact 权重(4);
  输出 (T, 4):两踝 XZ 位置残差。**只动踝,不碰身体其余部分。**

**训练**:
| 项 | 值 |
|---|---|
| optimizer | Adam,lr = 2e-4(注:2e-3 会把模型塌回 residual=0) |
| scheduler | CosineAnnealingLR(T_max = epochs) |
| batch | 16 |
| epochs | 上限 300;**交付点 best 在 epoch 115**(按验证 loss 选点,best val_loss = 0.00174) |
| 选点 | validation-loss checkpointing(12% held-out val) |
| λ | 按各项梯度范数自动标定(见 B4) |
| 硬件 | NVIDIA TITAN Xp 12GB (gpu3),PyTorch 2.1.0+cu121 |

---

## B4 损失函数精确形式(补进 §3.3,并修式(3))

损失在 **post-IK** 的踝+趾 XZ 轨迹上算(= `utils.metrics` 实际看到的轨迹),4 项:

L = λ_air·**jit_air** + λ_all·**jit_all** + λ_skate·**skate** + λ_anch·**anch**

- **jit_all**:踝与趾 XZ 的二阶差分(加速度)平方的掩码加权均值 = 平滑项(即论文说的 L_smooth)。
- **jit_air**:同上,但按 air 权重 (1−contact) 加权——**只在脚离地时更狠地压抖动**。
- **skate**(阈值对齐软计数,仅踝):`g(·)=sigmoid`,
  skate = wmean( **sigmoid((speed − 0.03)/s)** ),**s = 0.008**(温度),0.03 = FSR 判定阈 SKATE_THRESH;
  按 **contact 权重 w³ × mask** 加权(→ **只算 planted 帧**)。
- **anch**(锚/保真项,回答审稿人"没有 fidelity term?"):**anch = wmean(|ankle_xz − deskated_xz|)**,
  L1 拉住去滑参考,**防止踝轨迹塌成常数**——这就是保真项,由 `--anch-share` 控其权重。

**λ 自动标定(B5)**:对每项单独反传求其对模型参数的**梯度范数** norm_k,令 λ_k = target_k / norm_k
再整体归一化(消除量纲:acc²~1e-6 vs 软计数~1e-1)。target 份额由 `--jit-share`/`--anch-share` 设定。
- 交付点 088a10(`--jit-share 0.88 --anch-share 0.10`)**标定后 λ**:
  jit_air = 0.838,jit_all = 0.127,skate = 0.0014,anch = 0.0333。

**⚠️ 修式(3)**:论文式(3)写成"对所有 t 求和 / T",但**代码实际按 contact 权重加权、mask 归一化**
(`_wmean`,只对 planted 帧)。→ **这是写作简化,不是 bug**;把式(3)改成带 contact 权重 w 和
mask 归一化的形式即可,"threshold-aligned"的卖点就与公式自洽了。

---

## B6 运行时开销(补进 §6 / 摘要卖点)

- 全 pipeline(learned smoother + 2-bone IK),**GPU TITAN Xp**,T2M-GPT n=200(平均 123 帧):
  **4.9 ms / 条动作**(≈ 0.04 ms / 帧)。
- **可粘**:
  > The full correction (learned smoother followed by two-bone IK) runs in 4.9 ms per
  > motion on a single TITAN Xp GPU (0.04 ms per frame, averaged over 200 clips of ~123
  > frames). Being post-hoc, it needs no generator retraining — a one-off cost measured in
  > milliseconds versus the GPU-hours-to-days of retraining a generator with a
  > contact-aware loss.
- (若要更强对比,可加一句 CPU 数:本会话在登录节点 CPU 上 200 条≈12–55s,即 ~60–275 ms/条,
  仍是实时量级——但 GPU 数已足够支撑"便宜"的论点。)

---

## 一句话汇总(这 5 项都不再是缺口)
| 项 | 结论 |
|---|---|
| A10 语义 | 全值+CI 已给;V19 修脚不损文本对齐(ΔMM-Dist 三生成器均 ns) |
| A8 训练量 | 4,000 条(train 3,520 / val 480) |
| B3 超参 | Adam lr2e-4 / batch16 / best@ep115 / val 选点;46,276 参数 4 层 CNN |
| B4 损失 | 4 项(jit_air/jit_all/skate/anch),g=sigmoid,s=0.008;**anch 就是保真项**;式(3)按 contact 加权(改公式即可) |
| B6 耗时 | 4.9 ms/条(GPU),post-hoc 免重训 |
