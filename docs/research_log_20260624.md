# MotionFix 研究日志 — 2026-06-24

> 本文件完整、忠实地记录本次研究会话(2026-06-23 ~ 2026-06-24)的全部过程、
> 代码、结果数据、失败诊断、导师反馈与下一步决策。数据均直接取自实际运行输出。

---

## 0. 起点状态

会话开始时已有 V8_new(坐标修复版):
- 模型:`models/v8.py`,Transformer Encoder ×6,19.1M 参数
- Checkpoint:`checkpoints/v8/best.pth`(epoch 50,train_loss 0.0114)
- 50 个 MoMask 结果:FSR 14.1%→15.6%(变差),Jitter 0.0128→0.0278(×2.2),FootErr 0.0098m

核心已知问题:训练 loss 下降但实际效果退化(坐标系不匹配 + selective_replace 副作用)。

---

## 1. 用户三项任务

1. 按"只在脚关节 + 仅接触帧 + 水平滑动 → 学习修脚滑"的新方案,生成 V14
2. 把优先级 1+2 的 7 个指标作为以后所有版本的统一评判标准
3. 执行过程生成日志

---

## 2. 七指标统一评估框架(`utils/metrics.py`)

**优先级 1(核心):**
1. **FSR** — Foot Skating Ratio,接触帧中脚滑帧比例
2. **Jitter** — 脚部加速度 RMS
3. **Floating** — 接触标签帧中脚实际悬空的比例
4. **FootErr** — 修复后 vs 原始的脚部平均位移

**优先级 2(强烈建议):**
5. **BoneCV** — 骨骼长度变异系数(std/mean),时间稳定性
6. **Penetration** — 脚穿透地面深度(mean/max)
7. **ContactAcc** — 接触标签保持准确率

学术界对标(MaskControl ICCV 2025):Real FSR 0.0%,OmniControl/MaskControl 5.5%,
MoMask ~14.1%(我们测得),MDM 10.2%,GMD 10.1%,PriorMDM 9.0%。
完整评估还需 FID / Diversity / R-precision(需 HumanML3D evaluator)。

来源:
- Ismail-Fawaz et al., CVIU 2025, "Establishing a Unified Evaluation Framework for Human Motion Generation"
- MaskControl ICCV 2025; OmniControl; BioMoDiffuse; HumanScore (Stanford)

---

## 3. V14 — 模拟脚滑训练(失败)

### 3.1 数据制备(`data/prep/v14.py`)
- 源:HumanML3D 8177 个 motion → 转 (T,22,3)
- 失真:只在脚关节 [7,8,10,11]、只在 XZ、只在接触段,加平滑滑移漂移
- 非脚关节、Y 轴、非接触帧:零改动(已验证 max diff = 0.0000000000)
- 产出:**15980 训练对**(7990 motion × 2 增广)
- 统计:Total XZ drift/sample mean=15.63m,Max XZ drift/sample mean=0.0894m,~97% 帧有脚滑

### 3.2 训练(`training/v14.py`)
- 初次 num_workers=2 卡死;改打包 `data/training/v14_packed.pt`(4000 对,414MB)后 18× 加速
- 25 epochs,8 分 47 秒,**best loss 0.0521**(epoch 24)

### 3.3 测试结果(50 MoMask)
```
              FSR        Jitter     Floating  FootErr  ContactAcc  BoneCV   Penetration
Original     14.1%      0.0128      —         —        —           —        —
V14 α=0.5    15.6%      0.0286      0.0%      0.0102m  100.0%      0.0212   mean 0.0046m
```
**结论:FSR 没改善(15.6%,比原版还差),Jitter ×2.2。失败。**

### 3.4 Alpha sweep(p000021 + 10 motion)
```
Alpha   FSR_aft  Jitter   Ratio
0.10    26.8%    0.0116   1.1x
0.50    26.8%    0.0289   2.6x
0.90    27.1%    0.0507   4.6x
1.00    22.9%    0.0562   5.1x
```
50 motion 全量 α=0.9:**FSR 14.1%→14.1%(不变),Jitter ×3.7**
- 部分动作大幅改善:p004822 -17.8pp, p002627 -12.5pp, p012529 -7.5pp, p011997 -7.0pp
- 部分恶化:p009958 +6.1pp, p002104 +6.2pp, p009613 +4.3pp
- 发现:uniform alpha 无法区分好坏修正,等权放大

---

## 4. 物理修正(成功,迄今最佳 FSR)

### 4.1 方法(`utils/physics_fix.py`)
规则,无模型:接触帧(脚高 < ground+5cm)且水平速度 > skate_thresh 时,
阻尼超出阈值的速度。damp_factor=0.0 = 把速度压到刚好阈值。

### 4.2 damp sweep(50 MoMask)— 修正速度级联 bug 后
```
Version           FSR     Jitter   Float  FtErr   ContactAcc  BoneCV
Original          14.1%   0.0128   —      —       —           —
Physics damp=0.0  9.5%    0.0347   0.0%   0.0157  99.9%       0.0332   ← 最佳 FSR
Physics damp=0.1  16.8%   0.0318   0.0%   0.0141  99.9%       0.0303
Physics damp=0.3  19.6%   0.0173   0.0%   0.0081  100.0%      0.0247
```
**damp=0.0:FSR 14.1%→9.5%(-33%)。所有版本中唯一真正显著降 FSR 的方法。**

### 4.3 可视化分析(`analysis/physics_viz/`,stats.json)
```json
{
  "n_motions": 50,
  "FSR": {"before_mean": 0.1405, "after_mean": 0.0954, "delta_mean": -0.0451,
          "improved_count": 41, "worsened_count": 1},
  "Jitter": {"before_mean": 0.0128, "after_mean": 0.0347, "ratio": 2.70},
  "FootErr": 0.0157, "Floating": 0.0, "BoneCV": 0.0332
}
```
- **41/50 改善,1 恶化(p009958 倒退行走 +4.5pp),8 不变**
- 6 张图:01_trajectories_showcase, 02_fsr_barchart, 03_fsr_histogram,
  04_fsr_vs_jitter, 05_best_worst_trajectories, 06_metric_distributions
- 代价:Jitter ×2.7(damp=0.0 硬冻结脚 → 边界跳变)

---

## 5. V15 — 物理教师模型(失败:退化为恒等)

### 5.1 思路
物理修正 = 正确答案。训练对:MoMask(脚滑)→ 物理修正版(干净),root-relative。
- `data/prep/v15.py`:50 motion 分 40 训练 / 10 held-out(分层),5× 增广 = 200 训练对
- `training/v15.py`:V14 架构,100 epochs,4 分钟,best loss 0.2734

### 5.2 测试结果(10 held-out)
```
              FSR      Jitter    FootErr   BoneCV
Original     16.3%     0.0140    0.0000    0.0050
Physics      11.9%     0.0319    0.0140    0.0310   ← 教师
V15          18.2%     0.0169    0.0038    0.0119   ← 模型
```
**结论:V15 学到接近恒等映射(FootErr 仅 4mm)。FSR 没学到(18.2% 比原版还差)。
200 对对 19M Transformer 远远不够。**
- 唯一正面:V15 Jitter 0.0169 < 物理 0.0319(更平滑)

### 5.3 根因
L1-to-target 损失被 88% 应保持不变的维度主导 → 脚-XZ-接触帧的信号被冲淡 → 退化。

---

## 6. V16 — 自监督抗作弊损失(失败:钻代理空子)

### 6.1 思路(`models/v16.py`)
放弃 L1-to-target,直接优化目标(自监督,无 target):
```
Loss = λ_skate·soft_FSR + λ_smooth·jitter + λ_anchor·‖foot-input‖ + λ_preserve·‖nonfoot/Y-input‖
权重:skate=10, smooth=5, anchor=2, preserve=20
```
设计意图:soft_FSR 与 jitter 互相拉扯,迫使模型渐进减速而非硬冻结(抗抽搐)。

### 6.2 训练(`training/v16.py`)
40 motion 自监督,150 epochs,3 分 6 秒。四个损失分量同步下降:
skate 0.0545→0.0067(↓8×),smooth 0.0114→0.0014,anchor 0.2375→0.0872,preserve 0.5996→0.0604

### 6.3 测试结果(10 held-out)— 诊断关键
```
              FSR      Jitter    FootErr
Original     16.3%     0.0140    —
V16 full     35.8%     0.0056    0.12-0.25m   ← 模型原始输出
V16 blend    17.9%     0.0266    —            ← 50% blend 掩盖了灾难
```
**结论:模型钻了 soft-FSR 代理损失的空子。它输出一条全新的、超级平滑的轨迹
(Jitter 0.0056 满足 smooth,速度适中满足 soft_skate),但脚跑到 12-25cm 外、
FSR 35.8% 比原版差一倍。可微 soft-FSR ≠ 硬 FSR,模型利用了这个缝隙。**
这正是用户最初担心的"为了更小的值出现不符合实际的效果",发生在代理 vs 真实指标层。

---

## 7. 五次尝试的规律总结

| 方案 | 范式 | FSR | 失败原因 |
|------|------|-----|----------|
| V8 | L1 模拟噪声 | 15.6% ❌ | 信号冲淡→恒等映射 |
| V14 | L1 模拟脚滑 | 15.6% ❌ | 同上 |
| V15 | L1 物理教师 | 18.2% ❌ | 88%维度答案"不改"→恒等映射 |
| V16 | 自监督代理损失 | 35.8% ❌ | 钻代理与真实指标的空子 |
| **Physics** | **纯规则** | **9.5%** ✅ | **直接操作真实量,无法作弊** |

**教训:脚滑修正本质是约束满足问题(接触时脚速=0),不是学习问题。
学习模型要么信号太弱(L1 被主导)退化,要么给它任何可微代理就被钻空子。**

---

## 8. 导师 Frederick 反馈(原文)

> Dear Xin,
>
> Thank you for sending the progress report. You have made a more technical effort
> this round, especially by expanding the test set, adding additional metrics beyond
> Skating Ratio, and iterating on the V8/V9 correction pipeline. It is also good that
> you have started to think about cross-model behaviour rather than only testing a
> single baseline.
>
> That said, the report still needs to show more clearly what your own dissertation
> contribution is, beyond evaluation and iterative adjustment of existing ideas. At
> the moment, V8 looks like a promising selective foot-replacement strategy, but it
> is still only a partial fix: the gains are limited on rotational motions, and the
> transfer to T2M-GPT is negative. V9 is also not yet stable, so the method is not
> ready to be treated as a finished solution.
>
> Main points to address
> - The contribution should be stated more explicitly: what is genuinely new in your
>   pipeline, and why is it needed?
> - The evaluation needs tighter ablation evidence so it is clear which part of the
>   method helps, e.g. gating, smoothing, IK, or foot-only replacement.
> - The cross-model results need a more careful explanation, especially why MoMask
>   improves but T2M-GPT degrades.
> - The current method still looks prompt-sensitive, so you should show stronger
>   generalisation across motion categories.
> - FRDM / InfiniteDance is a useful reference, but make sure you separate what comes
>   from the paper from what is your own adaptation idea.
>
> What to do next
> Please focus on turning the current work into a clearly defensible MSc project
> contribution:
> - stabilise one method rather than keeping too many variants open,
> - add ablations and failure analysis,
> - clarify the technical novelty of your correction pipeline,
> - and make the evaluation show both physical plausibility and semantic preservation.
>
> At this stage, the project is moving in a better direction, but it still needs
> sharper definition and stronger evidence to be convincing as a final dissertation
> contribution.
>
> Regards,
> Frederick

---

## 9. V17 — 物理 + 学习平滑混合(部分成功,揭示根本对抗)

### 9.1 设计(`models/v17.py`)
- Stage 1:物理修正(damp=0.0)→ FSR 低
- Stage 2:**FootSmoother,1D 时序 CNN,25,736 参数**,残差式,只动脚 XZ,
  接触感知 Gaussian target,anti-skate 守护。脚 Y + 非脚关节直通。

### 9.2 训练演化
- 初版损失(smooth=acc²)→ 平滑器退化为不动(残差≈0),因 mean-acc² 梯度太弱
- 改 Gaussian-target 损失(match + anti-skate)→ 开始学习
- 接触感知 target(接触=物理冻结,空中=平滑)→ best loss 0.0136

### 9.3 测试结果(10 held-out,3-way 消融)
```
Metric          Original    Physics     Hybrid(V17)
FSR ↓           16.3%       11.9%       20.2%
Jitter ↓        0.0140      0.0319      0.0244
Floating ↓      0.0%        0.0%        0.0%
FootErr         0.0000      0.0140      0.0600
ContactAcc ↑    100.0%      100.0%      100.0%
BoneCV ↓        0.0050      0.0310      0.0489
Penetration ↓   0.0111      0.0111      0.0111
```
语义保持:非脚关节 max change 0.000000m,脚位移 0.060m,关节-帧修改 17.2%。

### 9.4 关键发现:FSR–Jitter 根本对抗
```
低 FSR  → 接触时脚速→0 → 平段+硬边界 → 抖动高
低抖动  → 脚速度平滑     → 接触时脚在动 → FSR 高
```
平滑器降了 Jitter(0.0319→0.0244,-24%),但把冻结接触段抹出速度,
FSR 涨到 20.2%(比原版还差)。**当前混合方案不是 Pareto 最优。**
物理的 damp_factor 其实已在描绘这条 Pareto 前沿。

---

## 10. 用户的核心约束(2026-06-24 明确)

1. 必须真正**同时降 FSR 和 Jitter**(不是 trade-off)
2. 必须有**学习/模型成分**——纯物理工作量不够,且已向导师汇报学习方法
3. **后训练(post-hoc)**范式不变
4. 纯物理视觉上是目前最好的,但比原始有**轻微抽搐**(残留 jitter)

---

## 11. V18 提案 — 接触约束的学习式足部精修(下一步)

### 核心:硬约束与学习解耦
```
接触帧:  脚速度 = 0   ← 硬性 mask(非学习)→ FSR 保证低,模型无法作弊
空中帧:  脚速度 = 模型预测  ← 学习部分,目标平滑 + 贴近原始
```
模型只负责空中阶段(落地前减速、离地后加速)。空中帧不计入 FSR,
故模型怎么调都不会让 FSR 变差。

### 为何能同时降两个
- FSR:接触硬约束速度=0 → 保证低(强制,非学习,不可作弊)
- Jitter:模型学空中减速 → 平滑边界 → 降抖动

### 为何这次不同
**FSR 不再交给学习,而是焊死。** 模型唯一工作是平滑——最坏抖动=物理水平,
最好抖动降到原始附近,FSR 始终安全。消掉了 V8-V17 反复踩的所有失败模式。

### 论文叙事
> 朴素后训练学习修正会失败(已系统证明)。提出接触约束的学习式足部精修:
> 将不可妥协的硬约束(接触零速,物理启发、不可作弊)与可学习目标(空中平滑)
> 解耦,首次在后训练框架下同时降低 FSR 和 Jitter。
> 诊断 + 方法(约束解耦)+ 评估框架 + 跨模型 = 完整 MSc 贡献。

### 风险
模型空中平滑必须真的把 Jitter 降到物理以下。但与以前不同:以前赌"FSR 会不会失控"
(总失控),现在 FSR 已焊死,只剩"Jitter 能不能降"这一更可控的问题。

---

## 12. 产出文件清单(本次会话)

```
utils/metrics.py                  7 指标统一框架
utils/physics_fix.py              物理修正(规则)
data/prep/v14.py                  V14 脚滑模拟数据制备
data/prep/v14_pack.py             打包数据集(18× 加速)
data/prep/v15.py                  V15 物理教师数据(40 train/10 test 分层)
models/v14.py                     V14 模型 + V14Loss
models/v16.py                     V16 模型 + V16Loss(自监督抗作弊)
models/v17.py                     V17 FootSmoother + SmootherLoss + hybrid_fix
training/v14.py training/v15.py training/v16.py training/v17.py
testing/v14.py testing/v14_alpha09.py testing/v15.py testing/v16.py testing/v17.py
testing/physics_sweep.py          物理 damp sweep
analysis/viz_physics.py           物理可视化(6 图)
analysis/sweep_alpha.py           V14 alpha sweep
checkpoints/v14|v15|v16|v17/best.pth
outputs/fixed/v14|v14_alpha09|v15|v16|v17/   各版本输出 + VERSION.md
analysis/physics_viz/  analysis/v16_viz/  analysis/v17_viz/   图 + summary.json
logs/v14_train.log v15_train.log v16_train.log v17_train.log
logs/v14_test.log v15_test.log v16_test.log v17_test.log
logs/physics_fix.log
docs/research_log_20260624.md     本文件
```

---

## 13. 当前决策点

V18(接触约束学习式精修)已提案,满足用户全部约束(同时降两指标 / 有学习成分 /
后训练 / 不可作弊)。等待用户确认是否开始搭建 V18,或调整设计。
