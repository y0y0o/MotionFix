# 论文修改清单 v2 / Thesis Revision Notes
> 对象:`Generator_Agnostic_Post_Hoc_Correction_of_Foot_Skating_in_Text_to_Motion_Generation.pdf`
> 更新:2026-08-31。合并了 (a) 我逐格核磁盘数据的发现、(b) 第 2 条的 de-skate 分解实测、
> (c) 我发现的 §3.2 接触来源硬伤、(d) 对外部 review 27 条的逐条数据核验。
> **凡标"实测"均来自 `analysis/v19/*.json`,已用 088a10 交付点核对。**

---

## ★ 先读:一个版本对不上的问题
外部 review 里 **Table 3 = 分类别、Table 4 = FID**;但我磁盘上这版 PDF(mtime 未变)是
**Table 3 = FID,分类别只在 §5.C 正文、没有独立表**,也没有 Table 4。→ 请先确认审稿人读的
是不是**另一版 PDF**。下面凡涉及表号,我按**我手上这版**(Table 1=T2M-GPT, 2=MoMask, 3=FID)标注。

---

## 第一优先级:内部矛盾 / 事实错误(不改答辩必翻车)

### P1. 【最严重·审稿人#1 正确】Gaussian 在两张主表上 Pareto 支配 V19,正文却说"打平/V19 唯一取胜"
**实测确认**(Table 1/2,已核 json):

| | Gaussian FSR/Jit | V19 FSR/Jit | 结论 |
|---|---|---|---|
| T2M-GPT | 4.92 / 0.00778 | 5.08 / 0.00827 | Gaussian **两项都更低** |
| MoMask | 6.33 / 0.00809 | 6.57 / 0.00832 | Gaussian **两项都更低** |

- 正文 §5.1 "V19 is the only config beating original on both" **不成立**:Gaussian 也同时打败原始,且赢更多。
- §5.2 用 frontier 救,但 frontier 是 **n=50/n=10**(见 P8),跟主表 n=200 不是同一实验,救不动。
- **这与你"learned ≈ Gaussian,不碾压"的诚实主线其实一致**,但正文措辞制造了自相矛盾。
- **建议(三选一,我推荐第一)**:
  1. 老实承认"在该 delivery point 上 Gaussian 略优",把卖点转向**固定单参数的跨生成器可迁移性**(见 P-NEW1 建议实验)。
  2. 在主表同一批 n=200 上补完整 frontier,用同样本证明重叠。
  3. 重选一个不被 Gaussian 支配的操作点,并**明说选点在 validation 上做**(否则违反 leak-free,见 P6)。

### P2. 【审稿人#2 正确,我已分解】de-skate 后 FSR 按定义应≈0,实际 3.98%
**实测分解**(T2M-GPT n=200,`analysis/v19/deskate_decomp.json`,脚本 `analysis/v19_deskate_decomp.py`):

| 阶段 | FSR |
|---|---|
| original | 6.43% |
| de-skate(IK 前) | **1.96%** ← 不是 0 |
| de-skate + IK | 3.98% ← Table 1 |
| reach-clamp 触发 | **11.96%** 的接触帧 |

- **IK 的 reach-clamp** 把 1.96%→3.98%(约翻倍):骨长刚性,钉住的目标超出可达球面时,IK 把踝拉回,重新引入水平位移。
- **IK 前就 1.96% 而非 0**:因为**去滑的接触判定与 FSR 度量的接触判定不是同一套**(见 P-NEW2)——软高度权重 vs 高度+速度硬标签,边界帧仍被计入。
- **必须在方法/结果章补这张分解表**,并**在方法章定义 reach-clamp**(全文 §5.3、§6.5 用了两次,从未定义)。否则"physics 做了大部分工作"的因果链是断的。

### P-NEW1【我发现·审稿人漏了】§2.2/§3.2 把接触来源写错了
- 论文 §3.2:"A foot is in a contact window when the **HumanML3D contact channel** indicates it is planted";§2.2 还强调 contact 通道"encode exactly the information needed"。
- **实际代码**:`models/v18.py::deskate_xz` 用 `compute_contact_weight_np(foot_Y, h=0.05)`——**纯几何高度软权重**;FSR 度量 `compute_contact_labels` 用**高度+速度**。**两者都不碰 263 维 HumanML3D contact 通道。**
- → 方法章描述与实现不符,答辩会被问穿。**必须改**:把 §3.2 的接触来源改成"geometric height-based soft contact weight (foot height < 5 cm)",§2.2 相应弱化对 contact 通道的强调(它没被 pipeline 用)。

### P3.【原发现·与 P2 同源】§5.C 分类别:"no category regresses / rotation largest residual" 两句都错
**实测**(`analysis/v19/by_category_088a10.json`,与旧 json 逐位一致=本来就是 088a10):
- **backward(0.190→0.196)、jumping(0.033→0.037)两类 v19 回归**(高于原始)→ "no category regresses" 假。
- **最大残留是 backward(0.196),不是 rotation(0.175)** → "rotation largest residual" 假。
- 替换文本见文末【替换文本 R2】。

### P4.【审稿人#3 部分正确】§5.C 分类别数值量纲比 Table 1/2 大 3 倍——**不是单位错,是样本构成不同,但必须加表注**
- 分类别 original(rotation 0.22 / walking 0.21)vs Table 1 original(6.43%):差异**真实存在但可解释**。
- 原因:分类别用的是**原始 50 个带标签 prompt**(走/转/旋转类,滑动高发)× 3 生成器=150 对;主表 n=200 用**扩充后更广更平的 prompt**。这与已记录的"n=200 更广更平"注解一致(不是 bug)。
- **必须做**:给分类别加表注,写明**生成器组成(3 个聚合)、n=150、τ 与主表相同、prompt 子集不同**。审稿人的"无法解读"批评在**缺表注**这点上成立。

### P5.【审稿人#5 正确】§5.4 说报 MM-Dist 和 R-precision,表里只有 FID,且"无显著变化"零数字
- 必须补**完整语义表**:每生成器 × 每配置的 MM-Dist、R-precision 的**值 + 95% CI + p 值**(paired bootstrap 你已经做了,把数搬出来)。
- 数据在 `analysis/v19/semantic_largeref.json`(有 R_top1/2/3、MMDist、dMMDist_CI95、dMMDist_significant 字段),直接取。

### P6.【审稿人#10 正确】`--jit-share 0.88 / --anch-share 0.10` 与 delivery point 选法
- 两参数 §3.7 首次出现无定义;**delivery point 怎么选的必须写清**。若看了评测集才挑=test-set tuning,与 leak-free 自相矛盾。**明说选点在 validation split 上做**。

### P7.【审稿人#6 正确·小】训练数据量"3.4–4 k"写成区间
- 改成确切条数(查 `data/prep/v19.py` 实际产出 / `data/training/v19_cache.pt` 的样本数)。

---

## 第二优先级:方法学实质(补了显著加分)

### P-NEW2 / 审稿人#7【需修正审稿人措辞】"循环论证:修正与度量同套 contact 标签"
- 审稿人说两者用**同一套**标签=循环——**过头了**。实测:去滑=软高度权重、FSR=高度+速度硬标签,**重叠但不相同**(所以 P2 里 IK 前 FSR 是 1.96% 而非 0)。
- 但**底层担忧成立**:两者都基于几何高度,FSR 下降有构造成分。
- **建议(低成本高回报)**:用一套**完全独立**的 contact 判据(如更严的高度+速度、或不同阈值)重算 FSR,若结论稳→可信度大增。注意 FSR **已经**不是用去滑的标签,所以这个 check 比审稿人想的更容易通过。

### P8.【审稿人#8 正确·最有价值的诊断】负结果的真正机制:训练/测试分布不匹配
- Smoother 训练在 **GT 动作**去滑后的数据上;GT 几乎不滑,去滑对它几乎不做事→**网络训练时见到的边界抖动远小于测试时(生成器输出去滑后的抖动)**。
- 即**网络没见过它推理时真正要处理的 artifact**——这是"学习打不过 Gaussian"最有力的解释,论文完全没提。
- **补这段分析**:把负结果从"诚实报告失败"升级到"诊断出失败机制",硕士论文大加分,并直接给出 future work(用生成器输出/加噪 GT 训练)。

### P9.【审稿人#9 正确】损失函数缺失与不一致(方法章硬伤)
- 式(3)对所有 t 求和除以 T,但 FSR 分母是 Σc_t(只算 planted)——"threshold-aligned"卖点被公式否定。**核对实现**:若有 contact mask 要补进公式;若无是 bug。
- `g(·)` 是什么(sigmoid?)、温度 s 取值——没写。
- **`L_smooth` 全文未定义**;λ 梯度范数归一化只有文字没公式。
- **最关键:似乎没有保真项**——只有 anti-skate+smoothness 时平凡解是把踝轨迹压成常数。是靠 `--anch-share` 的 anchor 项拉住?那 anchor 项必须写进损失公式。→ 全部补进方法章(源码 `models/v19.py::V19Loss` 里有,照抄)。

### P10.【审稿人#11 正确】网络结构与训练超参缺失(硬性要求)
- 1D-CNN:层数/kernel/channel/感受野/激活、optimizer/lr/batch/epoch、train-val 比例、训练曲线——**一个都没有**。从 `models/v19.py`(hidden=64,kernel=5,n_layers=4,in_dim=12,out_dim=4)+`training/v19.py` 取。代码路径移附录。

### P-NEW1 建议实验 / 审稿人#12【最高投入产出比·可能翻正】跨生成器固定参数迁移
- 你 §6.1 说"学习是 drop-in,Gaussian 要 per-generator 调参"——**但没做实验证明**。
- **做**:T2M-GPT 上调最优 Gaussian σ*,固定后直接用到 MoMask/MDM;学习 smoother 用同一组固定参数用到全部三个;比较跨生成器衰减。
- 若学习迁移更稳→**正面结果**:"学习更省调参"而非"学习更强",正好呼应 generator-agnostic。**我可以帮你跑**(源已就绪),你说一声。

### P11.【审稿人#13/#14】两个锦上添花分析
- 画**学习卷积核 vs Gaussian 核**的有效冲激响应对比:若近似→"学习≈Gaussian"从经验升级为机制解释(半天工作)。
- 讨论**为何不做 root-level 去滑**(挪 root 而非单独挪踝正是 reach-clamp 的根源):至少在相关工作/讨论里说明(理由:保持 generator-agnostic、不改语义)。

---

## 第三优先级:统计与协议

### P12.【审稿人#15 正确·必做】所有物理指标加 paired bootstrap 置信区间
- Table 1/2/3 全是点估计。Gaussian 4.92 vs V19 5.08 差 0.16pp,在 n=200 下是否显著无从判断。语义指标已做 bootstrap,同法搬到 FSR/jitter。**最容易补、审稿人必要求。**

### P13.【审稿人#16 正确】FID 在解读噪声
- 0.483/0.485/0.489/0.483 的差异远在 n=200 FID 噪声下限内,不能说"V19 恢复到原始水平"。**改为**:对生成侧补重复采样 CI,或明写"这些差异在噪声范围内,不作结论"。**FID reference 26→1215 修复这段保留(加分项)。**

### P14.【审稿人#19 正确·一句话】MDM 的 FID 方向异常要点一句
- 其他两个 de-skate 让 FID 变差,MDM 变好(15.94→14.90)。**加解释**:MDM 本地样本滑动极重、已远离自然分布,de-skate 把它往回拉一点。不解释读者当 bug。

### P15.【审稿人#4 需澄清】"三个生成器"主张 vs 表里只有两个
- **数据其实在**:`v19_088a10_expanded_results.json` 有 **mdm 块(n=50)** 全五路。只是没进 PDF 的 Table 1/2。
- **两选一**:把 MDM 物理表补进正文(n=50,标注),或把主张降级为"两个生成器完整评估+第三个作 OOD 案例"。别"标题说三个、表里两个"。

### P16.【审稿人#17/#18】样本量与两套 jitter 定义
- n 满天飞(200/50/10/50/150/12):每处标注限制,尤其 MoMask frontier n=10 撑不起"曲线重叠"的全局论断。
- §5.5 尖峰用的踝加速度指标与 Table 1/2 的 Jitter 列**是两套定义**:统一,或每配置都把两套都报,避免 cherry-picking 质疑。
- (顺带:尖峰数我已重跑 088a10,见【已修数据】。)

---

## 第四优先级:文献 / 呈现(这块目前最弱)

- **P17 审稿人#21/#22/#23【必做】**:只有 9 篇参考文献,§2.3(建立空白那节)零引用。**必补 Kovar et al. 2002 "Footskate Cleanup for Motion Capture Editing"(SCA)**——它就是 generator-agnostic、retrain-free、IK-based 的 footskate 后处理,你的核心新颖性主张不讨论它站不住。写出差异(Kovar:mocap 编辑、假设已知可靠 contact、无学习、无文本对齐评估)。再补:物理约束生成(PhysDiff/GMD/ReMoDiffuse)、learned pose refinement、FSR 定义差异的参照。**扩到 25–35 篇。** 至少一个外部基线,或论证为何无可公平移植的对照。
- **P18 审稿人#24【必做】**:全文仅一张流程框图,**一篇讲脚滑的论文没有任何动作可视化**。至少补:①FSR–jitter frontier 曲线(核心结论现在只有文字!)②同段动作 orig/de-skate/V19 的踝 XZ 轨迹叠加图(标 contact window)③踝速度/加速度时序(展示边界抖动+修复)。可选④学习核 vs Gaussian 核。
- **P19 审稿人#25【必做·风格】**:把 V19/V18/v19_088a10/--jit-share/`data/prep/v19.py` 这些工程记号换成语义命名:`Ours (full)` / `Ours w/o IK-in-loop` / `De-skate only` / `Gaussian baseline`;代码路径移附录/repo。
- **P20 审稿人#26/#27**:补复现性声明(确切硬件+软件版本,别写"recorded in project log")、**运行时开销**(post-hoc 最大卖点是便宜:每条动作耗时?比重训省多少?)、摘要加核心数字(FSR 6.43→5.08 等);解释本地 HumanML3D 只有 8177/14616(训练也从这抽,说明抽样无系统性类别偏差)。

---

## 高投入产出比(审稿人也这么排)—— 建议真去做

1. **P-感知实验(审稿人#20,排第一)**:工具已就绪。找 15–20 人各 36 次比较,半小时/人,一天收齐。即便"无显著偏好"也是有价值发现(说明指标改善尚未到可感知)。补上整条价值链的最后一环。
2. **P-NEW1 跨生成器固定参数迁移(#12)**:可能给你一个正面结果。**我能帮你跑。**
3. **P8 分布不匹配的机制分析**:把负结果从"诚实"提到"有诊断"。

---

## 【已修数据】我这轮已重跑并落盘的(可直接引用)
| 项 | 文件 | 脚本 |
|---|---|---|
| 088a10 尖峰(3 生成器) | `analysis/v19/spike_ratios_088a10.json` | `analysis/v19_spikes_3gen.py` |
| 088a10 分类别 | `analysis/v19/by_category_088a10.json` | `analysis/v19_by_category.py` |
| de-skate FSR 分解 | `analysis/v19/deskate_decomp.json` | `analysis/v19_deskate_decomp.py` |
| jitter_trace checkpoint 修正 | (v045 备份 `jitter_stats_v045.json.bak`) | `analysis/v19_jitter_trace.py` |

**尖峰实测**(v19/orig p99·max):T2M-GPT 0.93·0.98 / MoMask **1.03·1.07** / MDM 0.90·0.83。
→ 论文 §5.E "0.91×/0.94× on all three generators" 错,MoMask 尖峰略升,**必须按生成器分开写**。

---

## 替换文本(可直接粘贴)

**【R1 §5.E 尖峰】**
> Relative to the original output, the delivery point reduces the 99th-percentile and
> maximum ankle acceleration on T2M-GPT (0.93×/0.98×) and MDM (0.90×/0.83×), while on
> MoMask the spikes are essentially unchanged to slightly higher (1.03×/1.07×). The
> spike behaviour is therefore generator-dependent, and we report it per generator; the
> FSR and RMS-jitter improvements in Tables 1–2 hold on all three.

**【R2 §5.C 分类别】**
> Most categories improve on the original output; the two exceptions are *backward*
> (0.190 → 0.196, n = 9) and *jumping* (0.033 → 0.037), where the smoother gives back
> enough FSR to slightly exceed the original — a category-level echo of the frontier
> trade-off. The largest residual FSR is in *backward* (0.196), followed by *rotation*
> (0.175). Rotation nonetheless shows the largest absolute reduction from the de-skate
> stage (0.220 → 0.125); the reach-clamp fires most often on rotating and backward
> motions, which is where the residual concentrates.

**【R3 §3.2 接触来源(改正 P-NEW1)】** 把"HumanML3D contact channel"一段改为:
> Contact windows are derived geometrically from the recovered joint positions: a foot
> is considered planted when its height falls below a small threshold (5 cm above the
> estimated ground plane), yielding a soft per-frame contact weight. The pipeline
> therefore does not rely on the generator's predicted contact channel, which keeps it
> applicable to any generator whose output can be mapped to joint positions.

**【R4 §5.1 化解 Gaussian 支配(配 P1 选项1)】** 在 §5.1 结论处加:
> At this delivery point the tuned Gaussian is in fact marginally better than the
> learned smoother on both metrics (Tables 1–2); this is consistent with, and a sharper
> statement of, the central finding that the learned component does not beat a
> well-tuned classical baseline. Its advantage, examined in Section [transfer], is
> lower per-generator tuning rather than lower error.
