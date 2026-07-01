# MotionFix: Foot Skating Correction for VQ-Based Motion Generation

**Student:** Xin Wan  
**Date:** 16 June 2026  
**Progress Report — Supervisor Meeting**

---

## 1. Research Problem

VQ-based text-to-motion models (MoMask, T2M-GPT) offer **15–20× faster inference** than diffusion-based models (MDM), making them suitable for real-time applications. However, they suffer from **foot skating artifacts** — feet slide along the ground plane when they should remain stationary during ground-contact phases. This is a well-known limitation of VQ-VAE-based motion generation, caused by quantization error accumulation in the discrete latent space.

**Goal:** Develop MotionFix — a lightweight post-processing neural network that corrects foot skating artifacts in VQ-generated motions, without requiring retraining of the underlying T2M models.

---

## 2. Baseline Analysis

### 2.1 Skating Ratio Metric

Skating Ratio (SR) is defined as:

$$\text{SR} = \frac{\text{\# frames where foot slides during ground contact}}{\text{\# frames with ground contact}}$$

- Ground contact: foot height < 5th percentile + 5cm
- Sliding: foot horizontal velocity > 0.03 m/frame

### 2.2 MDM vs MoMask vs T2M-GPT

Evaluated on 50 test prompts across categories (walking, rotation, turning, jumping, etc.):

| Model | Type | Avg Skating Ratio | Notes |
|---|---|---|---|
| MDM | Diffusion | — | Lowest skating overall |
| MoMask (no_IK) | VQ-based | 13.6% | Without IK post-processing |
| MoMask (IK) | VQ-based | 12.7% | With built-in IK reduces skating by ~0.9pp |
| T2M-GPT | VQ-based | — | Different artifact pattern (see §4.2) |

**Key finding:** MoMask exhibits ~14% average skating ratio, significantly higher than MDM. IK preprocessing provides a modest improvement (~0.9pp) but does not eliminate the problem.

---

## 3. MotionFix Method

### 3.1 Architecture (V8 — Current Best)

MotionFix is a Transformer Encoder network that takes a full motion sequence and outputs corrected joint positions:

| Component | Specification |
|---|---|
| Input | (T, 22, 3) joint positions → flattened to (T, 66) |
| Input Projection | Linear(66 → 512) |
| Positional Encoding | Sinusoidal, max_len=500 |
| Transformer Encoder | 6 layers, d_model=512, 8 heads, FFN=2048 |
| Output Projection | Linear(512 → 256) → ReLU → Linear(256 → 66) |
| Total Parameters | **19.1M** (~76 MB) |
| Inference Overhead | ~100ms on GPU |

### 3.2 Training Data & Strategy

**Synthetic Distortion Training:**
- Source: 5,000 real motion-capture sequences from HumanML3D
- Each sequence is artificially distorted to simulate VQ artifacts
- 3 augmentations per sequence → **15,000 training pairs**

**Distortion types** (applied randomly):
| Distortion | Probability | Description |
|---|---|---|
| Foot skating injection | 40% | Random horizontal displacement (0–3cm) during ground contact |
| Foot drift | 20% | Cumulative sliding during contact phases (simulates VQ error accumulation) |
| Temporal smoothing | 20% | Moving average (window 7–15) that blurs contact transitions |
| Y-axis shift | 10% | Vertical offset (±8cm) simulating floor penetration |
| Spatial noise | 10% | Gaussian noise (σ=1–5cm) on all joints |

**Training details:**
- Loss: L1 + 0.5·L_vel + 2.0·(L_foot + L_foot_vel)
- Optimizer: Adam, lr=1e-4, StepLR scheduler (γ=0.5 every 15 epochs)
- Batch size: 32, Epochs: 50
- Best loss: **0.0114** (converged stably)

### 3.3 Selective Foot Replacement (Key Innovation)

A naive full-motion reconstruction destroys upper body quality. MotionFix uses **selective replacement**:

- **Training phase:** Network learns to reconstruct ALL 22 joints (builds global temporal understanding)
- **Inference phase:** Only foot joints (ankles 7,8 + toes 10,11) are modified, and only when skating is detected:
  - Hard gate: α = 0.5 if (height < ground + 5cm) AND (velocity > 0.03 m/frame), else α = 0
  - Modified position = (1−α) × original + α × predicted
  - All non-foot joints: **100% preserved**

This ensures upper body quality is never compromised.

### 3.4 V9 — Improvements Under Development

To address remaining limitations of V8 (foot twitching, kinematic inconsistency between foot and knee), V9 introduces:

| Improvement | Technique | Status |
|---|---|---|
| **Soft Gating** | Learned gate network (Linear→128→4→Sigmoid) × heuristic gate (height_score × vel_score) with sigmoid temperature | Implemented |
| **Temporal Smoothing** | Conv1d with Gaussian kernel (k=5, groups=4) over blend weights | Implemented |
| **Leg Chain IK** | Cosine-law knee position adjustment after foot correction (inference only) | Implemented |
| **Bone Length Loss** | L1 loss on predefined bone segment lengths (21 bone pairs) | Implemented |
| **Contact Velocity Loss** | Penalizes foot velocity during detected ground contact | Implemented |

**V9 parameters:** 19.2M (+66K from V8 for gate_network + temporal_smooth)

**Current V9 status:** Architecture and losses fully implemented and vectorized. Training produces loss of 0.114 (vs V8's 0.011), indicating the gate_network is not yet receiving proper gradient signal. This is a known issue being debugged — the training/inference path mismatch has been fixed and the model is fully vectorized, ready for re-training.

---

## 4. Results

### 4.1 MotionFix V8 on MoMask (50 Test Prompts)

| Version | Before | After | Change | Relative Improvement |
|---|---|---|---|---|
| MoMask no_IK | 13.6% | 10.8% | **−2.8pp** | −20.6% |
| MoMask IK | 12.7% | 9.9% | **−2.8pp** | −22.0% |

**Key observations:**
- MotionFix V8 provides consistent ~2.8pp reduction regardless of IK status
- IK preprocessing alone reduces skating by ~0.9pp; MotionFix adds another ~2.8pp
- Combined (IK + MotionFix): **12.7% → 9.9%** — best result
- Improvement seen across most categories; complex rotational motions remain challenging

### 4.2 Cross-Model Generalization

| Source Model | V8 Effect | Notes |
|---|---|---|
| MoMask | **Positive** (−2.8pp) | Consistent with training data characteristics |
| T2M-GPT | **Negative** (+0.9pp) | Synthetic distortions don't match T2M-GPT's artifacts |
| MDM | Not yet tested | Results pending |

**T2M-GPT negative transfer diagnosis:**
1. T2M-GPT uses different normalization (Mean/Std ~25× larger for root velocity dims)
2. T2M-GPT's quantization artifacts differ from MoMask's in spatial distribution
3. Synthetic training distortions (designed to mimic VQ errors) don't cover T2M-GPT's specific failure modes
4. **Implication:** MotionFix needs either:
   - Model-specific training data (T2M-GPT synthetic distortions), or
   - A more general distortion model that covers diverse VQ artifacts

### 4.3 Qualitative Observations

**Improvements (V8 on MoMask):**
- Simple walking/running: clear reduction in foot sliding
- Linear locomotion: feet more stable during contact phases
- Upper body: fully preserved (by design of selective replacement)

**Remaining Issues (motivating V9):**
- **Foot twitching:** abrupt transitions when hard gate switches on/off between frames
- **Knee-foot inconsistency:** foot position corrected but knee position unchanged → implausible leg configuration
- **Complex motions:** spinning, backward walking show limited improvement

---

## 5. Related Work Context

| Approach | Method | Foot Skating Handling |
|---|---|---|
| MDM (Tevet et al., 2023) | Diffusion-based generation | Inherently better (continuous denoising) |
| MoMask (Guo et al., 2024) | VQ-VAE + Masked Transformer | Optional IK post-processing |
| T2M-GPT (Zhang et al., 2023) | VQ-VAE + GPT | No built-in correction |
| **MotionFix (ours)** | Transformer post-processing | Model-agnostic foot correction |

MotionFix is positioned as a **model-agnostic post-processing module** — it can be applied to any T2M model's output without retraining the base model.

---

## 6. Current Status & Next Steps

### Completed
- [x] Baseline evaluation: MDM vs MoMask vs T2M-GPT skating analysis
- [x] MotionFix V8: Transformer architecture with selective foot replacement
- [x] Synthetic distortion training pipeline (15K pairs)
- [x] V8 testing on MoMask (50 prompts, no_IK + IK)
- [x] V8 testing on T2M-GPT (50 prompts)
- [x] V9 architecture design: soft gating + temporal smoothing + leg chain IK
- [x] V9 full vectorization (training speed ~100ms/batch vs >10s before)
- [x] Per-category skating breakdown for failure analysis

### In Progress / Blocked
- [ ] **V9 re-training** — current checkpoint (loss=0.114) is from broken training; needs re-training with vectorized model
- [ ] **MDM testing** — results.npy generated but MotionFix test not yet run
- [ ] **Semantic evaluation** — FID, R-Precision, Diversity (supervisor requested)

### Planned
- [ ] Systematic failure analysis writeup (per-category breakdown ready)
- [ ] T2M-GPT-specific training data generation (address negative transfer)
- [ ] Visual comparison videos for qualitative assessment
- [ ] Ablation study: gate_network contribution, temporal_smooth contribution, IK contribution
- [ ] Paper draft: Introduction + Method sections

---

## 7. Technical Challenges Encountered

| Challenge | Resolution |
|---|---|
| T2M-GPT showed 100% SR (all frames flagged) | Normalization mismatch: T2M-GPT uses own Mean/Std with 25× larger root velocity std → switched to MDM's t2m_mean/std for coordinate recovery |
| V9 gate_network not receiving gradient | Training path used `foot_only=False` which bypassed gating → fixed forward() to always go through soft blend path |
| V9 training extremely slow (CPU 99%, GPU 8%) | Python for-loops in _soft_blend_replace (~2350 iterations/sample) → full vectorization with `torch.repeat_interleave` and batched tensor indexing |
| Bone pair index out of bounds | Joint index 22 used but skeleton has only 22 joints (0–21) → corrected bone_pairs to valid SMPL indices |

---

## 8. Code & Reproducibility

All code at: `/home3/nxkh91/projects/motionfix/`

| File | Purpose |
|---|---|
| `motionfix_model.py` | V8 model (19.1M params, selective foot replacement) |
| `motionfix_model_v9.py` | V9 model (19.2M params, soft gating + IK) |
| `train.py` / `train_v9.py` | Training scripts |
| `test.py` / `test_v9.py` | Evaluation scripts |
| `test_momask.py` / `test_t2mgpt.py` / `test_mdm.py` | Cross-model testing |
| `prepare_data_v2.py` | Synthetic distortion training data generation |
| `dataset.py` | PyTorch Dataset loader |

**Environment:** Python 3.8, PyTorch, CUDA — conda environment `t2mgpt`

---

## 9. Key Takeaways for Discussion

1. **MotionFix V8 works** — provides ~21% relative reduction in foot skating on MoMask with zero upper body quality loss
2. **Limited cross-model generalization** — synthetic distortions don't cover all VQ artifacts; model-specific training may be needed
3. **V9 addresses known V8 weaknesses** — soft gating for smooth transitions, IK for kinematic consistency
4. **Semantic metrics (FID, R-Precision) are the critical next evaluation step** to ensure skating reduction doesn't degrade motion quality
5. **The approach is complementary to, not competing with, IK preprocessing** — they stack additively

---

*Generated with [Claude Code](https://claude.com/claude-code)*
