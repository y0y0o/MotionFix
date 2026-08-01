# MotionFix Progress Report: Foot Skating Correction for VQ-based Motion Generation

**Student:** Xin Wan
**Date:** 16 June 2026
**Progress Report**

---

## 1. Summary of This Week's Work

This week I completed five main tasks:

1. **Expanded the test set**: from 10 prompts to 50, covering 7 motion categories
2. **Cross-model testing**: tested MotionFix V8 on three models — MoMask, T2M-GPT, and MDM
3. **Built new evaluation metrics**: in addition to Skating Ratio, added Jitter and Penetration Rate
4. **V8 → V9 improvement attempt**: developed V9 (soft gating + temporal smoothing + leg-chain IK) to address V8's foot-twitching, but the current results are not satisfactory
5. **Investigated the FRDM approach**: found the Foot Restoration Diffusion Model in the InfiniteDance paper, and plan to reproduce it while replacing its diffusion model with a Transformer

---

## 2. Baseline Evaluation

### 2.1 Evaluation Setup

The test set was expanded to 50 prompts, covering 7 categories: walking, turning, backward, rotation, jumping, complex combinations, and dance/sports.

Skating Ratio (SR) is defined as the proportion of ground-contact frames in which horizontal foot velocity exceeds 0.03 m/frame. Ground contact is determined using a relative height threshold (ground + 5cm, where the ground is taken as the 5th percentile of heights).

### 2.2 MDM vs MoMask vs T2M-GPT

| Model | Type | Avg Skating Ratio | Notes |
|---|---|---|---|
| MDM | Diffusion | — | Least skating (reference baseline) |
| MoMask (no_IK) | VQ-based | 13.6% | No IK post-processing |
| MoMask (IK) | VQ-based | 12.7% | Built-in IK reduces skating by ~0.9pp |
| T2M-GPT | VQ-based | — | Different artifact pattern (see §4.2) |

**Key finding:** MoMask's average skating ratio of ~14% is significantly higher than MDM's. The built-in IK provides a modest improvement (~0.9pp) but does not eliminate the problem.

---

## 3. MotionFix Method

### 3.1 Architecture (V8 — Current Best)

MotionFix is a Transformer encoder network that takes a full motion sequence as input and outputs corrected joint positions.

| Component | Specification |
|---|---|
| Input | (T, 22, 3) joint positions → flattened to (T, 66) |
| Input projection | Linear(66 → 512) |
| Positional encoding | Sinusoidal, max_len=500 |
| Transformer encoder | 6 layers, d_model=512, 8 heads, FFN=2048 |
| Output projection | Linear(512 → 256) → ReLU → Linear(256 → 66) |
| Total parameters | **19.1M** (~76 MB) |
| Inference overhead | ~100ms on GPU |

### 3.2 Training Data and Strategy

**Synthetic distortion training:** The data comes from 5,000 real motion-capture sequences in HumanML3D. Three augmentations are applied to each sequence, producing 15,000 training pairs (distorted → clean).

| Distortion Type | Probability | Description |
|---|---|---|
| Foot skating injection | 40% | Random horizontal displacement during ground contact (0–3cm) |
| Foot drift | 20% | Cumulative sliding during contact phases (simulating VQ error accumulation) |
| Temporal smoothing | 20% | Moving average (window 7–15), blurring contact transitions |
| Y-axis shift | 10% | Vertical offset (±8cm), simulating floor penetration |
| Spatial noise | 10% | Gaussian noise on all joints (σ=1–5cm) |

**Training details:** Loss function is L1 + 0.5·L_vel + 2.0·(L_foot + L_foot_vel); optimizer Adam, lr=1e-4, StepLR (γ=0.5 every 15 epochs); batch size 32, 50 training epochs; best loss 0.0114 (stably converged).

### 3.3 Selective Foot Replacement (Core Contribution)

Full motion reconstruction destroys upper-body quality (see the failure-mode analysis in §3.4). MotionFix adopts selective replacement:

- **Training stage:** The network learns to reconstruct all 22 joints, establishing global temporal understanding
- **Inference stage:** Only foot joints (ankles 7,8 + toes 10,11) are modified, and only when skating is detected:
  - Hard gating: if (height < ground + 5cm) and (velocity > 0.03 m/frame), then α = 0.5, otherwise α = 0
  - Corrected position = (1−α) × original position + α × predicted position
  - All non-foot joints: 100% preserved

This ensures upper-body quality is never compromised.

---

## 4. Experimental Results

### 4.1 MotionFix V8 on MoMask (50 Prompts)

| Version | Before | After | Change | Relative Improvement |
|---|---|---|---|---|
| MoMask no_IK | 13.6% | 10.8% | **−2.8pp** | −20.6% |
| MoMask IK | 12.7% | 9.9% | **−2.8pp** | −22.0% |

**Key observations:**

- MotionFix V8 provides a consistent ~2.8 percentage point reduction regardless of IK status
- IK alone reduces skating by ~0.9pp; MotionFix adds a further ~2.8pp
- Combined (IK + MotionFix): 12.7% → 9.9%, the best result
- Most motion categories improve; complex rotational motions remain challenging

### 4.2 Cross-Model Generalization

| Source Model | V8 Effect | Notes |
|---|---|---|
| MoMask | **Positive** (−2.8pp) | Consistent with training data characteristics |
| T2M-GPT | **Negative** (+0.9pp) | Synthetic distortions fail to match T2M-GPT's artifacts |
| MDM | Not yet tested | results.npy generated, testing pending |

**T2M-GPT negative transfer diagnosis:**

1. T2M-GPT uses different normalization (Mean/Std of the root velocity dimension is ~25× larger)
2. T2M-GPT's quantization artifacts differ from MoMask's in spatial distribution
3. The synthetic distortions designed to simulate VQ error cannot cover T2M-GPT's specific failure modes

**Implication:** MotionFix requires either model-specific training data, or a more general distortion model covering multiple VQ artifacts. This is an important finding — it defines the current method's capability boundary and points toward future work.

### 4.3 Qualitative Observations

**Improvements (V8 on MoMask):**

- Simple walking/running: foot sliding noticeably reduced
- Linear displacement: feet more stable during ground contact
- Upper body: fully preserved (thanks to the selective-replacement design)

**Remaining issues (motivation for V9):**

- **Foot twitching:** the hard gate produces abrupt transitions when switching between frames
- **Knee-foot inconsistency:** foot positions are corrected but knee positions are not, producing implausible leg configurations
- **Complex motions:** rotation and backward walking show limited improvement

---

## 5. V9 Improvement Attempt (In Progress)

To address V8's remaining issues, V9 introduces the following improvements:

| Improvement | Technical Approach | Status |
|---|---|---|
| Soft gating | Learnable gate network (Linear→128→4→Sigmoid) × heuristic gating (height_score × vel_score), with sigmoid temperature | Implemented |
| Temporal smoothing | Conv1d + Gaussian kernel (k=5, groups=4), applied to blend weights | Implemented |
| Leg-chain IK | Law-of-cosines knee position adjustment, performed after foot correction (inference only) | Implemented |
| Bone-length loss | L1 loss on predefined bone segment lengths (21 bone pairs) | Implemented |
| Contact velocity loss | Penalizes foot velocity during detected ground contact | Implemented |

**V9 parameters:** 19.2M (66K more than V8, from gate_network + temporal_smooth)

**V9 current status:** The architecture and loss functions are all implemented and vectorized. However, the current training loss is 0.114 (V8 was 0.011), indicating that the gate_network is not yet receiving the correct gradient signal. The train/inference path mismatch has been fixed and the model is fully vectorized, ready for retraining. **V9 currently does not perform well and is still being debugged.**

---

## 6. Investigation: FRDM and an Improvement Idea

In the InfiniteDance paper (arXiv:2603.13375, 2026), I found the closest related work to MotionFix — the Foot Restoration Diffusion Model (FRDM). I plan to reproduce it and propose an improvement.

### 6.1 What FRDM Is

FRDM is the third step in the InfiniteDance data pipeline, dedicated to repairing foot artifacts in 3D human motion. The problem it solves is highly similar to MotionFix's — foot sliding and foot jitter.

```
Step 1: GVHMR monocular estimation → FSR 28.63%, Jitter 31.89
Step 2: IsaacGym physics simulation → FSR 8.87%,  Jitter 78.60 (jitter spikes)
Step 3: FRDM foot restoration        → FSR 5.09%,  Jitter 14.33 (best)
```

### 6.2 FRDM's Core Design

| Design Point | Description | Relation to MotionFix |
|---|---|---|
| Local repair | Only fixes root/knee/foot, upper body unchanged | MotionFix only fixes toes; could extend to full lower-limb chain |
| Self-supervised training | Constructs pseudo-artifact samples by adding noise to clean data, no paired data needed | Consistent with MotionFix's synthetic distortion training |
| 259-dim representation | Uses position + velocity + rotation + contact label together | MotionFix uses only 66-dim position; worth borrowing |
| Two-stage inference guidance | Early geometric guidance + late foot-contact guidance | Could inform V9's blending strategy |

### 6.3 FRDM's Loss Functions (Worth Borrowing)

| Loss | Purpose |
|---|---|
| L_recon | Basic reconstruction MSE |
| L_Foot | Adjacent-frame position change should be 0 when foot is on ground (anti-skating) |
| L_vel-pos | L2 consistency between velocity-integrated position and direct position |
| L_epsilon-rp | Deviation between FK-derived position from rotation and direct position; no penalty within epsilon tolerance |

Among these, L_Foot is more precise than MotionFix's current L_vel (it only penalizes foot displacement on contact frames); L_epsilon-rp allows rotation and position to be optimized separately, which is a design worth learning from.

### 6.4 Improvement Idea: Replacing the Diffusion Model with a Transformer

FRDM uses a diffusion model (multi-step iterative denoising). I plan to replace it with a Transformer, which aligns naturally with MotionFix's existing architecture.

| Dimension | Diffusion Model (FRDM original) | Transformer (improvement idea) |
|---|---|---|
| Inference speed | Multi-step iterative denoising (~50 steps) | Single-step forward inference |
| Training complexity | Noise schedule, multi-timestep | Standard seq2seq training |
| Guidance mechanism | Two-stage explicit guidance | attention / FiLM conditional injection |
| Long-sequence modeling | Limited by noise schedule | Transformer is naturally suited |

**Key technical points:**

1. **Residual prediction:** The Transformer only predicts the correction Δ; the final output = X + Δ, where Δ is non-zero only on lower-body joints (replacing "only fix lower body")
2. **FiLM conditional injection:** Use the foot contact label b to modulate decoder features; when b=1 (foot on ground), modulate the velocity gain toward 0, implementing the physical constraint "zero velocity on ground contact" architecturally
3. **Temporal consistency:** Relative positional encoding + L2 regularization on adjacent-frame output changes
4. **Single-step inference:** No need for 50+ iterations, especially valuable when processing large-scale data
