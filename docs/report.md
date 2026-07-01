# MotionFix: Foot Skating Correction for VQ-based Motion Generation

**Student:** Xin Wan  
**Date:** June 2026  
**Supervisor Meeting Report**

---

## 1. Research Problem

VQ-based text-to-motion models (MoMask, T2M-GPT) generate motions 15-20× faster than diffusion models (MDM), but suffer from **foot skating artifacts** — feet slide along the ground when they should be stationary during contact.

## 2. Baseline Evaluation

Evaluated MDM (diffusion) and MoMask (VQ) on 10 test prompts with varying complexity. Skating ratio (SR) measures the percentage of ground-contact frames where the foot is sliding (velocity > 0.03 m/frame).

### MDM vs MoMask Skating Comparison

| Test Prompt | MDM | MoMask | Difference |
|---|---|---|---|
| Spins around multiple times | 28.1% | 42.0% | +13.9% |
| Walks backward in a circle | 24.9% | 49.9% | +25.0% |
| Zigzag pattern | 20.5% | 16.2% | -4.4% |
| Backward curved path | 15.1% | 14.7% | -0.5% |
| Forward slowly and turns | 9.4% | 14.2% | +4.8% |
| Quickly changes direction | 7.1% | 11.7% | +4.6% |
| Jumps / Squats / Stands still | 0.0% | 0.0-0.9% | ~0% |
| **Average** | **11.7%** | **16.8%** | **+5.1%** |

**Key finding:** MoMask shows 42% higher average skating than MDM, with worst-case 49.9% on backward circular motion (2× worse than MDM).

## 3. MotionFix Method

### 3.1 Architecture

MotionFix is a lightweight Transformer Encoder post-processing network. It takes the full motion sequence as input, learns global temporal context, and outputs corrected foot joint positions.

<!-- GIF: Architecture diagram -->
<!-- ![Architecture](figures/architecture.gif) -->

**Network specification:**
- Input: motion sequence (T, 22 joints, 3 coordinates) → flattened to (T, 66)
- Transformer Encoder: 6 layers, 512 dim, 8 heads
- Parameters: 19M (~76MB)
- Inference time: ~0.1s additional overhead

### 3.2 Training Data

Training pairs are generated from HumanML3D (8,177 real motion-capture sequences):

1. Take a perfect motion-capture sequence (no skating)
2. Apply synthetic distortions to simulate VQ model artifacts:
   - **Foot skating injection**: random horizontal displacement during ground contact
   - **Foot drift**: cumulative sliding during contact phases
   - **Temporal smoothing**: moving average (window 7-15) that blurs contact transitions
   - **Y-axis shift / spatial noise**: simulate penetration and jitter
3. Train the network to reconstruct the original clean motion

Total: 5,000 motions × 3 distortions = 15,000 training pairs.

### 3.3 Selective Foot Replacement (V8)

A naive full-motion reconstruction destroys upper body quality (V3), while naive foot-only prediction causes mean regression (V5/V7). MotionFix V8 uses **selective replacement**:

- **Training**: reconstruct all 22 joints (full global context learning)
- **Inference**: 
  - Detect ground-contact frames per foot (height < ground + 5cm)
  - For contact frames with velocity > 0.03 m/frame (**skating detected**): blend network prediction with original (α = 0.5)
  - All other frames: keep original motion unchanged

This ensures upper body motion is fully preserved, and feet are only corrected when skating is detected.

### 3.4 Loss Function

$$L = L_{pos} + 0.5 \cdot L_{vel} + 2.0 \cdot (L_{foot} + L_{foot\_vel})$$

- $L_{pos}$: L1 loss on all joint positions
- $L_{vel}$: L1 loss on frame-to-frame velocity (smoothness)
- $L_{foot}$: additional L1 on foot joints (7, 8, 10, 11) with 2× weight
- $L_{foot\_vel}$: additional velocity loss on foot joints

## 4. Results

### 4.1 MotionFix V8 on MoMask Outputs

| Test Prompt | Before | After | Change |
|---|---|---|---|
| Walks backward in a circle | 49.9% | 32.7% | **-17.2%** |
| Forward slowly and turns | 14.2% | 3.6% | **-10.6%** |
| Forward, turns left, backward | 18.9% | 12.0% | **-6.9%** |
| Quickly changes direction | 11.7% | 5.9% | **-5.9%** |
| Spins around multiple times | 42.0% | 39.5% | -2.5% |
| Backward curved path | 14.7% | 12.5% | -2.1% |
| Zigzag pattern | 16.2% | 15.4% | -0.8% |
| Squats down | 0.9% | 1.8% | +0.9% |
| Jumps / Stands still | 0.0% | 0.0% | 0.0% |
| **Average** | **16.8%** | **12.3%** | **-4.5%** |

Average skating reduced from 16.8% → 12.3% (26.8% relative improvement).

### 4.2 Visual Comparison

<!-- Insert GIF/video comparisons here -->

**Backward circle (SR: 49.9% → 32.7%):**

| Before (MoMask) | After (MotionFix) |
|---|---|
| ![before](figures/backward_circle_before.gif) | ![after](figures/backward_circle_fixed.gif) |

**Forward slowly and turns (SR: 14.2% → 3.6%):**

| Before (MoMask) | After (MotionFix) |
|---|---|
| ![before](figures/forward_slow_before.gif) | ![after](figures/forward_slow_fixed.gif) |

### 4.3 Three-way Comparison

| Method | Avg Skating | Inference Speed | Type |
|---|---|---|---|
| MoMask (baseline) | 16.8% | 0.9s | VQ-based |
| **MoMask + MotionFix** | **12.3%** | **~1.0s** | **VQ + post-processing** |
| MDM (reference) | 11.7% | 18s | Diffusion-based |

MotionFix brings MoMask's quality close to MDM (12.3% vs 11.7%) while maintaining 18× faster inference.

## 5. Development Iterations

| Version | Approach | Result | Issue |
|---|---|---|---|
| V1-V2 | Residual connection | No change (0%) | Correction → 0 |
| V3 | Full reconstruction | SR: 16.8%→7.8% | Upper body collapsed |
| V4 | Foot residual | No change (0%) | Correction → 0 |
| V5 | Foot direct prediction | Feet stuck to ground | Mean regression |
| V7 | V3 train + foot inference | Feet stuck to ground | Mean regression |
| **V8** | **Selective replacement** | **SR: 16.8%→12.3%** | **Minor twitching** |

## 6. Current Limitations

1. **Minor twitching** at contact/air transition boundaries due to blending
2. **Spinning motion** shows limited improvement (42.0% → 39.5%), suggesting synthetic distortions do not fully capture rotational skating patterns
3. Blend factor α = 0.5 is global; per-frame adaptive blending may improve results

## 7. Next Steps

1. **Improve blending**: learnable per-frame α using a gating network
2. **Better synthetic distortions**: analyse real MoMask skating patterns and replicate them more faithfully
3. **Evaluate on T2M-GPT**: verify generalisation to other VQ architectures
4. **Compare with IK baseline**: quantitative comparison with MoMask's built-in inverse kinematics
5. **Perceptual quality metrics**: FID, diversity scores alongside skating ratio
