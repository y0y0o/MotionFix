# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MotionFix is a lightweight post-processing Transformer that corrects **foot skating artifacts** in VQ-based motion generation models (MoMask, T2M-GPT). It operates as a plug-and-play module — no retraining of the generators required.

- **Author:** Xin Wan (nxkh91), Durham University
- **Server:** gpu3, NVIDIA TITAN Xp 12GB
- **Environment:** `/home3/nxkh91/miniconda3/envs/t2mgpt/bin/python3`, PyTorch 2.1.0+cu121

## Commands

### Training
```bash
# V8 (stable baseline) — run from repo root
python prepare_data_v2.py          # Generate training_data_v2/ (15K pairs, 45K files)
python train.py                     # Train V8 → checkpoints_v8/

# V11 (experimental) — run from motionfix_v11/
cd motionfix_v11
python ../prepare_data.py           # Or use existing training_data_v10/
python train.py                     # Train V11 → checkpoints_v11/
```

All training scripts automatically resume from `latest.pth` if it exists. To start fresh: `rm -f checkpoints_v*/latest.pth`.

### Testing / Evaluation
```bash
python test.py                      # V8 evaluation on momask_results/
cd motionfix_v11 && python test.py  # V11 evaluation on momask_50_results/no_ik/
```

### Data
```bash
python prepare_data_v2.py           # V2 format: distorted_*.npy + target_*.npy
python motionfix_v10/prepare_data.py # V10 format: distorted_*.npy + target_*.npy + contact_*.npy
```
Training data is generated from HumanML3D (`/home3/nxkh91/HumanML3D/HumanML3D/new_joint_vecs`). The `convert_to_joints` function uses `recover_from_ric` from `/home3/nxkh91/Project/T2M-GPT/utils/motion_process.py`.

### Auto-sync
```bash
./sync.sh          # Start autosync.py watchdog (auto commit+push)
pkill -f autosync  # Stop it
```
`autosync.py` watches for file changes and commits after a 120s quiet period. It ignores `training_data*`, `checkpoints*`, `fixed_outputs*`, `*.npy`, `*.log`, and `.git`.

## Architecture

### Data Format
All models operate on **22-joint, 3D world-coordinate** joint positions (HumanML3D skeleton). A motion of shape `(T, 22, 3)` is flattened to `(T, 66)` as model input.

**Joint indices reference:**
| Index | Name | Index | Name |
|-------|------|-------|------|
| 0 | Pelvis | 7 | Left Ankle |
| 1 | Left Hip | 8 | Right Ankle |
| 2 | Right Hip | 9 | Spine2 |
| 3 | Spine1 | 10 | Left Foot |
| 4 | Left Knee | 11 | Right Foot |
| 5 | Right Knee | 12 | Neck |
| 6 | Spine3 | 13–21 | Shoulders/arms/head |

Foot joints for loss/selective-replace: **7, 8 (ankles), 10, 11 (feet)**.

### Core Architecture (V8 — Stable Baseline)
```
(T, 66) → Linear(66→512) → PositionalEncoding(sin) → TransformerEncoder ×6
        (d_model=512, nhead=8, FFN=2048, dropout=0.1)
        → Linear(512→256) → ReLU → Linear(256→66) → (T, 66)
```
- **19.1M** parameters
- Training: `foot_only=False` — model outputs all joints (learns full context)
- Inference: `foot_only=True` — `_selective_replace()` only blends foot joints at detected skating frames with `blend_alpha=0.5`

### Loss Function (V8, proven)
```
L_total = L1(pred, target)                          # Full body reconstruction
        + 0.5 * L1(vel_pred, vel_target)            # All-joint velocity
        + 2.0 * L1(foot_pred, foot_target)          # Foot position (direct)
        + 2.0 * L1(foot_vel_pred, foot_vel_target)  # Foot velocity (direct)
```

### _selective_replace (V8's Key Mechanism)
For each foot joint, frame-by-frame:
1. If foot height < ground + 5cm AND horizontal velocity > 0.03 m/frame → skating detected
2. Blend: `output = (1-0.5)*original + 0.5*predicted`
3. All non-foot joints and non-skating frames: **untouched** (hard constraint via `output = original.clone()`)

This is the mechanism that makes V8 work — it only modifies foot joints at skating frames, creating strong inductive bias.

## Version Status

| Version | Location | Status | Key Trait |
|---------|----------|--------|-----------|
| **V8** | `motionfix_model.py`, `train.py`, `test.py` | **Stable** | Selective foot replacement, proven ~2.9% FSR reduction on momask_results |
| V9 | `motionfix_model_v9.py`, `train_v9.py` | Current best | Soft gating + IK (separate architecture) |
| V10 | `motionfix_v10/` | Stored | Lower-body-only distortions, contact labels |
| V11 | `motionfix_v11/` | WIP | Currently fixed to match V8 loss + selective replace; trained on V2 data |

## Key Lessons Learned

1. **V8's `_selective_replace` is the critical mechanism.** Removing it (V11 original) caused near-identity output because conservative losses (L_smooth, L_upper_vel) overpowered corrective losses.

2. **Loss weight balance matters.** Corrective force (L_foot, L_recon) must exceed conservative force (L_smooth, L_upper_vel). V11's original FRDM-style losses had 2.0 conservative vs ~1.5 corrective.

3. **Training data distribution must match test data.** V10's lower-body-only tiny distortions (0.005–0.02) produced a model that over-reacts to MoMask's larger artifacts. V2's all-joint distortions with Y-shift (±5cm) create a more robust model.

4. **Both V8 and V11 produce large full-prediction foot errors (~0.9m L1)** on MoMask data. The selective replace mechanism succeeds *despite* these errors by only applying corrections at detected skating frames with blend_alpha=0.5. The jitter issue (7x increase) is inherent to frame-discrete blending.

5. **`autosync.log` must be excluded from file watching** — otherwise the script enters an infinite loop of detecting its own log writes.

## Dependencies

- **Python packages:** `torch`, `numpy`, `scipy` (scipy.ndimage.uniform_filter1d)
- **External path:** `/home3/nxkh91/Project/T2M-GPT/utils/motion_process.py` — `recover_from_ric()` for 263d → joint conversion
- **Data:** HumanML3D at `/home3/nxkh91/HumanML3D/HumanML3D/new_joint_vecs`
- **Normalization:** `t2m_mean.npy`, `t2m_std.npy` at `/home3/nxkh91/motion-diffusion-model-main/dataset/`
