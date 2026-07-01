# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MotionFix is a lightweight, generator-agnostic **post-processing pipeline** that removes **foot-skating artifacts** from text-to-motion generators (MoMask, T2M-GPT, MDM) — no retraining of the generators required. It reduces both foot-skating (FSR) and jitter while keeping bone lengths rigid (no leg-tear / foot-flip) and feet in contact.

**Current method (V18 + 2-bone IK)** — a physics + learning hybrid:
```
input → De-skate (physics) → Learned smoother (learning) → 2-bone IK (physics) → output
        plant-at-mean XZ      48.8K-param 1D-CNN            hip→knee→ankle + rigid toe
        (no drift, low FSR)   (rounds boundaries, low jitter) (rigid bones)
```
- **De-skate:** plant foot at per-contact-segment mean XZ — removes skating with no integration drift (reachable target). `models/v18.py::deskate_xz` / `deskated_target`.
- **Learned smoother:** `models/v18.py::FootRefiner` + `smooth_fix`; trained by `training/v18_ik.py` with a **direct objective** (`λ_jit·acc² + λ_skate·|v|·w³ + λ_anch·anchor`). Trained on MoMask held-out; **generalizes to MDM and T2M-GPT unchanged**.
- **2-bone IK:** `models/v18_ik.py::two_bone_ik` / `apply_ik` — analytic cosine-rule solve, clamps ankle to leg reach (fixes leg-tear), rigid toe (fixes foot-flip).

**Key result (n=50 each):** MoMask FSR 14.1%→12.7% / Jitter 0.0128→0.0107; MDM 11.9%→11.2% / 0.0142→0.0115; T2M-GPT 12.0%→11.0% / 0.0139→0.0107. BoneCV unchanged, ContactAcc 100% on all. Full derivation in `docs/v18_devlog.md`; results figure `analysis/v18_ik_scale/results.png`.

**Honest finding:** with the IK constraint, FSR↔Jitter is an irreducible trade-off — the learned smoother traces the *same* frontier as a tuned Gaussian (beats Original on both axes, does not Pareto-dominate the analytical filter).

- **Author:** Xin Wan (nxkh91), Durham University
- **Server:** gpu3, NVIDIA TITAN Xp 12GB
- **Environment:** `conda activate t2mgpt` (`/home3/nxkh91/miniconda3/envs/t2mgpt/bin/python3`), PyTorch 2.1.0+cu121
- **Working dir:** `/home3/nxkh91/projects/motionfix` (run all scripts from repo root)

## Commands

### Directory Structure
```
motionfix/
├── models/          # Model definitions (v8-v13)
├── data/datasets/   # Dataset classes
├── data/prep/       # Data preparation scripts
├── data/training/   # Training data (.npy, git-ignored)
├── data/test_inputs/# Input motion data (git-ignored)
├── training/        # Training entry-point scripts
├── testing/         # Test/evaluation scripts
├── checkpoints/     # Model weights (.pth, git-ignored)
├── outputs/         # Generated outputs (git-ignored)
├── analysis/        # Analysis scripts & results
├── utils/           # Utility scripts
├── scripts/         # Shell/automation scripts
├── docs/            # Documentation & reports
└── logs/            # Log files (git-ignored)
```

### V18 + IK — current method (run from repo root)
```bash
python training/v18_ik.py            # Train adaptive smoother → checkpoints/v18_ik/best.pth (~40s)
python testing/v18_ik.py             # 5-way ablation on MoMask held-out → analysis/v18_ik_viz/
python testing/v18ik_scale.py        # Cross-generator eval (MoMask/MDM/T2M-GPT, n=50) → analysis/v18_ik_scale/
python analysis/make_results_chart.py# 4-panel results figure → analysis/v18_ik_scale/results.png
python utils/render_v18ik.py [names] # Side-by-side Original vs Learned+IK videos → outputs/videos/v18_ik/
```

### Legacy training / testing (V8 baseline)
```bash
python data/prep/v2.py               # Generate data/training/v2/ (15K pairs)
python training/v8.py                # Train V8 → checkpoints/v8/
python testing/v8.py                 # V8 evaluation on momask_50
python testing/momask.py / mdm.py / t2mgpt.py   # V8 multi-model benchmarks
```
Training scripts auto-resume from `latest.pth`. To start fresh: `rm -f checkpoints/v*/latest.pth`.

### Metrics (`utils/metrics.py`, 7 metrics)
FSR (foot-skating ratio), Jitter (foot-accel RMS), Floating, FootErr (deviation vs original),
ContactAcc, BoneCV (bone-length consistency), Penetration.

### Data
```bash
python data/prep/v2.py              # V2 format: distorted_*.npy + target_*.npy
python data/prep/v10.py             # V10 format: distorted + target + contact
python data/prep/v13.py             # V13 format: amplified noise
```
Training data is generated from HumanML3D (`/home3/nxkh91/projects/HumanML3D/HumanML3D/new_joint_vecs`). The `convert_to_joints` function uses `recover_from_ric` from `/home3/nxkh91/projects/T2M-GPT/utils/motion_process.py`.

### Git / commits
Auto-sync is **disabled** — the `.git/hooks/post-commit` auto-push hook (moved to
`post-commit.disabled`) produced the noisy `auto-sync: changes at ...` history. Prefer
manual, meaningful commits. `.gitignore` globally excludes binaries (`*.npy *.pth *.pt
*.pkl *.mp4`), `logs/`, `__pycache__/`; `analysis/**/*.png` figures ARE tracked.
The remote URL carries **no token** — push authenticates via credential prompt/helper.
(Legacy `scripts/autosync.py` + `sync.sh` remain but should stay off.)

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
| **V18 + IK** | `models/v18.py`, `models/v18_ik.py`, `training/v18_ik.py`, `testing/v18ik_scale.py` | **CURRENT / final** | De-skate → learned smoother → 2-bone IK; FSR↓ Jitter↓ bones rigid, cross-generator |
| V18 | `models/v18.py`, `training/v18.py` | Superseded | Velocity-space contact mask (broke FSR-Jitter antagonism) but `cumsum` drift (FootErr 0.39m) |
| V14–V17 | `models/v14..v17.py` | Diagnostic | Physics baselines + learned smoothers; established the FSR↔Jitter frontier |
| V8 | `models/v8.py`, `testing/v8.py` | Stable baseline | Selective foot replacement, ~2.9% FSR reduction |
| V9–V13 | `models/v9..v13.py` | Stored | Soft gating+IK / dual-head / amplified-noise experiments |

## Key Lessons Learned

1. **V8's `_selective_replace` is the critical mechanism.** Removing it (V11 original) caused near-identity output because conservative losses (L_smooth, L_upper_vel) overpowered corrective losses.

2. **Loss weight balance matters.** Corrective force (L_foot, L_recon) must exceed conservative force (L_smooth, L_upper_vel). V11's original FRDM-style losses had 2.0 conservative vs ~1.5 corrective.

3. **Training data distribution must match test data.** V10's lower-body-only tiny distortions (0.005–0.02) produced a model that over-reacts to MoMask's larger artifacts. V2's all-joint distortions with Y-shift (±5cm) create a more robust model. (V10 code in `data/prep/v10.py`)

4. **Both V8 and V11 produce large full-prediction foot errors (~0.9m L1)** on MoMask data. The selective replace mechanism succeeds *despite* these errors by only applying corrections at detected skating frames with blend_alpha=0.5. The jitter issue (7x increase) is inherent to frame-discrete blending.

5. **`autosync.log` must be excluded from file watching** — otherwise the script enters an infinite loop of detecting its own log writes.

## Dependencies

- **Python packages:** `torch`, `numpy`, `scipy` (scipy.ndimage.uniform_filter1d)
- **External path:** `/home3/nxkh91/projects/T2M-GPT/utils/motion_process.py` — `recover_from_ric()` for 263d → joint conversion
- **Data:** HumanML3D at `/home3/nxkh91/projects/HumanML3D/HumanML3D/new_joint_vecs`
- **Normalization:** `t2m_mean.npy`, `t2m_std.npy` at `/home3/nxkh91/projects/mdm/dataset/`
