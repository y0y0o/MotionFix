"""
Diagnosis: Why training loss decreases but real performance degrades.
Tests the model on BOTH training data (V2) and MoMask test data.
"""
import torch
import numpy as np
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.v8 import MotionFixNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOOT_JOINTS = [7, 8, 10, 11]
JOINT_NAMES = {7:"L-Ankle", 8:"R-Ankle", 10:"L-Foot", 11:"R-Foot"}

# ── Load model ────────────────────────────────────────────
model = MotionFixNetwork(blend_alpha=0.5).to(DEVICE)
ckpt = torch.load("checkpoints/v8/best.pth", map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"V8: epoch {ckpt['epoch']+1}, train_loss={ckpt['loss']:.4f}")

# ── Test 1: Load a TRAINING sample ────────────────────────
# Training data: distorted (input) + target (clean)
# This is what the model was trained on
train_dir = "data/training/v2"
train_files = sorted([f for f in os.listdir(train_dir) if f.startswith("distorted_")])[:5]
print(f"\n{'='*70}")
print("TEST 1: Training data (V2 noise) — what the model was trained on")
print(f"{'='*70}")

for fname in train_files:
    idx = fname.replace("distorted_", "").replace(".npy", "")
    distorted = np.load(f"{train_dir}/distorted_{idx}.npy").astype(np.float32)
    target = np.load(f"{train_dir}/target_{idx}.npy").astype(np.float32)

    if distorted.ndim == 4: distorted = distorted[0]
    if target.ndim == 4: target = target[0]
    T = distorted.shape[0]

    # Run model
    x = torch.from_numpy(distorted.reshape(T, -1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        full_out = model(x, foot_only=False).squeeze(0).cpu().numpy().reshape(T, 22, 3)

    # Compute errors
    foot_errors = {}
    for fj in FOOT_JOINTS:
        err = np.linalg.norm(full_out[:, fj, :] - target[:, fj, :], axis=1)
        foot_errors[fj] = {'mean': float(err.mean()), 'max': float(err.max()),
                           'gt_1m': float((err > 1.0).mean())}

    # Full body error
    full_body_err = np.linalg.norm(full_out.reshape(T, -1) - target.reshape(T, -1), axis=1)
    avg_foot_err = np.mean([foot_errors[fj]['mean'] for fj in FOOT_JOINTS])

    print(f"\n  {fname} (T={T}):")
    print(f"    Full body L1 mean: {full_body_err.mean():.4f}m")
    print(f"    Avg foot error:    {avg_foot_err:.4f}m")
    for fj in FOOT_JOINTS:
        print(f"    {JOINT_NAMES[fj]:>10}: mean={foot_errors[fj]['mean']:.4f}m, "
              f"max={foot_errors[fj]['max']:.4f}m, >1m={foot_errors[fj]['gt_1m']:.1%}")

# ── Test 2: Load MoMask test data ─────────────────────────
print(f"\n{'='*70}")
print("TEST 2: MoMask data — what the model sees at inference")
print(f"{'='*70}")

momask_dir = "data/test_inputs/momask_50/momask_50_results/no_ik"
momask_files = sorted([f for f in os.listdir(momask_dir) if f.endswith('.npy')])[:5]

for fname in momask_files:
    motion = np.load(f"{momask_dir}/{fname}").astype(np.float32)
    if motion.ndim == 4: motion = motion[0]
    T = motion.shape[0]

    x = torch.from_numpy(motion.reshape(T, -1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        full_out = model(x, foot_only=False).squeeze(0).cpu().numpy().reshape(T, 22, 3)

    foot_errors = {}
    for fj in FOOT_JOINTS:
        err = np.linalg.norm(full_out[:, fj, :] - motion[:, fj, :], axis=1)
        foot_errors[fj] = {'mean': float(err.mean()), 'max': float(err.max()),
                           'gt_1m': float((err > 1.0).mean())}

    full_body_err = np.linalg.norm(full_out.reshape(T, -1) - motion.reshape(T, -1), axis=1)
    avg_foot_err = np.mean([foot_errors[fj]['mean'] for fj in FOOT_JOINTS])

    print(f"\n  {fname[:60]} (T={T}):")
    print(f"    Full body L1 mean: {full_body_err.mean():.4f}m")
    print(f"    Avg foot error:    {avg_foot_err:.4f}m")
    for fj in FOOT_JOINTS:
        print(f"    {JOINT_NAMES[fj]:>10}: mean={foot_errors[fj]['mean']:.4f}m, "
              f"max={foot_errors[fj]['max']:.4f}m, >1m={foot_errors[fj]['gt_1m']:.1%}")

# ── Test 3: Check what the model actually outputs ──────────
print(f"\n{'='*70}")
print("TEST 3: What does the model output? (coordinate range check)")
print(f"{'='*70}")

# On training data
distorted = np.load(f"{train_dir}/distorted_000000.npy").astype(np.float32)
target = np.load(f"{train_dir}/target_000000.npy").astype(np.float32)
if distorted.ndim == 4: distorted = distorted[0]; target = target[0]
T = distorted.shape[0]

x_train = torch.from_numpy(distorted.reshape(T, -1)).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    train_full = model(x_train, foot_only=False).squeeze(0).cpu().numpy().reshape(T, 22, 3)

# On MoMask
momask = np.load(f"{momask_dir}/{momask_files[0]}").astype(np.float32)
if momask.ndim == 4: momask = momask[0]
T_m = momask.shape[0]

x_momask = torch.from_numpy(momask.reshape(T_m, -1)).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    momask_full = model(x_momask, foot_only=False).squeeze(0).cpu().numpy().reshape(T_m, 22, 3)

print("\n  Coordinate statistics across ALL frames & joints:")
for label, out, inp in [("Training input", distorted, target),
                          ("Training output", train_full, target),
                          ("MoMask input", momask, momask),
                          ("MoMask output", momask_full, momask)]:
    x_range = (out[..., 0].min(), out[..., 0].max())
    y_range = (out[..., 1].min(), out[..., 1].max())
    z_range = (out[..., 2].min(), out[..., 2].max())
    print(f"  {label:<20}: X=[{x_range[0]:.2f}, {x_range[1]:.2f}], "
          f"Y=[{y_range[0]:.2f}, {y_range[1]:.2f}], Z=[{z_range[0]:.2f}, {z_range[1]:.2f}]")

# ── Test 4: Compare input-output distance ──────────────────
print(f"\n{'='*70}")
print("TEST 4: Input-Output distance (how much does model change things?)")
print(f"{'='*70}")

for label, inp, out in [("Training data", distorted, train_full),
                          ("MoMask data", momask, momask_full)]:
    diff = np.linalg.norm(out.reshape(-1, 66) - inp.reshape(-1, 66), axis=1)
    # Per-joint per-frame mean change
    joint_diff = np.linalg.norm((out - inp).reshape(-1, 3), axis=1)
    print(f"\n  {label}:")
    print(f"    Mean per-joint change: {joint_diff.mean():.4f}m")
    print(f"    Max per-joint change:  {joint_diff.max():.4f}m")
    print(f"    Per-joint std:          {joint_diff.std():.4f}m")
    # Foot-specific
    foot_diff = np.linalg.norm((out[:, FOOT_JOINTS, :] - inp[:, FOOT_JOINTS, :]).reshape(-1, 3), axis=1)
    print(f"    Foot mean change:      {foot_diff.mean():.4f}m")
    print(f"    Foot max change:       {foot_diff.max():.4f}m")

print("\nDone!")
