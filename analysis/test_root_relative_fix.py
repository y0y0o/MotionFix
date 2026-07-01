"""
Test V8 with root-relative coordinate fix.
Compares OLD (root_relative=False) vs NEW (root_relative=True) behavior.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.v8 import MotionFixNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOTION_NAME = "p000021_rotation_person_is_walking_normally_in_a"
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
OUT_DIR = "analysis/root_relative_fix"
FOOT_JOINTS = [7, 8, 10, 11]
JOINT_NAMES = {7: "L-Ankle", 8: "R-Ankle", 10: "L-Foot", 11: "R-Foot"}
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────
model = MotionFixNetwork(blend_alpha=0.5).to(DEVICE)
ckpt = torch.load("checkpoints/v8/best.pth", map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"V8: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

# ── Load data ─────────────────────────────────────────────
original = np.load(f"{INPUT_DIR}/{MOTION_NAME}.npy").astype(np.float32)
if original.ndim == 4: original = original[0]
T = original.shape[0]
print(f"Motion: {MOTION_NAME[:60]}, T={T}")

# ── Run both versions ─────────────────────────────────────
x = torch.from_numpy(original.reshape(T, -1)).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    # OLD: no coordinate conversion
    old_full = model(x, foot_only=False, root_relative=False).squeeze(0).cpu().numpy().reshape(T, 22, 3)
    old_fixed = model(x, foot_only=True, root_relative=False).squeeze(0).cpu().numpy().reshape(T, 22, 3)

    # NEW: with root-relative conversion
    new_full = model(x, foot_only=False, root_relative=True).squeeze(0).cpu().numpy().reshape(T, 22, 3)
    new_fixed = model(x, foot_only=True, root_relative=True).squeeze(0).cpu().numpy().reshape(T, 22, 3)

print("Forward passes done.")

# ── Helpers ───────────────────────────────────────────────
def compute_contact(motion, foot_joints=(7,8), h_thresh=0.05, v_thresh=0.5):
    T = motion.shape[0]; labels = np.zeros((T, 2), dtype=np.float32)
    for i, fj in enumerate(foot_joints):
        foot_y = motion[:, fj, 1]; ground = np.percentile(foot_y, 5); thresh = ground + h_thresh
        for t in range(T):
            if foot_y[t] < thresh:
                if t > 0:
                    vel = np.linalg.norm(motion[t, fj, [0,2]] - motion[t-1, fj, [0,2]])
                    if vel < v_thresh: labels[t, i] = 1.0
                else: labels[t, i] = 1.0
    return labels

def compute_fsr(motion):
    contact = compute_contact(motion); T = motion.shape[0]
    skating = 0; contact_count = 0
    for i, fj in enumerate([7,8]):
        for t in range(1, T):
            if contact[t, i] > 0.5:
                contact_count += 1
                vel = np.linalg.norm(motion[t, fj, [0,2]] - motion[t-1, fj, [0,2]])
                if vel > 0.03: skating += 1
    if contact_count == 0: return 0.0, 0, 0
    return skating / contact_count, contact_count, skating

def compute_jitter(motion):
    foot = motion[:, FOOT_JOINTS, :]; vel = foot[1:] - foot[:-1]
    acc = vel[1:] - vel[:-1]; return float(np.sqrt((acc**2).mean()))

def compute_foot_error(a, b):
    return float(np.mean([np.linalg.norm(a[:, fj, :] - b[:, fj, :], axis=1).mean() for fj in FOOT_JOINTS]))

# ── Compute metrics ───────────────────────────────────────
print("\n" + "=" * 70)
print("METRICS COMPARISON")
print("=" * 70)
header = f"{'Version':<30} {'FSR':>8} {'Jitter':>10} {'FootErr':>10} {'FullOutErr':>12} {'OutRange X':>20}"
print(header)
print("-" * 90)

results = {}
for label, full_out, fixed in [
    ("Original", None, original),
    ("OLD (no root_relative)", old_full, old_fixed),
    ("NEW (root_relative)", new_full, new_fixed),
]:
    if label == "Original":
        fsr, c, s = compute_fsr(fixed)
        jit = compute_jitter(fixed)
        ferr = 0.0
        full_err = 0.0
        x_range = f"[{fixed[..., 0].min():.2f}, {fixed[..., 0].max():.2f}]"
    else:
        fsr, c, s = compute_fsr(fixed)
        jit = compute_jitter(fixed)
        ferr = compute_foot_error(original, fixed)
        full_err = float(np.mean([np.linalg.norm(full_out[:, fj, :] - original[:, fj, :], axis=1).mean() for fj in FOOT_JOINTS]))
        x_range = f"[{full_out[..., 0].min():.2f}, {full_out[..., 0].max():.2f}]"

    results[label] = {'FSR': fsr, 'Jitter': jit, 'FootErr': ferr, 'FullErr': full_err, 'XRange': x_range}
    print(f"{label:<30} {fsr:>7.1%} {jit:>10.4f} {ferr:>9.4f}m {full_err:>11.4f}m {x_range:>20}")

# ── FIGURE 1: Trajectory comparison ───────────────────────
fig, axes = plt.subplots(4, 3, figsize=(20, 16))
for idx, fj in enumerate(FOOT_JOINTS):
    for dim, dname in enumerate(['X (L/R)', 'Y (Height)', 'Z (F/B)']):
        ax = axes[idx][dim]
        ax.plot(original[:, fj, dim], 'k-', linewidth=1.2, alpha=0.5, label='Original')
        ax.plot(old_fixed[:, fj, dim], 'r--', linewidth=1, alpha=0.7, label='OLD (no fix)')
        ax.plot(new_fixed[:, fj, dim], 'b-', linewidth=1.5, alpha=0.9, label='NEW (root-rel)')
        ax.set_title(f'{JOINT_NAMES[fj]} {dname}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 1: Foot Trajectories — OLD vs NEW (Root-Relative Fix)', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/01_trajectories.png', dpi=150); plt.close()

# ── FIGURE 2: Velocity comparison ─────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(18, 12))
for idx, fj in enumerate(FOOT_JOINTS):
    ax = axes[idx]
    for label, data, color, lw in [
        ("Original", original, 'gray', 1.5),
        ("OLD (no fix)", old_fixed, 'r', 1.0),
        ("NEW (root-rel)", new_fixed, 'b', 1.5),
    ]:
        vel = np.linalg.norm(data[1:, fj, :] - data[:-1, fj, :], axis=1)
        ax.plot(vel, color=color, linewidth=lw, alpha=0.8, label=label)
    ax.axhline(y=0.03, color='green', linestyle='--', alpha=0.4)
    ax.set_title(f'{JOINT_NAMES[fj]} Velocity'); ax.legend(fontsize=8)
    ax.set_ylabel('m/frame'); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 2: Velocity — OLD vs NEW (Root-Relative Fix)', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/02_velocity.png', dpi=150); plt.close()

# ── FIGURE 3: Acceleration (Jitter) ───────────────────────
fig, axes = plt.subplots(4, 1, figsize=(18, 12))
for idx, fj in enumerate(FOOT_JOINTS):
    ax = axes[idx]
    for label, data, color, lw in [
        ("Original", original, 'gray', 1.5),
        ("OLD (no fix)", old_fixed, 'r', 1.0),
        ("NEW (root-rel)", new_fixed, 'b', 1.5),
    ]:
        vel = data[1:, fj, :] - data[:-1, fj, :]
        acc = np.linalg.norm(vel[1:] - vel[:-1], axis=1)
        ax.plot(acc, color=color, linewidth=lw, alpha=0.8, label=label)
    ax.set_title(f'{JOINT_NAMES[fj]} Acceleration (Jitter source)')
    ax.legend(fontsize=8); ax.set_ylabel('m/frame²'); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 3: Acceleration — OLD vs NEW (Root-Relative Fix)', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/03_acceleration.png', dpi=150); plt.close()

# ── FIGURE 4: Full output error ───────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for idx, fj in enumerate(FOOT_JOINTS):
    ax = axes[idx//2][idx%2]
    old_err = np.linalg.norm(old_full[:, fj, :] - original[:, fj, :], axis=1)
    new_err = np.linalg.norm(new_full[:, fj, :] - original[:, fj, :], axis=1)
    ax.plot(old_err, 'r-', alpha=0.7, linewidth=1, label=f'OLD (mean={old_err.mean():.3f}m)')
    ax.plot(new_err, 'b-', alpha=0.9, linewidth=1.5, label=f'NEW (mean={new_err.mean():.3f}m)')
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5)
    ax.set_title(f'{JOINT_NAMES[fj]} — Full Output vs Original Error')
    ax.set_xlabel('Frame'); ax.set_ylabel('L2 Error (m)')
    ax.legend(); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 4: Full Output Error — OLD vs NEW', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/04_full_output_error.png', dpi=150); plt.close()

# ── FIGURE 5: Per-frame displacement ──────────────────────
fig, axes = plt.subplots(4, 1, figsize=(18, 12))
for idx, fj in enumerate(FOOT_JOINTS):
    ax = axes[idx]
    old_disp = np.linalg.norm(old_fixed[:, fj, :] - original[:, fj, :], axis=1)
    new_disp = np.linalg.norm(new_fixed[:, fj, :] - original[:, fj, :], axis=1)
    ax.bar(np.arange(T)-0.2, old_disp, width=0.4, color='red', alpha=0.5, label=f'OLD ({old_disp.mean():.3f}m)')
    ax.bar(np.arange(T)+0.2, new_disp, width=0.4, color='blue', alpha=0.7, label=f'NEW ({new_disp.mean():.3f}m)')
    ax.set_title(f'{JOINT_NAMES[fj]} — |fixed - original| per frame')
    ax.set_ylabel('Displacement (m)'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5)
fig.suptitle('Figure 5: Per-frame Foot Displacement — OLD vs NEW', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/05_displacement.png', dpi=150); plt.close()

# ── Save ──────────────────────────────────────────────────
with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save fixed outputs
np.save(f"{OUT_DIR}/{MOTION_NAME}_fixed_old.npy", old_fixed)
np.save(f"{OUT_DIR}/{MOTION_NAME}_fixed_new.npy", new_fixed)

print(f"\nSaved to {OUT_DIR}/")
print("  - 01_trajectories.png, 02_velocity.png, 03_acceleration.png")
print("  - 04_full_output_error.png, 05_displacement.png")
print("  - results.json, *_fixed_old.npy, *_fixed_new.npy")
print("Done!")
