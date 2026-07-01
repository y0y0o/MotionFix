"""
V8 vs V13 deep-dive comparison on p000021.
Both versions show foot flashing — this analysis explains why V8's is less severe.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ──────────────────────────────────────────────────
MOTION_NAME = "p000021_rotation_person_is_walking_normally_in_a"
INPUT_PATH  = "data/test_inputs/momask_50/momask_50_results/no_ik"
V8_FIXED    = "outputs/fixed/v8"
V13_FIXED   = "outputs/fixed/v13_momask"
OUT_DIR     = "analysis/v8_vs_v13"
FOOT_JOINTS = [7, 8, 10, 11]
JOINT_NAMES = {7:"L-Ankle", 8:"R-Ankle", 10:"L-Foot", 11:"R-Foot"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────
original = np.load(f"{INPUT_PATH}/{MOTION_NAME}.npy").astype(np.float32)
if original.ndim == 4: original = original[0]
T = original.shape[0]

v8_fixed = np.load(f"{V8_FIXED}/{MOTION_NAME}.npy").astype(np.float32)
v13_fixed = np.load(f"{V13_FIXED}/{MOTION_NAME}_fixed.npy").astype(np.float32)
print(f"Original: {original.shape}, V8 fixed: {v8_fixed.shape}, V13 fixed: {v13_fixed.shape}")

# ── Load V8 model & run forward ────────────────────────────
from models.v8 import MotionFixNetwork
model_v8 = MotionFixNetwork(blend_alpha=0.5).to(DEVICE)
ckpt = torch.load("checkpoints/v8/best.pth", map_location=DEVICE)
model_v8.load_state_dict(ckpt['model_state_dict'])
model_v8.eval()
print(f"V8 loaded: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

x = torch.from_numpy(original.reshape(T, -1)).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    v8_full = model_v8(x, foot_only=False).squeeze(0).cpu().numpy().reshape(T, 22, 3)
    v8_foot = model_v8(x, foot_only=True).squeeze(0).cpu().numpy().reshape(T, 22, 3)
print(f"V8 forward done: full_output shape {v8_full.shape}")

# ── Metrics ────────────────────────────────────────────────
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
    acc = vel[1:] - vel[:-1]; return np.sqrt((acc**2).mean())

def compute_foot_error(a, b):
    return np.mean([np.linalg.norm(a[:, fj, :] - b[:, fj, :], axis=1).mean() for fj in FOOT_JOINTS])

def detect_modification(orig, fixed, fj, eps=1e-4):
    return np.linalg.norm(fixed[:, fj, :] - orig[:, fj, :], axis=1) > eps

def find_groups(modified):
    groups = []; in_group = False; start = 0
    for t in range(len(modified)):
        if modified[t] and not in_group: start = t; in_group = True
        elif not modified[t] and in_group:
            groups.append((start, t-1, t-start)); in_group = False
    if in_group: groups.append((start, len(modified)-1, len(modified)-start))
    return groups

# ── Compute metrics ────────────────────────────────────────
v8_full_errors = {fj: np.linalg.norm(v8_full[:, fj, :] - original[:, fj, :], axis=1) for fj in FOOT_JOINTS}

mod = {}
for label, fixed in [("V8", v8_fixed), ("V13", v13_fixed)]:
    mod[label] = {}
    for fj in FOOT_JOINTS:
        modded = detect_modification(original, fixed, fj)
        mod[label][fj] = {'modified': modded, 'groups': find_groups(modded),
                          'n_modified': modded.sum(), 'n_total': T}

metrics = {}
for label, data in [("Original", original), ("V8-fixed", v8_fixed), ("V8-foot_out", v8_foot),
                     ("V13-fixed", v13_fixed)]:
    fsr, c, s = compute_fsr(data); jit = compute_jitter(data)
    ferr = compute_foot_error(original, data)
    metrics[label] = {'FSR': fsr, 'Jitter': jit, 'FootErr': ferr}
    print(f"  {label:<18}: FSR={fsr:.1%} ({s}/{c}), Jitter={jit:.4f}, FootErr={ferr:.4f}m")

# ── FIGURE 1: V8 full_output error ─────────────────────────
print("Generating Figure 1: V8 full_output error...")
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for idx, fj in enumerate(FOOT_JOINTS):
    ax = axes[idx//2][idx%2]
    ax.plot(v8_full_errors[fj], 'b-', linewidth=1, alpha=0.8)
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='5cm')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='30cm')
    ax.set_title(f'{JOINT_NAMES[fj]} (joint {fj}) — V8 Full Output Error', fontsize=12)
    ax.set_xlabel('Frame'); ax.set_ylabel('L2 Error (m)'); ax.legend(); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 1: V8 Model Raw Output Error (full_output vs original)', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/01_v8_full_output_error.png', dpi=150); plt.close()
print("  → 01_v8_full_output_error.png")

# ── FIGURE 2: Foot trajectories — original vs V8-fixed vs V13-fixed ──
print("Generating Figure 2: Trajectory comparison...")
fig, axes = plt.subplots(4, 3, figsize=(20, 16))
for idx, fj in enumerate(FOOT_JOINTS):
    for dim, dname in enumerate(['X (L/R)', 'Y (Height)', 'Z (F/B)']):
        ax = axes[idx][dim]
        ax.plot(original[:, fj, dim], 'k-', linewidth=1.2, alpha=0.6, label='Original')
        ax.plot(v8_fixed[:, fj, dim], 'b-', linewidth=1, alpha=0.8, label='V8-fixed')
        ax.plot(v13_fixed[:, fj, dim], 'r-', linewidth=1, alpha=0.8, label='V13-fixed')
        ax.set_title(f'{JOINT_NAMES[fj]} {dname}'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 2: Foot Trajectories — Original vs V8 vs V13', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/02_trajectory_comparison.png', dpi=150); plt.close()
print("  → 02_trajectory_comparison.png")

# ── FIGURE 3: Velocity comparison ──────────────────────────
print("Generating Figure 3: Velocity comparison...")
fig, axes = plt.subplots(4, 1, figsize=(18, 12))
for idx, fj in enumerate(FOOT_JOINTS):
    ax = axes[idx]
    for label, data, color, lw in [("Original", original, 'gray', 1.5),
                                    ("V8-fixed", v8_fixed, 'b', 1.0),
                                    ("V13-fixed", v13_fixed, 'r', 1.0)]:
        vel = np.linalg.norm(data[1:, fj, :] - data[:-1, fj, :], axis=1)
        ax.plot(vel, color=color, linewidth=lw, alpha=0.8, label=label)
    ax.axhline(y=0.03, color='green', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_title(f'{JOINT_NAMES[fj]} Velocity'); ax.legend(fontsize=8)
    ax.set_ylabel('m/frame'); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 3: Foot Velocity — Original vs V8 vs V13', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/03_velocity_comparison.png', dpi=150); plt.close()
print("  → 03_velocity_comparison.png")

# ── FIGURE 4: Acceleration (Jitter source) ─────────────────
print("Generating Figure 4: Acceleration...")
fig, axes = plt.subplots(4, 1, figsize=(18, 12))
for idx, fj in enumerate(FOOT_JOINTS):
    ax = axes[idx]
    for label, data, color, lw in [("Original", original, 'gray', 1.5),
                                    ("V8-fixed", v8_fixed, 'b', 1.0),
                                    ("V13-fixed", v13_fixed, 'r', 1.0)]:
        vel = data[1:, fj, :] - data[:-1, fj, :]
        acc = np.linalg.norm(vel[1:] - vel[:-1], axis=1)
        ax.plot(acc, color=color, linewidth=lw, alpha=0.8, label=label)
    ax.set_title(f'{JOINT_NAMES[fj]} Acceleration (Jitter component)')
    ax.legend(fontsize=8); ax.set_ylabel('m/frame²'); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 4: Foot Acceleration — Jitter Source Comparison', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/04_acceleration_comparison.png', dpi=150); plt.close()
print("  → 04_acceleration_comparison.png")

# ── FIGURE 5: Modification heatmap ─────────────────────────
print("Generating Figure 5: Modification heatmap...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
for ax, label in [(ax1, "V8"), (ax2, "V13")]:
    heatmap = np.zeros((4, T))
    for idx, fj in enumerate(FOOT_JOINTS):
        heatmap[idx] = mod[label][fj]['modified'].astype(float)
    im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1, extent=[0, T, 0, 4])
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels([JOINT_NAMES[j] for j in FOOT_JOINTS])
    ax.set_xlabel('Frame')
    n_groups = [len(mod[label][fj]['groups']) for fj in FOOT_JOINTS]
    pct = [mod[label][fj]['n_modified']/T*100 for fj in FOOT_JOINTS]
    ax.set_title(f'{label}: {np.mean(n_groups):.0f} groups avg, {np.mean(pct):.1f}% frames modified')
    for idx in range(4):
        ax.text(T+2, 3.5-idx, f'{pct[idx]:.1f}% ({n_groups[idx]}grp)', fontsize=7, va='center')
fig.suptitle('Figure 5: Modification Heatmap — V8 vs V13', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/05_modification_heatmap.png', dpi=150); plt.close()
print("  → 05_modification_heatmap.png")

# ── FIGURE 6: Per-frame displacement ───────────────────────
print("Generating Figure 6: Per-frame displacement...")
fig, axes = plt.subplots(4, 1, figsize=(18, 12))
for idx, fj in enumerate(FOOT_JOINTS):
    ax = axes[idx]
    v8_disp = np.linalg.norm(v8_fixed[:, fj, :] - original[:, fj, :], axis=1)
    v13_disp = np.linalg.norm(v13_fixed[:, fj, :] - original[:, fj, :], axis=1)
    ax.bar(np.arange(T)-0.2, v8_disp, width=0.4, color='blue', alpha=0.6, label='V8')
    ax.bar(np.arange(T)+0.2, v13_disp, width=0.4, color='red', alpha=0.6, label='V13')
    ax.set_title(f'{JOINT_NAMES[fj]} — Per-frame |fixed - original| (V8={v8_disp.mean():.3f}m, V13={v13_disp.mean():.3f}m)')
    ax.set_ylabel('Displacement (m)'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, linewidth=1, label='5cm')
fig.suptitle('Figure 6: Per-Frame Foot Displacement — V8 (blue) vs V13 (red)', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/06_displacement_comparison.png', dpi=150); plt.close()
print("  → 06_displacement_comparison.png")

# ── FIGURE 7: V8 modification magnitude histogram ──────────
print("Generating Figure 7: Modification magnitude distributions...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for idx, fj in enumerate(FOOT_JOINTS):
    ax = axes[idx//2][idx%2]
    v8_disp = np.linalg.norm(v8_fixed[:, fj, :] - original[:, fj, :], axis=1)
    v13_disp = np.linalg.norm(v13_fixed[:, fj, :] - original[:, fj, :], axis=1)
    v8_only_mod = v8_disp[mod['V8'][fj]['modified']]
    v13_only_mod = v13_disp[mod['V13'][fj]['modified']]
    ax.hist(v8_only_mod, bins=30, alpha=0.5, color='blue', label=f'V8 (n={len(v8_only_mod)})', density=True)
    ax.hist(v13_only_mod, bins=30, alpha=0.5, color='red', label=f'V13 (n={len(v13_only_mod)})', density=True)
    ax.set_title(f'{JOINT_NAMES[fj]} — Displacement Distribution (modified frames only)')
    ax.set_xlabel('Displacement (m)'); ax.set_ylabel('Density'); ax.legend(); ax.grid(True, alpha=0.3)
fig.suptitle('Figure 7: Modification Magnitude Distribution — V8 vs V13', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{OUT_DIR}/07_displacement_histogram.png', dpi=150); plt.close()
print("  → 07_displacement_histogram.png")

# ── Save metrics ───────────────────────────────────────────
metrics_out = {
    'v8': {
        'full_error_mean': {str(fj): float(v8_full_errors[fj].mean()) for fj in FOOT_JOINTS},
        'full_error_max': {str(fj): float(v8_full_errors[fj].max()) for fj in FOOT_JOINTS},
        'full_error_gt_1m_pct': {str(fj): float((v8_full_errors[fj] > 1.0).mean()) for fj in FOOT_JOINTS},
        'fsr_fixed': metrics['V8-fixed']['FSR'],
        'fsr_foot_out': metrics['V8-foot_out']['FSR'],
        'jitter_fixed': metrics['V8-fixed']['Jitter'],
        'jitter_foot_out': metrics['V8-foot_out']['Jitter'],
        'foot_error_fixed': metrics['V8-fixed']['FootErr'],
        'foot_error_foot_out': metrics['V8-foot_out']['FootErr'],
        'mod_patterns': {
            str(fj): {
                'n_modified': int(mod['V8'][fj]['n_modified']),
                'n_groups': len(mod['V8'][fj]['groups']),
                'group_sizes': [g[2] for g in mod['V8'][fj]['groups']],
                'group_gaps': [mod['V8'][fj]['groups'][i+1][0] - mod['V8'][fj]['groups'][i][1]
                               for i in range(len(mod['V8'][fj]['groups'])-1)] if len(mod['V8'][fj]['groups']) > 1 else [],
            } for fj in FOOT_JOINTS
        },
    },
    'v13': {
        'fsr_fixed': metrics['V13-fixed']['FSR'],
        'jitter_fixed': metrics['V13-fixed']['Jitter'],
        'foot_error_fixed': metrics['V13-fixed']['FootErr'],
        'mod_patterns': {
            str(fj): {
                'n_modified': int(mod['V13'][fj]['n_modified']),
                'n_groups': len(mod['V13'][fj]['groups']),
                'group_sizes': [g[2] for g in mod['V13'][fj]['groups']],
                'group_gaps': [mod['V13'][fj]['groups'][i+1][0] - mod['V13'][fj]['groups'][i][1]
                               for i in range(len(mod['V13'][fj]['groups'])-1)] if len(mod['V13'][fj]['groups']) > 1 else [],
            } for fj in FOOT_JOINTS
        },
    },
    'original': {'fsr': metrics['Original']['FSR'], 'jitter': metrics['Original']['Jitter']},
}
with open(f'{OUT_DIR}/metrics.json', 'w') as f:
    json.dump(metrics_out, f, indent=2, default=float)
print(f"\nMetrics saved. Done!")
