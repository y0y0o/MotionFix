"""
Deep dive analysis: p000021 V13 MoMask — Sudden foot flash/jump investigation.

p000021 stats:
  FSR: 22.5% → 16.7% (-5.8pp)
  Jitter: 0.011 → 0.288 (26x increase!)
  Foot error: 19.9cm avg, 178.7cm max
  T=196 frames
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Load data
# ============================================================
ORIG_PATH = "momask_50_results/no_ik/p000021_rotation_person_is_walking_normally_in_a.npy"
FIXED_PATH = "fixed_outputs_v13_momask/p000021_rotation_person_is_walking_normally_in_a_fixed.npy"

original = np.load(ORIG_PATH).astype(np.float32)
fixed = np.load(FIXED_PATH).astype(np.float32)

if len(original.shape) == 4:
    original = original[0]
if len(fixed.shape) == 4:
    fixed = fixed[0]

T = original.shape[0]
print(f"T={T}, original shape={original.shape}, fixed shape={fixed.shape}")

# Joint indices
FOOT_JOINTS = [7, 8, 10, 11]  # Left Ankle, Right Ankle, Left Foot, Right Foot
JOINT_NAMES = {7: "Left Ankle", 8: "Right Ankle", 10: "Left Foot", 11: "Right Foot"}

# ============================================================
# Frame-by-frame analysis
# ============================================================

def analyze_foot_motion(data, label):
    """Detailed frame-by-frame foot motion analysis"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for fj in FOOT_JOINTS:
        pos = data[:, fj, :]  # (T, 3)
        vel = pos[1:] - pos[:-1]  # (T-1, 3)
        vel_mag = np.linalg.norm(vel, axis=1)  # (T-1,)
        acc = vel[1:] - vel[:-1]  # (T-2, 3)
        acc_mag = np.linalg.norm(acc, axis=1)  # (T-2,)
        jerk = acc[1:] - acc[:-1]  # (T-3, 3)
        jerk_mag = np.linalg.norm(jerk, axis=1)  # (T-3,)

        print(f"\n{JOINT_NAMES[fj]} (joint {fj}):")
        print(f"  Height range: [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]")
        print(f"  Height p5/median/p95: {np.percentile(pos[:,1], 5):.3f} / {np.median(pos[:,1]):.3f} / {np.percentile(pos[:,1], 95):.3f}")
        print(f"  Velocity: mean={vel_mag.mean():.4f}, max={vel_mag.max():.4f}, p99={np.percentile(vel_mag, 99):.4f}")
        print(f"  Acceleration: mean={acc_mag.mean():.4f}, max={acc_mag.max():.4f}, p99={np.percentile(acc_mag, 99):.4f}")
        print(f"  Jerk: mean={jerk_mag.mean():.4f}, max={jerk_mag.max():.4f}, p99={np.percentile(jerk_mag, 99):.4f}")

        # Find top spike frames
        top_vel_idx = np.argsort(vel_mag)[-10:][::-1]
        top_acc_idx = np.argsort(acc_mag)[-10:][::-1]
        top_jerk_idx = np.argsort(jerk_mag)[-10:][::-1]

        print(f"\n  Top 10 velocity spike frames (magnitude):")
        for idx in top_vel_idx:
            print(f"    frame {idx}: vel={vel_mag[idx]:.4f}, pos=({pos[idx,0]:.3f}, {pos[idx,1]:.3f}, {pos[idx,2]:.3f})")

        print(f"\n  Top 10 acceleration spike frames:")
        for idx in top_acc_idx:
            print(f"    frame {idx}: acc={acc_mag[idx]:.4f}")

        print(f"\n  Top 10 jerk spike frames:")
        for idx in top_jerk_idx:
            print(f"    frame {idx}: jerk={jerk_mag[idx]:.4f}")

    return

# Run analysis on both
analyze_foot_motion(original, "ORIGINAL (MoMask raw)")
analyze_foot_motion(fixed, "FIXED (V13)")

# ============================================================
# Frame-by-frame foot displacement: original vs fixed
# ============================================================
print(f"\n{'='*60}")
print("  FRAME-BY-FRAME FOOT DISPLACEMENT (original vs fixed)")
print(f"{'='*60}")

for fj in FOOT_JOINTS:
    diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
    top_diff_idx = np.argsort(diff)[-20:][::-1]

    print(f"\n{JOINT_NAMES[fj]} (joint {fj}):")
    print(f"  Displacement: mean={diff.mean():.4f}, max={diff.max():.4f}, p95={np.percentile(diff, 95):.4f}")
    print(f"  Frames with displacement > 0.3m: {np.sum(diff > 0.3)}")
    print(f"  Frames with displacement > 0.5m: {np.sum(diff > 0.5)}")
    print(f"  Frames with displacement > 1.0m: {np.sum(diff > 1.0)}")

    print(f"\n  Top 20 displacement frames:")
    for idx in top_diff_idx:
        orig_pos = original[idx, fj, :]
        fixed_pos = fixed[idx, fj, :]
        print(f"    frame {idx}: displ={diff[idx]:.4f}m | orig=({orig_pos[0]:.3f},{orig_pos[1]:.3f},{orig_pos[2]:.3f}) fixed=({fixed_pos[0]:.3f},{fixed_pos[1]:.3f},{fixed_pos[2]:.3f})")

# ============================================================
# Selective replace analysis: which frames were modified?
# ============================================================
print(f"\n{'='*60}")
print("  SELECTIVE REPLACE: WHICH FRAMES ARE MODIFIED?")
print(f"{'='*60}")

for fj in FOOT_JOINTS:
    diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
    modified_frames = np.where(diff > 0.0001)[0]
    print(f"\n{JOINT_NAMES[fj]} (joint {fj}): {len(modified_frames)}/{T} frames modified ({len(modified_frames)/T*100:.1f}%)")

    # Check if modifications are on contiguous frames or isolated
    if len(modified_frames) > 0:
        gaps = modified_frames[1:] - modified_frames[:-1]
        isolated = np.where(gaps > 1)[0]
        print(f"  Isolated single-frame modifications: {len(isolated)}")
        if len(isolated) > 0:
            print(f"  Gap positions: {list(gaps[isolated])}")

        # Show modification groups
        groups = []
        start = modified_frames[0]
        prev = modified_frames[0]
        for f in modified_frames[1:]:
            if f - prev > 1:
                groups.append((start, prev))
                start = f
            prev = f
        groups.append((start, prev))
        print(f"  Continuous groups: {len(groups)}")
        for g in groups[:10]:
            print(f"    frames [{g[0]}, {g[1]}] ({g[1]-g[0]+1} frames)")

# ============================================================
# Detailed skating detection for the V13 fixed output
# (replicating _selective_replace logic)
# ============================================================
print(f"\n{'='*60}")
print("  SIMULATED SKATING DETECTION on FIXED output")
print(f"{'='*60}")

for fj in FOOT_JOINTS:
    heights = fixed[:, fj, 1]  # Y coordinate
    ground_level = np.percentile(heights, 5)
    contact_threshold = ground_level + 0.05
    print(f"\n{JOINT_NAMES[fj]} (joint {fj}): ground={ground_level:.4f}, contact_thresh={contact_threshold:.4f}")

    skating_frames = []
    contact_frames = []
    for t in range(1, T):
        height = heights[t]
        if height < contact_threshold:
            contact_frames.append(t)
            vel = np.linalg.norm(fixed[t, fj, [0, 2]] - fixed[t-1, fj, [0, 2]])
            if vel > 0.03:
                skating_frames.append((t, vel, height))

    print(f"  Contact frames: {len(contact_frames)}/{T-1}")
    print(f"  Skating frames detected: {len(skating_frames)}")

    if len(skating_frames) > 0:
        print(f"  Skating frame details (velocity > 0.03 during contact):")
        for t, vel, height in skating_frames[:15]:
            # Check if this frame was *not* modified (model output untouched)
            diff = np.linalg.norm(fixed[t, fj, :] - original[t, fj, :])
            print(f"    frame {t}: vel={vel:.4f}, height={height:.4f}, diff_from_orig={diff:.4f}")

# ============================================================
# Full model prediction analysis (what does the model output?)
# ============================================================
print(f"\n{'='*60}")
print("  MODEL PREDICTION (full_output) ANALYSIS")
print(f"{'='*60}")

# We need to run the actual model to get full_output vs selective_replace
# For now, compute the effective "correction" magnitude per frame
for fj in FOOT_JOINTS:
    diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
    # Frames where correction > 0: these are where blend was applied
    corrected_frames = np.where(diff > 0.0001)[0]

    if len(corrected_frames) > 0:
        # For corrected frames, compute  correction direction
        corrections = fixed[corrected_frames, fj, :] - original[corrected_frames, fj, :]
        avg_correction = np.mean(np.linalg.norm(corrections, axis=1))
        max_correction = np.max(np.linalg.norm(corrections, axis=1))

        print(f"\n{JOINT_NAMES[fj]} (joint {fj}):")
        print(f"  Corrections applied at {len(corrected_frames)} frames")
        print(f"  Avg correction magnitude: {avg_correction:.4f}m")
        print(f"  Max correction magnitude: {max_correction:.4f}m")

        # Show extreme corrections
        correction_mags = np.linalg.norm(corrections, axis=1)
        extreme_idx = np.argsort(correction_mags)[-10:][::-1]
        print(f"  Top 10 largest corrections:")
        for i in extreme_idx:
            actual_frame = corrected_frames[i]
            mag = correction_mags[i]
            dir_vec = corrections[i]
            print(f"    frame {actual_frame}: {mag:.4f}m, dir=({dir_vec[0]:.3f},{dir_vec[1]:.3f},{dir_vec[2]:.3f})")

# ============================================================
# Temporal consistency: check if corrections are coherent
# ============================================================
print(f"\n{'='*60}")
print("  TEMPORAL CONSISTENCY OF CORRECTIONS")
print(f"{'='*60}")

for fj in FOOT_JOINTS:
    diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
    corrected_frames = np.where(diff > 0.0001)[0]

    if len(corrected_frames) < 2:
        print(f"\n{JOINT_NAMES[fj]}: too few corrections to analyze")
        continue

    # Check if consecutive correction magnitudes are smooth
    correction_diffs = []
    for i in range(len(corrected_frames) - 1):
        if corrected_frames[i+1] - corrected_frames[i] == 1:
            # Consecutive frames
            mag1 = diff[corrected_frames[i]]
            mag2 = diff[corrected_frames[i+1]]
            correction_diffs.append(abs(mag1 - mag2))

    if correction_diffs:
        print(f"\n{JOINT_NAMES[fj]} (joint {fj}):")
        print(f"  Consecutive correction pairs: {len(correction_diffs)}")
        print(f"  Correction magnitude changes (consecutive):")
        print(f"    mean={np.mean(correction_diffs):.4f}, max={np.max(correction_diffs):.4f}, p95={np.percentile(correction_diffs, 95):.4f}")

        # Check for on/off patterns (correction appears and disappears)
        gaps = corrected_frames[1:] - corrected_frames[:-1]
        isolated_corrections = np.sum(gaps > 1)
        print(f"  Isolated correction events (gap > 1 frame after): {isolated_corrections}")
        if isolated_corrections > 0:
            jump_gaps = gaps[gaps > 1]
            print(f"  Gap sizes: min={jump_gaps.min()}, max={jump_gaps.max()}")

# ============================================================
# PLOTS
# ============================================================
SAVE_DIR = "analysis_v13/p000021"
import os
os.makedirs(SAVE_DIR, exist_ok=True)

# Plot 1: Foot trajectories (XZ, Y over time) — original vs fixed
fig, axes = plt.subplots(4, 3, figsize=(20, 18))
row_labels = ['X (horizontal)', 'Y (height)', 'Z (depth)']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']  # per foot joint

for row, dim_name in enumerate(row_labels):
    dim = row  # 0=X, 1=Y, 2=Z
    for col, fj in enumerate(FOOT_JOINTS):
        ax = axes[col, row]
        ax.plot(original[:, fj, dim], color='gray', alpha=0.5, linewidth=1, label='Original')
        ax.plot(fixed[:, fj, dim], color=colors[col], linewidth=1.2, label='V13 Fixed')
        ax.set_title(f'{JOINT_NAMES[fj]} - {dim_name}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(fontsize=7)

fig.suptitle('p000021 — Foot Trajectories: Original vs V13 Fixed', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{SAVE_DIR}/01_foot_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {SAVE_DIR}/01_foot_trajectories.png")

# Plot 2: Foot displacement (diff between original and fixed) per frame
fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
for i, fj in enumerate(FOOT_JOINTS):
    ax = axes[i]
    diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
    ax.bar(range(T), diff, color=colors[i], alpha=0.7, width=1)
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='0.3m threshold')
    ax.axhline(y=0.5, color='darkred', linestyle='--', alpha=0.5, label='0.5m threshold')
    ax.set_ylabel(f'Displacement (m)', fontsize=11)
    ax.set_title(f'{JOINT_NAMES[fj]} (joint {fj}) — Frame Displacement', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

    # Annotate top spikes
    top5 = np.argsort(diff)[-5:][::-1]
    for t in top5:
        ax.annotate(f't={t}\n{diff[t]:.2f}m', (t, diff[t]),
                   textcoords="offset points", xytext=(0, 10), ha='center',
                   fontsize=7, color='red', fontweight='bold')

axes[-1].set_xlabel('Frame', fontsize=12)
fig.suptitle('p000021 — Frame-by-Frame Foot Displacement (Original vs V13)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{SAVE_DIR}/02_frame_displacement.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {SAVE_DIR}/02_frame_displacement.png")

# Plot 3: Velocity comparison
fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
for i, fj in enumerate(FOOT_JOINTS):
    ax = axes[i]
    vel_orig = np.linalg.norm(original[1:, fj, :] - original[:-1, fj, :], axis=1)
    vel_fixed = np.linalg.norm(fixed[1:, fj, :] - fixed[:-1, fj, :], axis=1)
    ax.plot(vel_orig, color='gray', alpha=0.5, linewidth=1, label='Original')
    ax.plot(vel_fixed, color=colors[i], linewidth=1.2, label='V13 Fixed')
    ax.set_ylabel(f'Velocity (m/frame)', fontsize=11)
    ax.set_title(f'{JOINT_NAMES[fj]} — Foot Velocity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

axes[-1].set_xlabel('Frame', fontsize=12)
fig.suptitle('p000021 — Foot Velocity: Original vs V13 Fixed', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{SAVE_DIR}/03_velocity_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {SAVE_DIR}/03_velocity_comparison.png")

# Plot 4: Acceleration comparison (key for jitter)
fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
for i, fj in enumerate(FOOT_JOINTS):
    ax = axes[i]
    vel_orig = original[1:, fj, :] - original[:-1, fj, :]
    vel_fixed = fixed[1:, fj, :] - fixed[:-1, fj, :]
    acc_orig = np.linalg.norm(vel_orig[1:] - vel_orig[:-1], axis=1)
    acc_fixed = np.linalg.norm(vel_fixed[1:] - vel_fixed[:-1], axis=1)
    ax.plot(acc_orig, color='gray', alpha=0.5, linewidth=1, label='Original')
    ax.plot(acc_fixed, color=colors[i], linewidth=1.2, label='V13 Fixed')
    ax.set_ylabel(f'Acceleration (m/frame²)', fontsize=11)
    ax.set_title(f'{JOINT_NAMES[fj]} — Foot Acceleration (= Jitter Source)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

    # Mark extreme acceleration spikes
    top3 = np.argsort(acc_fixed)[-3:][::-1]
    for t in top3:
        ax.axvline(x=t, color='red', linestyle=':', alpha=0.5, linewidth=0.8)

axes[-1].set_xlabel('Frame', fontsize=12)
fig.suptitle('p000021 — Foot Acceleration: Original vs V13 Fixed', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{SAVE_DIR}/04_acceleration_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {SAVE_DIR}/04_acceleration_comparison.png")

# Plot 5: Heatmap — which frames are modified?
fig, ax = plt.subplots(figsize=(18, 4))
diff_matrix = np.zeros((4, T))
for i, fj in enumerate(FOOT_JOINTS):
    diff_matrix[i] = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)

im = ax.imshow(diff_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_yticks(range(4))
ax.set_yticklabels([JOINT_NAMES[fj] for fj in FOOT_JOINTS])
ax.set_xlabel('Frame', fontsize=12)
ax.set_title('p000021 — Modified Frame Heatmap (V13 Corrections)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Displacement (m)')
fig.savefig(f'{SAVE_DIR}/05_modification_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {SAVE_DIR}/05_modification_heatmap.png")

# Plot 6: Zoomed foot height near contact threshold
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
for i, fj in enumerate([7, 8]):  # Ankles only
    row, col = i // 2, i % 2
    ax = axes[i]
    h_orig = original[:, fj, 1]
    h_fixed = fixed[:, fj, 1]
    ground = np.percentile(h_fixed, 5)
    threshold = ground + 0.05

    ax.plot(h_orig, color='gray', alpha=0.5, linewidth=1.5, label='Original')
    ax.plot(h_fixed, color=colors[i], linewidth=1.5, label='V13 Fixed')
    ax.axhline(y=ground, color='blue', linestyle='-', alpha=0.5, label=f'Ground ({ground:.3f})')
    ax.axhline(y=threshold, color='orange', linestyle='--', alpha=0.5, label=f'Contact thresh ({threshold:.3f})')

    # Mark frames where skating is detected in fixed output
    for t in range(1, T):
        if h_fixed[t] < threshold:
            vel = np.linalg.norm(fixed[t, fj, [0, 2]] - fixed[t-1, fj, [0, 2]])
            if vel > 0.03:
                ax.scatter(t, h_fixed[t], color='red', s=30, zorder=5)

    ax.set_title(f'{JOINT_NAMES[fj]} — Height & Skating Detection', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height (m)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

axes[-1].set_xlabel('Frame', fontsize=12)
fig.suptitle('p000021 — Foot Height & Skating Detection (V13)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{SAVE_DIR}/06_skating_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {SAVE_DIR}/06_skating_detection.png")

print(f"\n{'='*60}")
print("DEEP DIVE COMPLETE")
print(f"All outputs in {SAVE_DIR}/")
print(f"{'='*60}")
