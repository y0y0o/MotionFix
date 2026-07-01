"""
Prepare V14 Training Data
==========================
NEW PARADIGM: Simulate realistic foot skating (horizontal sliding during
ground contact) instead of random noise.

Key differences from V2:
  - Distortion ONLY on foot joints [7, 8, 10, 11]
  - Distortion ONLY in XZ plane (horizontal)
  - Distortion ONLY during ground contact frames
  - Smooth, accumulating drift (mimics VQ model errors)
  - Non-foot joints, Y-axis, and non-contact frames remain EXACTLY as original

This teaches the model:
  "When you see horizontal foot slide during ground contact → fix it.
   Everything else → leave alone."

Training pairs: ~15K (all 8177 motions × 2 augmentations)
"""

import numpy as np
import glob
import os
import sys
import time
import torch

# Remove current dir from path to avoid shadowing T2M-GPT's 'utils' package
_motionfix_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if p not in ('', _motionfix_dir)]
sys.path.insert(0, '/home3/nxkh91/projects/T2M-GPT')
from utils.motion_process import recover_from_ric

# ── Paths ─────────────────────────────────────────────────────────
HUMANML3D_PATH = "/home3/nxkh91/projects/HumanML3D/HumanML3D/new_joint_vecs"
MEAN_PATH = "/home3/nxkh91/projects/mdm/dataset/t2m_mean.npy"
STD_PATH = "/home3/nxkh91/projects/mdm/dataset/t2m_std.npy"
OUTPUT_DIR = "data/training/v14"

# ── Config ────────────────────────────────────────────────────────
NUM_MOTIONS = None         # None = use all available
AUGMENT_PER_MOTION = 2     # 2 variants per original → ~16K pairs
FOOT_JOINTS = [7, 8, 10, 11]  # L-Ankle, R-Ankle, L-Foot, R-Foot
SLIDE_INTENSITY = 0.06     # max horizontal drift per contact segment (m)
MIN_CONTACT_LEN = 3        # minimum frames for a contact segment
MIN_MOTION_LEN = 30        # skip very short motions


# ═══════════════════════════════════════════════════════════════════
# Data conversion
# ═══════════════════════════════════════════════════════════════════

def convert_to_joints(motion_263, mean, std):
    """Convert (T, 263) representation → (T, 22, 3) joint positions."""
    motion_denorm = motion_263 * std + mean
    motion_tensor = torch.from_numpy(motion_denorm).float().cuda()
    motion_joints = recover_from_ric(motion_tensor, 22)
    return motion_joints.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════
# Contact detection (for sliding simulation during data prep)
# ═══════════════════════════════════════════════════════════════════

def find_contact_segments(heights, ground, h_thresh=0.05):
    """
    Find contiguous contact segments for a single foot joint.

    Returns:
        list of (start_frame, end_frame) tuples (end is exclusive)
    """
    T = len(heights)
    segments = []
    in_contact = False
    seg_start = 0

    for t in range(T):
        if heights[t] < ground + h_thresh:
            if not in_contact:
                in_contact = True
                seg_start = t
        else:
            if in_contact:
                in_contact = False
                if t - seg_start >= MIN_CONTACT_LEN:
                    segments.append((seg_start, t))

    # Handle last segment
    if in_contact and T - seg_start >= MIN_CONTACT_LEN:
        segments.append((seg_start, T))

    return segments


# ═══════════════════════════════════════════════════════════════════
# Foot skating simulation
# ═══════════════════════════════════════════════════════════════════

def apply_smooth_slide(segment_y, length, intensity):
    """
    Generate smooth horizontal drift for one contact segment.

    Uses a sinusoidal drift curve with random phase/amplitude:
    — simulates the foot "wandering" while on the ground.

    Returns: drift_x (length,), drift_z (length,)
    """
    # Random drift angle in horizontal plane
    angle = np.random.uniform(0, 2 * np.pi)

    # Random amplitude (0.3x to 1.0x of intensity)
    amp = np.random.uniform(0.3, 1.0) * intensity

    # Random phase offset so each segment starts differently
    phase = np.random.uniform(0, 2 * np.pi)

    # Generate smooth curve: uses sin with varying frequency
    # Frequency: 1-3 cycles per segment
    freq = np.random.uniform(1.0, 3.0)
    t = np.linspace(0, 1, length)
    curve = np.sin(freq * 2 * np.pi * t + phase)

    # Apply smooth envelope (zero at start/end)
    envelope = 1.0 - np.cos(2 * np.pi * t)  # 0→1→0
    envelope = envelope ** 0.5              # Sharper rise/fall

    drift = amp * curve * envelope

    dx = drift * np.cos(angle)
    dz = drift * np.sin(angle)

    return dx, dz


def simulate_foot_skating(motion, intensity=None, seed=None):
    """
    Simulate VQ-model foot skating on clean HumanML3D motion.

    For each foot joint independently:
    1. Find ground contact segments
    2. In each segment, apply smooth horizontal drift (XZ only)
    3. Non-foot joints, Y-axis, and non-contact frames: UNTOUCHED

    Args:
        motion: (T, 22, 3) clean joint positions (root-relative)
        intensity: max drift per contact segment (m)
        seed: random seed for reproducibility

    Returns:
        distorted: (T, 22, 3) motion with simulated foot skating
    """
    if seed is not None:
        np.random.seed(seed)

    if intensity is None:
        intensity = SLIDE_INTENSITY * np.random.uniform(0.5, 2.0)

    distorted = motion.copy()
    T = motion.shape[0]

    for fj in FOOT_JOINTS:
        heights = motion[:, fj, 1]
        ground = np.percentile(heights, 5)

        segments = find_contact_segments(heights, ground)

        for seg_start, seg_end in segments:
            length = seg_end - seg_start
            dx, dz = apply_smooth_slide(None, length, intensity)

            # Apply ONLY to X (dim 0) and Z (dim 2)
            distorted[seg_start:seg_end, fj, 0] += dx
            distorted[seg_start:seg_end, fj, 2] += dz
            # Y (dim 1) is NOT modified — foot stays on ground

    return distorted


# ═══════════════════════════════════════════════════════════════════
# Multi-variant generation
# ═══════════════════════════════════════════════════════════════════

def generate_variants(clean_motion, n_variants=2, base_seed=0):
    """
    Generate multiple distorted variants of the same clean motion.

    Each variant has:
    - Different random seed → different slide directions/amplitudes
    - Different intensity multiplier (0.5x — 2.0x)
    - Sometimes only left foot slides, sometimes only right, sometimes both
    """
    variants = []
    for i in range(n_variants):
        seed = (base_seed * 1000 + i * 137 + 42) % (2**31 - 1)
        intensity = SLIDE_INTENSITY * np.random.RandomState(seed).uniform(0.5, 2.0)
        distorted = simulate_foot_skating(clean_motion, intensity=intensity, seed=seed)
        variants.append(distorted)
    return variants


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MotionFix V14 — Prepare Training Data")
    print("  Strategy: Foot Skating Simulation (Horizontal Drift@Contact)")
    print("=" * 70)
    t_start = time.time()

    # ── Load mean/std ──
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    print(f"  Mean: {MEAN_PATH}  shape={mean.shape}")
    print(f"  Std:  {STD_PATH}  shape={std.shape}")

    # ── List source files ──
    all_files = sorted(glob.glob(f"{HUMANML3D_PATH}/*.npy"))
    print(f"  Source motions: {len(all_files)} (.npy in {HUMANML3D_PATH})")

    if NUM_MOTIONS is not None:
        all_files = all_files[:NUM_MOTIONS]
    n_src = len(all_files)
    print(f"  Using: {n_src} motions × {AUGMENT_PER_MOTION} variants "
          f"= ~{n_src * AUGMENT_PER_MOTION} training pairs")

    # ── Convert and generate ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_motions = []
    convert_fail = 0
    too_short = 0

    print(f"\n  Phase 1/2: Converting (T,263) → (T,22,3)...")
    for i, filepath in enumerate(all_files):
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n_src}, valid: {len(valid_motions)}")

        try:
            motion_263 = np.load(filepath)
            motion_joints = convert_to_joints(motion_263, mean, std)

            if motion_joints.shape[1] != 22 or motion_joints.shape[2] != 3:
                convert_fail += 1
                continue

            if motion_joints.shape[0] < MIN_MOTION_LEN:
                too_short += 1
                continue

            valid_motions.append(motion_joints)

        except Exception as e:
            convert_fail += 1
            if convert_fail <= 3:
                print(f"    ⚠  Conversion error: {type(e).__name__}: {e}")

    n_valid = len(valid_motions)
    print(f"    Converted: {n_valid} motions "
          f"(failed: {convert_fail}, too_short: {too_short})")

    # ── Generate training pairs ──
    print(f"\n  Phase 2/2: Generating training pairs with foot skating...")
    pair_count = 0
    stats = {
        'total_slide_xz': [],   # total horizontal drift per pair
        'max_slide_xz': [],     # max horizontal drift per pair
        'modified_frames': [],  # frames actually modified per pair
    }

    for i, clean in enumerate(valid_motions):
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n_valid}, pairs: {pair_count}")

        base_seed = i * 7919  # prime multiplier for diverse seeds

        for aug_idx, distorted in enumerate(
            generate_variants(clean, n_variants=AUGMENT_PER_MOTION, base_seed=base_seed)
        ):
            # Save
            np.save(f"{OUTPUT_DIR}/distorted_{pair_count:06d}.npy", distorted)
            np.save(f"{OUTPUT_DIR}/target_{pair_count:06d}.npy", clean)

            # Stats
            diff = distorted - clean
            diff_xz = np.sqrt(diff[:, FOOT_JOINTS, 0]**2 + diff[:, FOOT_JOINTS, 2]**2)
            stats['total_slide_xz'].append(float(diff_xz.sum()))
            stats['max_slide_xz'].append(float(diff_xz.max()))
            stats['modified_frames'].append(
                int((diff_xz.max(axis=1) > 1e-6).sum())
            )

            pair_count += 1

    elapsed = time.time() - t_start
    print(f"\n  ✅ Done! {pair_count} training pairs → {OUTPUT_DIR}/")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Data quality report ──
    print(f"\n{'─'*70}")
    print(f"  📊 Training Data Quality Report")
    print(f"{'─'*70}")
    print(f"  Total pairs:          {pair_count}")
    print(f"  Unique source motions: {n_valid}")
    print(f"  Augmentations/motion: {AUGMENT_PER_MOTION}")
    print(f"")
    print(f"  ── Distortion Statistics ──")
    print(f"  Total XZ drift/sample:  mean={np.mean(stats['total_slide_xz']):.2f}m, "
          f"max={np.max(stats['total_slide_xz']):.2f}m")
    print(f"  Max XZ drift/sample:    mean={np.mean(stats['max_slide_xz']):.4f}m, "
          f"max={np.max(stats['max_slide_xz']):.4f}m")
    print(f"  Modified frames/sample: mean={np.mean(stats['modified_frames']):.1f}, "
          f"max={np.max(stats['modified_frames'])}")
    print(f"")
    modified_pct = np.mean(stats['modified_frames']) / np.mean([m.shape[0] for m in valid_motions]) * 100
    print(f"  ~{modified_pct:.0f}% of frames have foot sliding (should be ~20-40%)")
    print(f"{'─'*70}")

    # ── Verify: non-foot joints untouched ──
    print(f"\n  🔍 Verifying non-foot joints are untouched...")
    sample_dist = np.load(f"{OUTPUT_DIR}/distorted_000000.npy")
    sample_targ = np.load(f"{OUTPUT_DIR}/target_000000.npy")
    non_foot = [j for j in range(22) if j not in FOOT_JOINTS]
    nf_diff = np.abs(sample_dist[:, non_foot, :] - sample_targ[:, non_foot, :])
    if nf_diff.max() < 1e-6:
        print(f"  ✅ Non-foot joints (18/22): ZERO modification — PERFECT")
    else:
        print(f"  ⚠  Non-foot joints modified: max={nf_diff.max():.6f}m")

    # ── Verify: Y-axis untouched ──
    y_diff = np.abs(sample_dist[:, :, 1] - sample_targ[:, :, 1])
    if y_diff.max() < 1e-6:
        print(f"  ✅ Y-axis (all joints): ZERO modification — PERFECT")
    else:
        print(f"  ⚠  Y-axis modified: max={y_diff.max():.6f}m")

    # ── Verify: non-contact frames untouched ──
    # Check XZ drift on foot joints
    fj_diff_xz = np.sqrt(
        (sample_dist[:, FOOT_JOINTS, 0] - sample_targ[:, FOOT_JOINTS, 0])**2 +
        (sample_dist[:, FOOT_JOINTS, 2] - sample_targ[:, FOOT_JOINTS, 2])**2
    )
    modified_frames = fj_diff_xz.max(axis=1) > 1e-6
    print(f"  ℹ️  Foot XZ modified in {modified_frames.sum()}/{sample_dist.shape[0]} frames "
          f"({modified_frames.sum()/sample_dist.shape[0]*100:.0f}%)")
    print(f"{'─'*70}")

    print(f"\n  💾 Saved to: {OUTPUT_DIR}/")
    print(f"     distorted_000000.npy — distorted_{pair_count-1:06d}.npy")
    print(f"     target_000000.npy     — target_{pair_count-1:06d}.npy")


if __name__ == "__main__":
    main()
