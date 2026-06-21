"""
MotionFix V12 (FRDM-inspired) — Test Script

Tests the V12 model on MoMask-generated motions (momask_50_results/no_ik/).

Inference: Enhanced selective replace with:
  1. Window expansion: blend [t-k, t+k] around skating frames → no isolated jumps
  2. Velocity-aware gating: reject predictions > 0.5m displacement → no 4m teleports

Metrics: FSR (Foot Skating Ratio) and Jitter before/after fixing.
"""

import torch
import numpy as np
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motionfix_model_v12 import MotionFixNetworkV12


# ================================================================
#  Metrics
# ================================================================
def compute_contact_labels(motion, foot_joints=(7, 8),
                           height_thresh=0.05, vel_thresh=0.5):
    """
    motion: (T, 22, 3) joint world coordinates
    returns: (T, 2) binary foot-ground contact labels
    """
    T = motion.shape[0]
    labels = np.zeros((T, 2), dtype=np.float32)

    for i, fj in enumerate(foot_joints):
        foot_y = motion[:, fj, 1]
        ground = np.percentile(foot_y, 5)
        threshold = ground + height_thresh

        for t in range(T):
            if foot_y[t] < threshold:
                if t > 0:
                    vel = np.linalg.norm(
                        motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]]
                    )
                    if vel < vel_thresh:
                        labels[t, i] = 1.0
                else:
                    labels[t, i] = 1.0
    return labels


def compute_fsr(motion, foot_joints=(7, 8), vel_thresh=0.03):
    """
    Foot Skating Ratio: fraction of contact frames where foot slides.

    Returns: (fsr, contact_count, skating_count)
    """
    contact = compute_contact_labels(motion, foot_joints)
    T = motion.shape[0]
    skating = 0
    contact_count = 0

    for i, fj in enumerate(foot_joints):
        for t in range(1, T):
            if contact[t, i] > 0.5:
                contact_count += 1
                vel = np.linalg.norm(
                    motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]]
                )
                if vel > vel_thresh:
                    skating += 1

    if contact_count == 0:
        return 0.0, 0, 0
    return skating / contact_count, contact_count, skating


def compute_jitter(motion):
    """
    Jitter metric: RMS of acceleration (foot joints only).
    Higher = more jittery / shaky.
    """
    foot_joints = [7, 8, 10, 11]
    foot_motion = motion[:, foot_joints, :]  # (T, 4, 3)

    vel = foot_motion[1:] - foot_motion[:-1]         # (T-1, 4, 3)
    acc = vel[1:] - vel[:-1]                          # (T-2, 4, 3)
    return np.sqrt((acc ** 2).mean())


def compute_mean_foot_error(original, fixed):
    """Mean L1 distance between original and fixed foot positions."""
    foot_joints = [7, 8, 10, 11]
    diffs = []
    for fj in foot_joints:
        diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
        diffs.append(diff)
    return np.mean(diffs)


# ================================================================
#  Fix Motion
# ================================================================
def fix_motion(model, motion, device='cuda'):
    """
    motion: (T, 22, 3) numpy array
    returns: (T, 22, 3) numpy array with enhanced selective replace

    Uses foot_only=True which triggers:
      - Window expansion (±window_size frames)
      - Velocity-aware gating
    """
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1).astype(np.float32)       # (T, 66)
    motion_tensor = torch.from_numpy(motion_flat).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        fixed_tensor = model(motion_tensor, foot_only=True)

    fixed_flat = fixed_tensor.squeeze(0).cpu().numpy()           # (T, 66)
    fixed = fixed_flat.reshape(T, 22, 3)
    return fixed


# ================================================================
#  Main
# ================================================================
def main():
    print("=" * 60)
    print("MotionFix V12 (FRDM-inspired) - Test")
    print("  Enhanced Selective Replace:")
    print("    - Window expansion (±2 frames)")
    print("    - Velocity gate (reject >0.5m jumps)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ---- Load model ----
    model = MotionFixNetworkV12(
        blend_alpha=0.5,
        window_size=2,
        velocity_gate=0.5,
    ).to(device)
    ckpt_path = "checkpoints_v12/best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Run train_v12.py first.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

    # ---- Find test files ----
    momask_dir = "../momask_50_results/no_ik"
    momask_files = sorted(glob.glob(f"{momask_dir}/*.npy"))

    if len(momask_files) == 0:
        print(f"\nNo MoMask files found in {momask_dir}")
        return

    output_dir = "fixed_outputs_v12"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Test each file ----
    header = (f"{'Name':<40} | {'FSR_before':>9} | {'FSR_after':>9} | "
              f"{'Jitter_before':>9} | {'Jitter_after':>9} | {'FtErr(m)':>8}")
    print(f"\n{header}")
    print("-" * len(header))

    results = []

    for filepath in momask_files:
        filename = os.path.basename(filepath)
        name = filename.replace('.npy', '')

        data = np.load(filepath)
        motion = data[0] if len(data.shape) == 4 else data   # (T, 22, 3)

        fsr_before, _, _ = compute_fsr(motion)
        jitter_before = compute_jitter(motion)

        fixed = fix_motion(model, motion, device)

        fsr_after, _, _ = compute_fsr(fixed)
        jitter_after = compute_jitter(fixed)
        foot_error = compute_mean_foot_error(motion, fixed)

        print(f"{name[:40]:<40} | {fsr_before:>8.1%} | {fsr_after:>8.1%} | "
              f"{jitter_before:>9.4f} | {jitter_after:>9.4f} | {foot_error:>7.4f}")

        # Save
        np.save(f"{output_dir}/{filename}", motion)
        np.save(f"{output_dir}/{filename.replace('.npy', '_fixed.npy')}", fixed)

        results.append({
            'name': name,
            'fsr_before': fsr_before,
            'fsr_after': fsr_after,
            'jitter_before': jitter_before,
            'jitter_after': jitter_after,
            'foot_error': foot_error,
        })

    # ---- Summary ----
    if results:
        avg_fsr_b = np.mean([r['fsr_before'] for r in results])
        avg_fsr_a = np.mean([r['fsr_after'] for r in results])
        avg_jit_b = np.mean([r['jitter_before'] for r in results])
        avg_jit_a = np.mean([r['jitter_after'] for r in results])
        avg_ft_err = np.mean([r['foot_error'] for r in results])

        print("-" * len(header))
        print(f"{'AVERAGE':<40} | {avg_fsr_b:>8.1%} | {avg_fsr_a:>8.1%} | "
              f"{avg_jit_b:>9.4f} | {avg_jit_a:>9.4f} | {avg_ft_err:>7.4f}")
        print(f"{'CHANGE':<40} | {'':>9} | {avg_fsr_a-avg_fsr_b:>+8.1%} | "
              f"{'':>9} | {avg_jit_a-avg_jit_b:>+9.4f} |")

        # Check for extreme outliers
        max_jump = max(r['foot_error'] for r in results)
        max_jump_name = max(results, key=lambda r: r['foot_error'])['name']
        print(f"\nMax foot modification: {max_jump:.4f}m ({max_jump_name[:50]})")
        if max_jump > 0.5:
            print(f"  ⚠️  Large modifications detected — check {max_jump_name}")

    print(f"\nSaved to: {output_dir}/")


if __name__ == "__main__":
    main()
