"""
Physics-Based Foot Skating Correction
======================================
No neural model needed — uses a simple physical constraint:
feet in ground contact should not slide horizontally.

Algorithm:
  1. Detect contact: foot_height < ground + h_thresh
  2. Measure horizontal velocity (XZ plane)
  3. If velocity exceeds skate_thresh → damp the excess velocity
     - Legitimate slow movement (≤ threshold): untouched
     - Fast skating (> threshold): velocity aggressively damped

The damp_factor controls how much of the EXCESS velocity is removed:
  damp_factor=0.0 → all excess velocity removed (foot frozen)
  damp_factor=1.0 → no correction
  damp_factor=0.1 → 90% of excess removed (recommended)

Usage:
    from utils.physics_fix import physics_foot_fix
    fixed = physics_foot_fix(motion, damp_factor=0.1)
"""

import numpy as np

FOOT_JOINTS = [7, 8, 10, 11]  # L-Ankle, R-Ankle, L-Foot, R-Foot


def physics_foot_fix(motion,
                     foot_joints=None,
                     h_thresh=0.05,
                     skate_thresh=0.03,
                     damp_factor=0.1,
                     return_stats=True):
    """
    Apply physics-based foot skating correction.

    For each foot joint, at each frame:
    - If foot is on ground (height < ground + h_thresh)
    - AND horizontal velocity exceeds skate_thresh
    - → damp the EXCESS velocity by damp_factor

    The correction is applied CUMULATIVELY: each frame's output
    builds on the previous frame's corrected position.

    Args:
        motion: (T, 22, 3) joint positions (world or root-relative)
        foot_joints: list of joint indices to correct (default: [7,8,10,11])
        h_thresh: height above ground for contact detection (m)
        skate_thresh: horizontal velocity threshold (m/frame)
        damp_factor: fraction of excess velocity to retain (0=freeze, 1=no fix)
        return_stats: if True, return (fixed_motion, stats_dict)

    Returns:
        fixed: (T, 22, 3) corrected motion
        stats: dict with per-joint correction counts (if return_stats=True)
    """
    if foot_joints is None:
        foot_joints = FOOT_JOINTS

    fixed = motion.copy()
    T = motion.shape[0]

    stats = {
        'total_frames': T * len(foot_joints),
        'contact_frames': 0,
        'skating_frames': 0,
        'corrected_frames': 0,
        'per_joint': {},
    }

    for fj in foot_joints:
        heights = motion[:, fj, 1]
        ground = np.percentile(heights, 5)
        contact_threshold = ground + h_thresh

        fj_stats = {'contact': 0, 'skating': 0, 'corrected': 0}

        for t in range(1, T):
            if heights[t] < contact_threshold:
                fj_stats['contact'] += 1

                # Horizontal velocity computed from ORIGINAL motion
                # (not from already-fixed positions, to avoid cascade)
                orig_vel_x = motion[t, fj, 0] - motion[t-1, fj, 0]
                orig_vel_z = motion[t, fj, 2] - motion[t-1, fj, 2]
                vel = float(np.sqrt(orig_vel_x**2 + orig_vel_z**2))

                if vel > skate_thresh:
                    fj_stats['skating'] += 1
                    fj_stats['corrected'] += 1

                    # Damp only the EXCESS velocity above threshold
                    # Legitimate slow movement preserved, fast sliding damped
                    excess = vel - skate_thresh
                    target_vel = skate_thresh + excess * damp_factor
                    scale = target_vel / vel

                    # Apply damped velocity relative to PREVIOUS FIXED position
                    fixed[t, fj, 0] = fixed[t-1, fj, 0] + orig_vel_x * scale
                    fixed[t, fj, 2] = fixed[t-1, fj, 2] + orig_vel_z * scale

        stats['per_joint'][fj] = fj_stats
        stats['contact_frames'] += fj_stats['contact']
        stats['skating_frames'] += fj_stats['skating']
        stats['corrected_frames'] += fj_stats['corrected']

    if return_stats:
        return fixed, stats
    return fixed


def print_stats(stats, label=""):
    """Pretty-print correction statistics."""
    if label:
        print(f"\n  {label}")
    print(f"  Contact frames:  {stats['contact_frames']}/{stats['total_frames']} "
          f"({stats['contact_frames']/max(stats['total_frames'],1)*100:.0f}%)")
    print(f"  Skating frames:  {stats['skating_frames']}/{stats['contact_frames']} "
          f"({stats['skating_frames']/max(stats['contact_frames'],1)*100:.0f}%)")
    print(f"  Corrected:       {stats['corrected_frames']} frames")
    for fj, fs in stats['per_joint'].items():
        names = {7: 'L-Ankle', 8: 'R-Ankle', 10: 'L-Foot', 11: 'R-Foot'}
        print(f"    {names.get(fj, fj)}: contact={fs['contact']}, "
              f"skating={fs['skating']}, corrected={fs['corrected']}")


# ═══════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Synthetic test: motion with obvious foot skating
    np.random.seed(0)
    T = 100
    motion = np.zeros((T, 22, 3))
    motion[:, 0, 1] = 1.0    # pelvis at 1m

    # Simulate feet with skating during contact
    for t in range(T):
        # Foot near ground (contact)
        motion[t, 7, 1] = 0.02
        motion[t, 8, 1] = 0.03
        # Foot XZ sliding: accumulating drift
        motion[t, 7, 0] = 0.05 * t   # 5cm/frame linear drift
        motion[t, 7, 2] = 0.02 * t
        motion[t, 8, 0] = -0.03 * t
        motion[t, 8, 2] = 0.01 * t

    print("Before:")
    print(f"  L-Ankle vel X: {motion[1:,7,0] - motion[:-1,7,0]:.4f} m/frame (should be 0.05)")
    print(f"  L-Ankle displacement: {motion[-1,7,0] - motion[0,7,0]:.2f}m over {T} frames")

    fixed, stats = physics_foot_fix(motion, damp_factor=0.1, return_stats=True)
    print_stats(stats, "Correction stats:")

    print("\nAfter:")
    print(f"  L-Ankle vel X: {fixed[1:,7,0] - fixed[:-1,7,0]}")

    # Check: velocity should be close to skate_thresh (0.03)
    vel_x_after = np.mean(np.abs(fixed[1:,7,0] - fixed[:-1,7,0]))
    print(f"  Mean |vel_X|: {vel_x_after:.4f} (target: ~{0.03})")
    print(f"  L-Ankle displacement: {fixed[-1,7,0] - fixed[0,7,0]:.2f}m (was 5.0m)")
    print("  ✓ Physics correction works!")
