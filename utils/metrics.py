"""
Unified Motion Evaluation Metrics
===================================
7 standard metrics for ALL MotionFix versions:

  Priority 1 (核心 — 必须有):
    1. FSR      — Foot Skating Ratio (脚滑比例)
    2. Jitter   — Foot Acceleration RMS (脚步加速度抖动)
    3. Floating — Foot floating above ground during contact (浮空)
    4. FootErr  — Mean foot position error vs original (脚步误差)

  Priority 2 (强烈建议):
    5. BoneCV   — Bone Length Coefficient of Variation (骨骼长度稳定性)
    6. Penetration — Ground penetration depth (地面穿透)
    7. ContactAcc — Contact label preservation accuracy (接触准确率)

Usage:
    from utils.metrics import evaluate_all, compute_fsr, compute_jitter

    metrics = evaluate_all(fixed_motion, original_motion, label="V14")
    # → dict with all 7 metrics + prints formatted table
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ── Joint indices (HumanML3D 22-joint skeleton) ──────────────────────
ANKLE_JOINTS = [7, 8]       # L-Ankle, R-Ankle
FOOT_JOINTS = [7, 8, 10, 11]  # + L-Foot, R-Foot

# ── Bone definitions: (parent_joint, child_joint) ────────────────────
# HumanML3D 22-joint skeleton
BONE_PAIRS = [
    (0, 1),   # pelvis → L-hip
    (0, 2),   # pelvis → R-hip
    (1, 4),   # L-hip → L-knee
    (2, 5),   # R-hip → R-knee
    (4, 7),   # L-knee → L-ankle
    (5, 8),   # R-knee → R-ankle
    (7, 10),  # L-ankle → L-foot
    (8, 11),  # R-ankle → R-foot
    (0, 3),   # pelvis → spine1
    (3, 6),   # spine1 → spine2
    (6, 9),   # spine2 → neck
    (9, 12),  # neck → head
    (9, 13),  # neck → L-collar
    (9, 14),  # neck → R-collar
    (13, 15), # L-collar → L-shoulder
    (14, 16), # R-collar → R-shoulder
    (15, 17), # L-shoulder → L-elbow
    (16, 18), # R-shoulder → R-elbow
    (17, 19), # L-elbow → L-wrist
    (18, 20), # R-elbow → R-wrist
]

# Joint names for reporting
JOINT_NAMES = {
    0: "Pelvis", 1: "L-Hip", 2: "R-Hip", 3: "Spine1", 4: "L-Knee",
    5: "R-Knee", 6: "Spine2", 7: "L-Ankle", 8: "R-Ankle", 9: "Neck",
    10: "L-Foot", 11: "R-Foot", 12: "Head", 13: "L-Collar", 14: "R-Collar",
    15: "L-Shoulder", 16: "R-Shoulder", 17: "L-Elbow", 18: "R-Elbow",
    19: "L-Wrist", 20: "R-Wrist", 21: "Nose",
}


# ═══════════════════════════════════════════════════════════════════════
# Contact Detection (shared by FSR, Floating, ContactAcc)
# ═══════════════════════════════════════════════════════════════════════

def compute_contact_labels(motion: np.ndarray,
                           foot_joints: Tuple[int, ...] = (7, 8),
                           h_thresh: float = 0.05,
                           v_thresh: float = 0.5) -> np.ndarray:
    """
    Binary contact labels for ankle joints.

    Contact = foot near ground AND nearly stationary in horizontal plane.

    Args:
        motion: (T, 22, 3) joint positions in world/root-relative coords
        foot_joints: joint indices to check (default: ankles)
        h_thresh: height threshold above ground (m)
        v_thresh: maximum horizontal velocity for "stable contact" (m/frame)

    Returns:
        labels: (T, len(foot_joints)) — 1.0 = contact, 0.0 = not
    """
    T = motion.shape[0]
    labels = np.zeros((T, len(foot_joints)), dtype=np.float32)

    for i, fj in enumerate(foot_joints):
        foot_y = motion[:, fj, 1]
        ground = np.percentile(foot_y, 5)
        threshold = ground + h_thresh

        for t in range(T):
            if foot_y[t] < threshold:
                if t > 0:
                    vel = np.linalg.norm(
                        motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]]
                    )
                    if vel < v_thresh:
                        labels[t, i] = 1.0
                else:
                    labels[t, i] = 1.0

    return labels


# ═══════════════════════════════════════════════════════════════════════
# Metric 1: FSR — Foot Skating Ratio
# ═══════════════════════════════════════════════════════════════════════

def compute_fsr(motion: np.ndarray,
                foot_joints: Tuple[int, ...] = (7, 8),
                h_thresh: float = 0.05,
                v_thresh: float = 0.5,
                skate_thresh: float = 0.03) -> Tuple[float, int, int]:
    """
    Foot Skating Ratio: fraction of contact frames where the foot slides
    horizontally faster than skate_thresh.

    Lower is better. SOTA: ~5.5% (OmniControl/MaskControl).

    Returns:
        fsr: skating_frames / contact_frames
        contact_count: total contact frames (across both feet)
        skating_count: frames with horizontal velocity > skate_thresh
    """
    T = motion.shape[0]
    contact = compute_contact_labels(motion, foot_joints, h_thresh, v_thresh)

    skating = 0
    contact_count = 0

    for i, fj in enumerate(foot_joints):
        for t in range(1, T):
            if contact[t, i] > 0.5:
                contact_count += 1
                vel = np.linalg.norm(
                    motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]]
                )
                if vel > skate_thresh:
                    skating += 1

    if contact_count == 0:
        return 0.0, 0, 0
    return skating / contact_count, contact_count, skating


# ═══════════════════════════════════════════════════════════════════════
# Metric 2: Jitter — Foot Acceleration RMS
# ═══════════════════════════════════════════════════════════════════════

def compute_jitter(motion: np.ndarray,
                   foot_joints: Tuple[int, ...] = (7, 8, 10, 11)) -> float:
    """
    Jitter: RMS of foot joint acceleration (2nd derivative).
    Measures motion smoothness. Lower is better.

    Computed as:
        vel[t] = pos[t] - pos[t-1]
        acc[t] = vel[t] - vel[t-1]
        jitter = sqrt(mean(acc^2))
    """
    foot = motion[:, foot_joints, :]   # (T, 4, 3)
    vel = foot[1:] - foot[:-1]          # (T-1, 4, 3)
    acc = vel[1:] - vel[:-1]            # (T-2, 4, 3)
    return float(np.sqrt((acc ** 2).mean()))


# ═══════════════════════════════════════════════════════════════════════
# Metric 3: Floating — Foot hovering above ground during contact
# ═══════════════════════════════════════════════════════════════════════

def compute_floating(motion: np.ndarray,
                     foot_joints: Tuple[int, ...] = (7, 8),
                     h_thresh: float = 0.05) -> Tuple[float, int, int]:
    """
    Floating: fraction of contact-labeled frames where the foot is actually
    ABOVE the ground threshold. Indicates false contact classification
    or the model lifting feet when they should be planted.

    Lower is better.

    Returns:
        floating_ratio, floating_frame_count, total_contact_frames
    """
    contact = compute_contact_labels(motion, foot_joints, h_thresh)
    T = motion.shape[0]

    floating_count = 0
    contact_count = 0

    for i, fj in enumerate(foot_joints):
        foot_y = motion[:, fj, 1]
        ground = np.percentile(foot_y, 5)

        for t in range(T):
            if contact[t, i] > 0.5:
                contact_count += 1
                # Check: is foot actually above contact threshold?
                if foot_y[t] > ground + h_thresh:
                    floating_count += 1

    if contact_count == 0:
        return 0.0, 0, 0
    return floating_count / contact_count, floating_count, contact_count


# ═══════════════════════════════════════════════════════════════════════
# Metric 4: Foot Error — Position deviation from original
# ═══════════════════════════════════════════════════════════════════════

def compute_foot_error(fixed: np.ndarray,
                       original: np.ndarray,
                       foot_joints: Tuple[int, ...] = (7, 8, 10, 11)) -> float:
    """
    Mean Euclidean distance between fixed and original foot positions.
    Measures how much the model changed the feet (should be small).

    Lower is better — but NOT zero (model should make corrections).
    """
    errors = []
    for fj in foot_joints:
        diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
        errors.append(diff.mean())
    return float(np.mean(errors))


# ═══════════════════════════════════════════════════════════════════════
# Metric 5: Bone Length Consistency (BoneCV)
# ═══════════════════════════════════════════════════════════════════════

def compute_bone_length_consistency(motion: np.ndarray,
                                    bone_pairs: Optional[List[Tuple[int, int]]] = None
                                    ) -> float:
    """
    Bone Length Consistency: mean coefficient of variation (CV = std/mean)
    of bone lengths over time across all bones.

    Lower CV = more stable skeleton. High CV indicates bones
    stretching/shrinking → physically implausible.

    Returns:
        mean_cv: average (std/mean) across all bone pairs
    """
    if bone_pairs is None:
        bone_pairs = BONE_PAIRS

    cvs = []
    for parent, child in bone_pairs:
        lengths = np.linalg.norm(
            motion[:, child, :] - motion[:, parent, :], axis=1
        )
        mean_len = lengths.mean()
        if mean_len > 1e-6:
            cv = lengths.std() / mean_len
            cvs.append(cv)

    if not cvs:
        return 0.0
    return float(np.mean(cvs))


# ═══════════════════════════════════════════════════════════════════════
# Metric 6: Ground Penetration
# ═══════════════════════════════════════════════════════════════════════

def compute_ground_penetration(motion: np.ndarray,
                               foot_joints: Tuple[int, ...] = (7, 8, 10, 11)
                               ) -> Tuple[float, float, int]:
    """
    Ground Penetration: how far foot joints sink below the estimated
    ground plane.

    Returns:
        mean_penetration: average depth below ground (m)
        max_penetration: deepest penetration (m)
        penetration_frames: number of frames with penetration
    """
    total_pen = 0.0
    max_pen = 0.0
    pen_count = 0
    T = motion.shape[0]

    for fj in foot_joints:
        foot_y = motion[:, fj, 1]
        ground = np.percentile(foot_y, 5)

        for t in range(T):
            if foot_y[t] < ground:
                pen = ground - foot_y[t]
                total_pen += pen
                if pen > max_pen:
                    max_pen = pen
                pen_count += 1

    if pen_count == 0:
        return 0.0, 0.0, 0

    return float(total_pen / pen_count), float(max_pen), pen_count


# ═══════════════════════════════════════════════════════════════════════
# Metric 7: Contact Accuracy
# ═══════════════════════════════════════════════════════════════════════

def compute_contact_accuracy(fixed: np.ndarray,
                             original: np.ndarray,
                             foot_joints: Tuple[int, ...] = (7, 8),
                             h_thresh: float = 0.05,
                             v_thresh: float = 0.5) -> float:
    """
    Contact Accuracy: fraction of frames where the contact label
    (0/1) matches between fixed and original motion.

    High accuracy (>95%) means the model preserves original contact
    patterns — i.e., it doesn't inadvertently change foot-ground
    interaction states.

    Returns:
        accuracy: 0.0 — 1.0
    """
    c_fixed = compute_contact_labels(fixed, foot_joints, h_thresh, v_thresh)
    c_orig = compute_contact_labels(original, foot_joints, h_thresh, v_thresh)

    matches = (c_fixed == c_orig).sum()
    total = c_fixed.size

    return float(matches / total)


# ═══════════════════════════════════════════════════════════════════════
# Combined evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_all(fixed_motion: np.ndarray,
                 original_motion: Optional[np.ndarray] = None,
                 label: str = "",
                 verbose: bool = True,
                 also_original: bool = False) -> Dict:
    """
    Compute all 7 metrics for a motion.

    Args:
        fixed_motion: (T, 22, 3) — the fixed/processed motion
        original_motion: (T, 22, 3) — reference (optional)
        label: name for print output
        verbose: print formatted table
        also_original: also compute metrics on original (shows before/after)

    Returns:
        dict with keys: FSR, Jitter, Floating, FootErr, BoneCV,
                        PenetrationMean, PenetrationMax, ContactAcc,
                        + per-metric raw counts
    """
    m = {}

    # ── Always computed ──
    fsr, c_n, s_n = compute_fsr(fixed_motion)
    m['FSR'] = fsr
    m['ContactFrames'] = c_n
    m['SkatingFrames'] = s_n

    m['Jitter'] = compute_jitter(fixed_motion)

    floating, float_n, cont_n = compute_floating(fixed_motion)
    m['Floating'] = floating
    m['FloatingFrames'] = float_n

    m['BoneCV'] = compute_bone_length_consistency(fixed_motion)

    pen_mean, pen_max, pen_n = compute_ground_penetration(fixed_motion)
    m['PenetrationMean'] = pen_mean
    m['PenetrationMax'] = pen_max
    m['PenetrationFrames'] = pen_n

    # ── Comparative (need original) ──
    if original_motion is not None:
        m['FootErr'] = compute_foot_error(fixed_motion, original_motion)
        m['ContactAcc'] = compute_contact_accuracy(fixed_motion, original_motion)

    # ── Original metrics (before/after comparison) ──
    if also_original and original_motion is not None:
        m_orig = {}
        m_orig['FSR'] = compute_fsr(original_motion)[0]
        m_orig['Jitter'] = compute_jitter(original_motion)
        m_orig['Floating'] = compute_floating(original_motion)[0]
        m_orig['BoneCV'] = compute_bone_length_consistency(original_motion)
        pen_m, pen_x, pen_n_orig = compute_ground_penetration(original_motion)
        m_orig['PenetrationMean'] = pen_m
        m_orig['PenetrationMax'] = pen_x
        m['original'] = m_orig

    if verbose:
        _print_metrics(m, label, original_motion is not None)

    return m


# ═══════════════════════════════════════════════════════════════════════
# Pretty printing
# ═══════════════════════════════════════════════════════════════════════

def format_fsr(fsr: float) -> str:
    """Color-coded FSR string."""
    if fsr < 0.03:     return f"\033[32m{fsr:6.1%}\033[0m"   # green: excellent
    elif fsr < 0.08:   return f"\033[33m{fsr:6.1%}\033[0m"   # yellow: acceptable
    else:               return f"\033[31m{fsr:6.1%}\033[0m"   # red: poor


def _print_metrics(m: Dict, label: str, has_ref: bool):
    """Pretty-print all metrics."""
    print(f"\n{'─'*62}")
    print(f"  📊 {label}")
    print(f"{'─'*62}")

    # Priority 1 metrics
    print(f"  {'FSR':<24s}: {m.get('FSR', 0):>6.2%}  "
          f"({m.get('SkatingFrames', 0)} skating / {m.get('ContactFrames', 0)} contact frames)")
    print(f"  {'Jitter':<24s}: {m.get('Jitter', 0):>8.4f} m/frame²")
    print(f"  {'Floating':<24s}: {m.get('Floating', 0):>6.2%}  "
          f"({m.get('FloatingFrames', 0)} floating frames)")
    if has_ref and 'FootErr' in m:
        print(f"  {'Foot Error':<24s}: {m['FootErr']:>7.4f} m")
    if has_ref and 'ContactAcc' in m:
        print(f"  {'Contact Accuracy':<24s}: {m['ContactAcc']:>6.2%}")

    # Priority 2 metrics
    print(f"  {'Bone Length CV':<24s}: {m.get('BoneCV', 0):>8.4f}")
    print(f"  {'Ground Penetration':<24s}: mean {m.get('PenetrationMean', 0):.4f}m, "
          f"max {m.get('PenetrationMax', 0):.4f}m "
          f"({m.get('PenetrationFrames', 0)} frames)")

    # Before/after comparison
    if 'original' in m:
        o = m['original']
        print(f"\n  ── Before ──────────────────────────────────────────")
        print(f"  {'FSR (before)':<24s}: {o['FSR']:>6.2%}")
        print(f"  {'Jitter (before)':<24s}: {o['Jitter']:>8.4f}")
        print(f"  {'Floating (before)':<24s}: {o['Floating']:>6.2%}")
        print(f"  {'BoneCV (before)':<24s}: {o['BoneCV']:>8.4f}")

    print(f"{'─'*62}")


def print_summary_table(results: List[Dict], label: str = "SUMMARY"):
    """
    Print aggregated summary across multiple motions.

    Args:
        results: list of dicts from evaluate_all()
    """
    if not results:
        print("No results to summarize.")
        return

    n = len(results)
    keys_always = ['FSR', 'Jitter', 'Floating', 'BoneCV',
                   'PenetrationMean', 'PenetrationMax']
    keys_ref = ['FootErr', 'ContactAcc']

    print(f"\n{'='*70}")
    print(f"  📋 {label} — {n} motions aggregated")
    print(f"{'='*70}")

    def _avg(key):
        vals = [r[key] for r in results if key in r]
        return np.mean(vals) if vals else 0.0

    print(f"  {'FSR':<24s}: {_avg('FSR'):>6.2%}")
    print(f"  {'Jitter':<24s}: {_avg('Jitter'):>8.4f} m/frame²")
    print(f"  {'Floating':<24s}: {_avg('Floating'):>6.2%}")
    print(f"  {'Foot Error':<24s}: {_avg('FootErr'):>7.4f} m")
    print(f"  {'Contact Accuracy':<24s}: {_avg('ContactAcc'):>6.2%}")
    print(f"  {'Bone Length CV':<24s}: {_avg('BoneCV'):>8.4f}")
    print(f"  {'Penetration (mean)':<24s}: {_avg('PenetrationMean'):>7.4f} m")
    print(f"  {'Penetration (max)':<24s}: {_avg('PenetrationMax'):>7.4f} m")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create synthetic test motion
    np.random.seed(42)
    T = 100
    motion = np.zeros((T, 22, 3))
    # Simulate pelvis at ~1m height
    motion[:, 0, 1] = 1.0
    # Feet near ground
    motion[:, 7, 1] = 0.02
    motion[:, 8, 1] = 0.03
    # Add some walking motion
    for t in range(T):
        motion[t, 7, 0] = 0.1 * np.sin(t * 0.1)
        motion[t, 8, 0] = -0.1 * np.sin(t * 0.1)
        motion[t, 7, 2] = 0.05 * t / T
        motion[t, 8, 2] = 0.05 * t / T

    print("Testing metrics module...")
    metrics = evaluate_all(motion, motion, label="Synthetic Test",
                           also_original=True)
    print(f"\nAll 7 metrics computed successfully.")
    print(f"  FSR={metrics['FSR']:.3f}, Jitter={metrics['Jitter']:.4f}, "
          f"Floating={metrics['Floating']:.3f}")
    print(f"  BoneCV={metrics['BoneCV']:.4f}, "
          f"Penetration={metrics['PenetrationMean']:.4f}m")
    print(f"  FootErr={metrics.get('FootErr', 0):.4f}, "
          f"ContactAcc={metrics.get('ContactAcc', 0):.3f}")
