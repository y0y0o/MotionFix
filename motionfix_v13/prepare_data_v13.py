"""
Prepare V13 Training Data — V8 pipeline with 3-5x amplified noise.

Reuses clean target motions from training_data_v2 (skips slow GPU conversion).
Applies amplified distortions → training_data_v13/.
"""
import numpy as np
import glob
import os
import sys
from scipy.ndimage import uniform_filter1d

# Reuse V8's clean targets (already converted from HumanML3D 263d → joints)
SOURCE_DATA_DIR = "training_data_v2"
OUTPUT_DIR = "training_data_v13"
NUM_SAMPLES = 5000
AUGMENT_PER_SAMPLE = 3
V8_AUGMENT_PER_SAMPLE = 3  # V8 used 3 augmentations per sample


# ================================================================
#  Distortion Functions — AMPLIFIED (3-5x vs V8)
# ================================================================

def add_foot_skating_heavy(motion, intensity=1.2):
    """
    Heavy foot skating: horizontal displacement when foot on ground.
    V8: intensity=0.3, dx up to 0.03
    V13: intensity=1.2, dx up to 0.12 (4x)
    """
    distorted = motion.copy()
    T = motion.shape[0]
    for foot_idx in [10, 11]:
        heights = motion[:, foot_idx, 1]
        ground = np.percentile(heights, 5)
        contact_threshold = ground + 0.05
        for t in range(T - 1):
            if heights[t] < contact_threshold:
                dx = np.random.uniform(-intensity, intensity) * 0.1
                dz = np.random.uniform(-intensity, intensity) * 0.1
                distorted[t, foot_idx, 0] += dx
                distorted[t, foot_idx, 2] += dz
    return distorted


def add_foot_skating_very_heavy(motion, intensity=2.5):
    """
    Very heavy foot skating — simulates worst MoMask artifacts.
    dx up to 0.25 per frame (8x V8).
    """
    return add_foot_skating_heavy(motion, intensity=intensity)


def add_temporal_smoothing(motion):
    """Temporal smoothing — narrow window preserves more artifact structure."""
    distorted = motion.copy()
    window = np.random.randint(3, 9)  # narrower than V8's 7-15
    for joint in range(motion.shape[1]):
        for dim in range(3):
            distorted[:, joint, dim] = uniform_filter1d(
                motion[:, joint, dim], size=window, mode='nearest'
            )
    return distorted


def add_foot_drift_heavy(motion):
    """
    Heavy foot drift — cumulative sliding during ground contact.
    V8: drift 0.01/step, cum. ~0.2
    V13: drift 0.04/step, cum. ~0.8 (4x)
    """
    distorted = motion.copy()
    T = motion.shape[0]
    for foot_idx in [10, 11]:
        heights = motion[:, foot_idx, 1]
        ground = np.percentile(heights, 5)
        contact_threshold = ground + 0.05
        in_contact = False
        drift_x, drift_z = 0.0, 0.0
        for t in range(T):
            if heights[t] < contact_threshold:
                if not in_contact:
                    drift_x = np.random.uniform(-0.04, 0.04)
                    drift_z = np.random.uniform(-0.04, 0.04)
                    in_contact = True
                distorted[t, foot_idx, 0] += drift_x * (t % 20)
                distorted[t, foot_idx, 2] += drift_z * (t % 20)
            else:
                in_contact = False
    return distorted


def add_y_shift_heavy(motion):
    """Y-axis shift — larger range. V8: ±0.08, V13: ±0.15 (2x)."""
    distorted = motion.copy()
    shift = np.random.uniform(-0.15, 0.15)
    distorted[:, :, 1] += shift
    return distorted


def add_spatial_noise_heavy(motion):
    """Spatial noise on ALL joints. V8: 0.01-0.05, V13: 0.03-0.12 (3x)."""
    noise_std = np.random.uniform(0.03, 0.12)
    noise = np.random.normal(0, noise_std, motion.shape)
    return motion + noise


def add_joint_jitter(motion):
    """
    Per-joint high-frequency jitter — simulates VQ quantization noise.
    Each joint gets independent random perturbation, stronger on lower body.
    """
    distorted = motion.copy()
    T, J, D = motion.shape
    # Stronger on lower body (joints 0,1,2,4,5,7,8,10,11)
    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    for j in lower_body:
        jitter = np.random.normal(0, 0.05, (T, D))  # 5cm jitter
        distorted[:, j, :] += jitter
    return distorted


def add_pelvis_drift(motion):
    """
    Pelvis horizontal drift — simulates VQ cumulative error propagating
    down the kinematic chain. Pelvis shifts, rest follows.
    """
    distorted = motion.copy()
    T = motion.shape[0]
    # Smooth random walk for pelvis XZ
    steps_x = np.random.normal(0, 0.015, T).cumsum()  # cum ~0.15m over 100f
    steps_z = np.random.normal(0, 0.015, T).cumsum()
    for t in range(T):
        distorted[t, 0, 0] += steps_x[t]  # Pelvis X
        distorted[t, 0, 2] += steps_z[t]  # Pelvis Z
    return distorted


def add_combined_heavy(motion):
    """Apply multiple heavy distortions together — worst-case simulation."""
    distorted = motion.copy()
    # Apply 2-3 distortions sequentially
    funcs = [
        lambda m: add_foot_skating_heavy(m, intensity=np.random.uniform(0.8, 1.5)),
        add_foot_drift_heavy,
        add_pelvis_drift,
        add_joint_jitter,
        add_y_shift_heavy,
    ]
    np.random.shuffle(funcs)
    for f in funcs[:np.random.randint(2, 4)]:
        distorted = f(distorted)
    return distorted


# ================================================================
#  Main
# ================================================================

def main():
    print("=" * 60)
    print("MotionFix V13 — Prepare Training Data (Amplified Noise)")
    print("  V8 architecture + 3-5x noise for stronger corrections")
    print("=" * 60)

    # ---- Load unique clean targets from V8's training_data_v2 ----
    # V8 stored clean targets at every 3rd file (AUGMENT_PER_SAMPLE=3)
    all_targets = sorted(glob.glob(f"{SOURCE_DATA_DIR}/target_*.npy"))
    unique_targets = all_targets[::V8_AUGMENT_PER_SAMPLE]
    print(f"Training data V2: {len(all_targets)} total targets")
    print(f"Unique motions:   {len(unique_targets)}")

    # Limit to NUM_SAMPLES
    unique_targets = unique_targets[:NUM_SAMPLES]
    print(f"Using:            {len(unique_targets)} motions")

    clean_motions = []
    for path in unique_targets:
        m = np.load(path).astype(np.float32)  # (T, 22, 3)
        clean_motions.append(m)
    print(f"Loaded: {len(clean_motions)} clean motions")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Distortion function list — biased toward heavy skating
    # V8: 40% skating, 20% drift, 20% smooth, 10% y_shift, 10% noise
    # V13: 50% heavy skating variants, rest on other distortions
    distortion_funcs = [
        ('foot_skating_h', add_foot_skating_heavy),        # 20%
        ('foot_skating_h', add_foot_skating_heavy),
        ('foot_skating_vh', add_foot_skating_very_heavy),  # 15%
        ('foot_skating_vh', add_foot_skating_very_heavy),
        ('foot_drift_h', add_foot_drift_heavy),             # 15%
        ('foot_drift_h', add_foot_drift_heavy),
        ('combined_h', add_combined_heavy),                 # 15%
        ('combined_h', add_combined_heavy),
        ('temporal_smooth', add_temporal_smoothing),        # 10%
        ('temporal_smooth', add_temporal_smoothing),
        ('y_shift_h', add_y_shift_heavy),                  # 10%
        ('y_shift_h', add_y_shift_heavy),
        ('spatial_noise_h', add_spatial_noise_heavy),       # 5%
        ('joint_jitter', add_joint_jitter),                 # 5%
        ('pelvis_drift', add_pelvis_drift),                 # 5%
    ]

    print(f"\nGenerating training pairs...")
    print(f"  {len(clean_motions)} motions × {AUGMENT_PER_SAMPLE} augments = "
          f"{len(clean_motions) * AUGMENT_PER_SAMPLE} pairs")
    pair_count = 0

    for i, real_motion in enumerate(clean_motions):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(clean_motions)}, pairs: {pair_count}")

        for _ in range(AUGMENT_PER_SAMPLE):
            dist_name, dist_func = distortion_funcs[
                np.random.randint(len(distortion_funcs))
            ]
            distorted = dist_func(real_motion)

            np.save(f"{OUTPUT_DIR}/distorted_{pair_count:06d}.npy", distorted)
            np.save(f"{OUTPUT_DIR}/target_{pair_count:06d}.npy", real_motion)
            pair_count += 1

    print(f"\nDone! Training pairs: {pair_count}")
    print(f"Output: {OUTPUT_DIR}/")

    # ---- Validate distortion magnitude ----
    print("\nValidation — distortion magnitude (on first motion):")
    sample = clean_motions[0]
    for name, func in [
        ('foot_skating_h', add_foot_skating_heavy),
        ('foot_skating_vh', add_foot_skating_very_heavy),
        ('foot_drift_h', add_foot_drift_heavy),
        ('temporal_smooth', add_temporal_smoothing),
        ('y_shift_h', add_y_shift_heavy),
        ('spatial_noise_h', add_spatial_noise_heavy),
        ('joint_jitter', add_joint_jitter),
        ('pelvis_drift', add_pelvis_drift),
        ('combined_h', add_combined_heavy),
    ]:
        dist = func(sample)
        diff = np.abs(dist - sample).mean()
        max_diff = np.abs(dist - sample).max()
        print(f"  {name:20s}: mean_diff={diff:.4f}, max_diff={max_diff:.4f}")

    # ---- Compare with V8 ----
    print(f"\n  Expected lower body mean diff: ~0.15-0.30m")
    print(f"  V8 lower body mean diff:       ~0.02m")
    print(f"  V12.0 lower body mean diff:    ~0.02m")
    print(f"  V12.1 lower body mean diff:    ~0.25m")
    print(f"  MoMask artifact:               ~0.4m")


if __name__ == "__main__":
    main()
