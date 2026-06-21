"""
MotionFix V12 - Prepare Training Data (FRDM-style Self-Supervised)

Key FRDM principles applied:
  1. SELF-SUPERVISED: Start with CLEAN HumanML3D motions, add synthetic noise
     → No need for paired artifact/clean data

  2. LOWER-BODY ONLY: Noise applied ONLY to lower body joints
     (pelvis, hips, knees, ankles, feet = joints 0,1,2,4,5,7,8,10,11)
     Upper body stays perfectly clean — training/inference alignment

  3. CONTACT-AWARE NOISE: Noise magnitude correlates with contact labels
     - Contact frames (foot on ground): STRONGER noise — simulates physics sim artifacts
     - Non-contact frames: WEAKER noise — preserves natural motion

  4. PHYSICS-SIM ARTIFACT SIMULATION:
     - High-frequency jitter → simulates incorrect friction modeling
     - Horizontal foot sliding on contact → simulates skating
     - Gaussian noise on knees/hips → simulates simulation drift
     - Temporal smoothing → simulates over-regularization

Output format: training_data_v12/
  distorted_XXXXXX.npy  — (T, 22, 3) lower-body-noised motion
  target_XXXXXX.npy     — (T, 22, 3) clean original motion
  contact_XXXXXX.npy    — (T, 2) foot-ground contact labels {0,1}
"""

import numpy as np
import glob
import os
import sys
import torch
from scipy.ndimage import uniform_filter1d

sys.path.append('/home3/nxkh91/Project/T2M-GPT')
from utils.motion_process import recover_from_ric

HUMANML3D_PATH = "/home3/nxkh91/HumanML3D/HumanML3D/new_joint_vecs"
MEAN_PATH = "/home3/nxkh91/motion-diffusion-model-main/dataset/t2m_mean.npy"
STD_PATH = "/home3/nxkh91/motion-diffusion-model-main/dataset/t2m_std.npy"
OUTPUT_DIR = "../training_data_v12"
NUM_SAMPLES = 5000
AUGMENT_PER_SAMPLE = 3
MAX_LENGTH = 300

# ---- Joint definitions (HumanML3D 22-joint skeleton) ----
# Lower body joints for perturbation (FRDM: root + knee + foot area)
LOWER_BODY_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]
#  0: pelvis    1: left_hip    2: right_hip
#  4: left_knee 5: right_knee
#  7: left_ankle 8: right_ankle  10: left_foot  11: right_foot

FOOT_JOINTS = [7, 8, 10, 11]       # ankles + feet — key for skating detection
ANKLE_JOINTS = [7, 8]              # ankles — for contact label computation
KNEE_JOINTS = [4, 5]
HIP_JOINTS = [1, 2]

# Noise magnitude ranges (FRDM-style, progressively stronger toward feet)
# V12.1: Scaled up 3-5x to match MoMask artifact magnitudes (~0.4m avg foot error)
NOISE_CONFIG = {
    'pelvis':  {'joints': [0],  'pos_std': (0.02, 0.08)},
    'hips':    {'joints': [1, 2], 'pos_std': (0.03, 0.12)},
    'knees':   {'joints': [4, 5], 'pos_std': (0.05, 0.20)},
    'ankles':  {'joints': [7, 8], 'pos_std': (0.10, 0.35)},
    'feet':    {'joints': [10, 11], 'pos_std': (0.15, 0.50)},
}


# ================================================================
#  Contact Label Computation
# ================================================================
def compute_contact_labels(motion, ankle_joints=ANKLE_JOINTS,
                           height_thresh=0.05, vel_thresh=0.5):
    """
    Compute foot-ground contact labels for a clean motion.

    motion: (T, 22, 3)  joint world coordinates
    returns: (T, 2)     binary labels [left, right]

    Criteria:
      - Foot height < ground_level + height_thresh (5cm)
      - Foot horizontal velocity < vel_thresh (0.5 m/frame ≈ stationary)
    """
    T = motion.shape[0]
    labels = np.zeros((T, 2), dtype=np.float32)

    for i, fj in enumerate(ankle_joints):
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


# ================================================================
#  FRDM-style Distortion Functions (lower-body only)
# ================================================================
def add_gaussian_noise_lower(motion, contact, scale=1.0):
    """
    Gaussian noise on lower body joints.
    Noise scaled by joint type (feet > ankles > knees > hips > pelvis).
    Contact frames get 1.5× extra noise on feet/ankles.

    This is the primary FRDM-style perturbation — simulates general
    physics simulation inaccuracy.
    """
    distorted = motion.copy()

    for region in NOISE_CONFIG.values():
        std_min, std_max = region['pos_std']
        std = np.random.uniform(std_min, std_max) * scale

        for j in region['joints']:
            noise = np.random.randn(motion.shape[0], 3).astype(np.float32) * std
            distorted[:, j, :] += noise

    # Extra noise on contact frames for feet and ankles (FRDM: contact-aware)
    for fj in FOOT_JOINTS:
        foot_idx = 0 if fj in [7, 10] else 1  # left=0, right=1
        for t in range(motion.shape[0]):
            if contact[t, foot_idx] > 0.5:
                # Contact frame: extra jitter (simulates physics sim friction error)
                # V12.1: scaled up from 0.02 to 0.08
                extra = np.random.randn(3).astype(np.float32) * 0.08 * scale
                distorted[t, fj, :] += extra

    return distorted


def add_foot_jitter(motion, contact, jitter_std=0.05):
    """
    High-frequency jitter on feet.
    V12.1: increased from 0.012 to 0.05 to match MoMask artifact scale.
    Simulates the "buzzing" artifact from physics simulation
    when friction coefficients don't match real surfaces.
    """
    distorted = motion.copy()
    T = motion.shape[0]

    for fj in FOOT_JOINTS:
        foot_idx = 0 if fj in [7, 10] else 1
        for t in range(T):
            if contact[t, foot_idx] > 0.5:
                # High-frequency: random sign flip per frame
                jitter = np.random.randn(3).astype(np.float32) * jitter_std
                jitter = jitter * np.random.choice([-1, 1])
                distorted[t, fj, :] += jitter

    return distorted


def add_foot_skating(motion, contact, skate_mag=0.15):
    """
    Horizontal foot sliding on contact frames.
    V12.1: increased from 0.03 to 0.15 to match MoMask skating artifact scale.
    Directly simulates the foot skating artifact from VQ-based generators.
    """
    distorted = motion.copy()
    T = motion.shape[0]

    for fj in FOOT_JOINTS:
        foot_idx = 0 if fj in [7, 10] else 1
        for t in range(T):
            if contact[t, foot_idx] > 0.5:
                # Random horizontal offset
                dx = np.random.randn() * skate_mag
                dz = np.random.randn() * skate_mag
                distorted[t, fj, 0] += dx  # X
                distorted[t, fj, 2] += dz  # Z

    return distorted


def add_temporal_smoothing_lower(motion, window_size=None):
    """
    Temporal smoothing on lower body.
    Simulates over-regularization / motion blur artifacts.
    """
    if window_size is None:
        window_size = np.random.randint(5, 12)

    distorted = motion.copy()
    for j in LOWER_BODY_JOINTS:
        for dim in range(3):
            distorted[:, j, dim] = uniform_filter1d(
                motion[:, j, dim], size=window_size, mode='nearest'
            )
    return distorted


def add_knee_drift(motion, drift_std=0.08):
    """
    Slow drift on knee positions — simulates physics simulation
    accumulating errors over time (common in IsaacGym PHC output).
    V12.1: increased from 0.02 to 0.08.
    """
    distorted = motion.copy()
    T = motion.shape[0]

    for fj in KNEE_JOINTS:
        drift_x = np.cumsum(np.random.randn(T).astype(np.float32) * drift_std / T)
        drift_z = np.cumsum(np.random.randn(T).astype(np.float32) * drift_std / T)
        distorted[:, fj, 0] += drift_x
        distorted[:, fj, 2] += drift_z

    return distorted


# ================================================================
#  Format Conversion
# ================================================================
def convert_to_joints(motion_263, mean, std):
    """Convert (T, 263) motion representation to (T, 22, 3) joint positions."""
    motion_denorm = motion_263 * std + mean
    motion_tensor = torch.from_numpy(motion_denorm).float().cuda()
    motion_joints = recover_from_ric(motion_tensor, 22)
    return motion_joints.cpu().numpy()


# ================================================================
#  Main
# ================================================================
def main():
    print("=" * 60)
    print("MotionFix V12 - Prepare Training Data (FRDM-style)")
    print("=" * 60)
    print(f"  Strategy: Self-supervised (clean data + synthetic noise)")
    print(f"  Perturbation: LOWER-BODY ONLY ({len(LOWER_BODY_JOINTS)} joints)")
    print(f"  Noise: Contact-aware (stronger on contact frames)")
    print(f"  Output: {NUM_SAMPLES * AUGMENT_PER_SAMPLE} training pairs")
    print("=" * 60)

    # Load normalization
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    print("Loaded normalization parameters")

    # Load HumanML3D files
    all_files = sorted(glob.glob(f"{HUMANML3D_PATH}/*.npy"))
    print(f"Found {len(all_files)} motion files in HumanML3D")

    use_files = all_files[:NUM_SAMPLES]
    print(f"Using {len(use_files)} files")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Convert to joint positions ----
    print("\nConverting (T, 263) → (T, 22, 3)...")
    valid_motions = []

    for i, filepath in enumerate(use_files):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(use_files)}, converted: {len(valid_motions)}")

        try:
            motion_263 = np.load(filepath)
            motion_joints = convert_to_joints(motion_263, mean, std)

            if motion_joints.shape[1] == 22 and motion_joints.shape[2] == 3:
                if motion_joints.shape[0] > MAX_LENGTH:
                    motion_joints = motion_joints[:MAX_LENGTH]
                valid_motions.append(motion_joints)
        except Exception as e:
            if i < 5:
                print(f"  Skip {os.path.basename(filepath)}: {e}")

    print(f"  Converted: {len(valid_motions)} valid motions")
    print(f"  Sample shape: {valid_motions[0].shape}")

    if len(valid_motions) == 0:
        print("ERROR: No valid motions!")
        sys.exit(1)

    # ---- Generate training pairs ----
    print(f"\nGenerating {AUGMENT_PER_SAMPLE} augmentations per motion...")

    # Distortion type distribution (FRDM-style)
    distortion_types = [
        ('gaussian',  add_gaussian_noise_lower),   # 40% — primary FRDM noise
        ('gaussian',  add_gaussian_noise_lower),
        ('jitter',    add_foot_jitter),             # 20% — physics sim jitter
        ('skating',   add_foot_skating),            # 20% — foot skating
        ('smooth',    add_temporal_smoothing_lower), # 10% — over-smoothing
        ('knee_drift', add_knee_drift),             # 10% — sim drift
    ]

    pair_count = 0
    stats = {name: 0 for name, _ in distortion_types}

    for i, real_motion in enumerate(valid_motions):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(valid_motions)}, pairs: {pair_count}")

        # Pre-compute contact labels from clean motion
        contact = compute_contact_labels(real_motion)

        for aug_idx in range(AUGMENT_PER_SAMPLE):
            # Random distortion type
            idx = np.random.randint(len(distortion_types))
            dist_name, dist_func = distortion_types[idx]

            if dist_name == 'gaussian':
                scale = np.random.uniform(0.7, 2.0)
                distorted = dist_func(real_motion, contact, scale=scale)
            elif dist_name == 'jitter':
                std = np.random.uniform(0.02, 0.10)
                distorted = dist_func(real_motion, contact, jitter_std=std)
            elif dist_name == 'skating':
                mag = np.random.uniform(0.05, 0.25)
                distorted = dist_func(real_motion, contact, skate_mag=mag)
            elif dist_name == 'smooth':
                distorted = dist_func(real_motion)
            elif dist_name == 'knee_drift':
                std = np.random.uniform(0.04, 0.15)
                distorted = dist_func(real_motion, drift_std=std)
            else:
                distorted = dist_func(real_motion, contact)

            # Verify: upper body should be EXACTLY preserved (FRDM key constraint)
            upper_body_joints = [j for j in range(22) if j not in LOWER_BODY_JOINTS]
            upper_diff = np.abs(
                distorted[:, upper_body_joints, :] -
                real_motion[:, upper_body_joints, :]
            ).max()
            if upper_diff > 1e-6:
                print(f"  WARNING: Upper body modified! max_diff={upper_diff:.6f}")
                # Fix: restore upper body
                distorted[:, upper_body_joints, :] = real_motion[:, upper_body_joints, :]

            # Save triplet
            np.save(f"{OUTPUT_DIR}/distorted_{pair_count:06d}.npy", distorted)
            np.save(f"{OUTPUT_DIR}/target_{pair_count:06d}.npy", real_motion)
            np.save(f"{OUTPUT_DIR}/contact_{pair_count:06d}.npy", contact)

            stats[dist_name] += 1
            pair_count += 1

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Done! Generated {pair_count} training pairs")
    print(f"  Files: {pair_count * 3} (distorted + target + contact)")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"\nDistortion distribution:")
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {name:15s}: {count:5d} ({100*count/pair_count:.1f}%)")

    # ---- Validate ----
    print(f"\nValidation — checking first sample:")
    d = np.load(f"{OUTPUT_DIR}/distorted_000000.npy")
    t = np.load(f"{OUTPUT_DIR}/target_000000.npy")
    c = np.load(f"{OUTPUT_DIR}/contact_000000.npy")

    lower_diff = np.abs(d[:, LOWER_BODY_JOINTS, :] - t[:, LOWER_BODY_JOINTS, :])
    upper_diff = np.abs(d[:, upper_body_joints, :] - t[:, upper_body_joints, :])

    print(f"  Lower body mean diff: {lower_diff.mean():.4f}  (should be > 0)")
    print(f"  Upper body mean diff: {upper_diff.mean():.10f}  (should be 0)")
    print(f"  Contact label mean:   {c.mean():.4f}  ({c.sum():.0f} contact frames)")

    if upper_diff.max() > 1e-6:
        print(f"  WARNING: Upper body was modified! max={upper_diff.max():.6f}")
    else:
        print(f"  ✓ Upper body perfectly preserved")


if __name__ == "__main__":
    main()
