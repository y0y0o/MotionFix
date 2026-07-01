import numpy as np
import glob
import os
import sys
import torch
from scipy.ndimage import uniform_filter1d

sys.path.append('/home3/nxkh91/projects/T2M-GPT')
from utils.motion_process import recover_from_ric

HUMANML3D_PATH = "/home3/nxkh91/projects/HumanML3D/HumanML3D/new_joint_vecs"
MEAN_PATH = "/home3/nxkh91/projects/mdm/dataset/t2m_mean.npy"
STD_PATH = "/home3/nxkh91/projects/mdm/dataset/t2m_std.npy"
OUTPUT_DIR = "data/training/v1"
NUM_SAMPLES = 5000
AUGMENT_PER_SAMPLE = 3


def add_temporal_smoothing(motion, window_size=5):
    distorted = motion.copy()
    for joint in range(motion.shape[1]):
        for dim in range(3):
            distorted[:, joint, dim] = uniform_filter1d(
                motion[:, joint, dim], size=window_size, mode='nearest'
            )
    return distorted


def add_y_shift(motion):
    distorted = motion.copy()
    shift = np.random.uniform(-0.05, 0.05)
    distorted[:, :, 1] += shift
    return distorted


def add_spatial_noise(motion):
    noise_std = np.random.uniform(0.005, 0.02)
    noise = np.random.normal(0, noise_std, motion.shape)
    return motion + noise


def convert_to_joints(motion_263, mean, std):
    motion_denorm = motion_263 * std + mean
    motion_tensor = torch.from_numpy(motion_denorm).float().cuda()
    motion_joints = recover_from_ric(motion_tensor, 22)
    return motion_joints.cpu().numpy()


def main():
    print("=" * 60)
    print("MotionFix - Prepare Training Data")
    print("=" * 60)

    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    print("Loaded normalization parameters")

    all_files = sorted(glob.glob(f"{HUMANML3D_PATH}/*.npy"))
    print(f"Found {len(all_files)} motion files")

    use_files = all_files[:NUM_SAMPLES]
    print(f"Using {len(use_files)} files")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nConverting (T, 263) -> (T, 22, 3)...")
    valid_motions = []

    for i, filepath in enumerate(use_files):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(use_files)}, converted: {len(valid_motions)}")

        try:
            motion_263 = np.load(filepath)
            motion_joints = convert_to_joints(motion_263, mean, std)

            if motion_joints.shape[1] == 22 and motion_joints.shape[2] == 3:
                valid_motions.append(motion_joints)
        except Exception as e:
            if i < 5:
                print(f"  Skip {os.path.basename(filepath)}: {e}")

    print(f"  Converted: {len(valid_motions)} motions")

    if len(valid_motions) == 0:
        print("No valid data!")
        sys.exit(1)

    print(f"\nSample shape: {valid_motions[0].shape}")
    print(f"Y range: [{valid_motions[0][:,:,1].min():.3f}, {valid_motions[0][:,:,1].max():.3f}]")

    print(f"\nGenerating training pairs ({AUGMENT_PER_SAMPLE} per motion)...")
    pair_count = 0

    for i, real_motion in enumerate(valid_motions):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(valid_motions)}, pairs: {pair_count}")

        for _ in range(AUGMENT_PER_SAMPLE):
            dist_type = np.random.choice(['smooth', 'shift', 'noise'])

            if dist_type == 'smooth':
                window = np.random.randint(3, 8)
                distorted = add_temporal_smoothing(real_motion, window)
            elif dist_type == 'shift':
                distorted = add_y_shift(real_motion)
            else:
                distorted = add_spatial_noise(real_motion)

            np.save(f"{OUTPUT_DIR}/distorted_{pair_count:06d}.npy", distorted)
            np.save(f"{OUTPUT_DIR}/target_{pair_count:06d}.npy", real_motion)
            pair_count += 1

    print(f"\nDone!")
    print(f"  Training pairs: {pair_count}")
    print(f"  Total files: {pair_count * 2}")
    print(f"  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()