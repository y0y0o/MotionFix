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
OUTPUT_DIR = "training_data_v2"
NUM_SAMPLES = 5000
AUGMENT_PER_SAMPLE = 3


def add_foot_skating(motion, intensity=0.3):
    """
    直接模拟滑步：当脚在地面时，加入水平位移
    这是最重要的失真！
    """
    distorted = motion.copy()
    T = motion.shape[0]

    for foot_idx in [10, 11]:  # 左右脚趾
        heights = motion[:, foot_idx, 1]

        # 找到最低点作为地面
        ground = np.percentile(heights, 5)
        contact_threshold = ground + 0.05

        for t in range(T - 1):
            if heights[t] < contact_threshold:
                # 脚在地面时，加入随机水平位移（模拟滑步）
                dx = np.random.uniform(-intensity, intensity) * 0.1
                dz = np.random.uniform(-intensity, intensity) * 0.1
                distorted[t, foot_idx, 0] += dx
                distorted[t, foot_idx, 2] += dz

    return distorted


def add_temporal_smoothing_strong(motion):
    """时间平滑 - 加大窗口"""
    distorted = motion.copy()
    window = np.random.randint(7, 15)  # 更大的窗口

    for joint in range(motion.shape[1]):
        for dim in range(3):
            distorted[:, joint, dim] = uniform_filter1d(
                motion[:, joint, dim], size=window, mode='nearest'
            )
    return distorted


def add_foot_drift(motion):
    """
    脚部漂移：接触地面时脚缓慢滑动
    更接近VQ模型的真实错误
    """
    distorted = motion.copy()
    T = motion.shape[0]

    for foot_idx in [10, 11]:
        heights = motion[:, foot_idx, 1]
        ground = np.percentile(heights, 5)
        contact_threshold = ground + 0.05

        # 找到接触段
        in_contact = False
        drift_x = 0
        drift_z = 0

        for t in range(T):
            if heights[t] < contact_threshold:
                if not in_contact:
                    # 开始新的接触段，随机漂移方向
                    drift_x = np.random.uniform(-0.01, 0.01)
                    drift_z = np.random.uniform(-0.01, 0.01)
                    in_contact = True

                # 累积漂移
                distorted[t, foot_idx, 0] += drift_x * (t % 20)
                distorted[t, foot_idx, 2] += drift_z * (t % 20)
            else:
                in_contact = False

    return distorted


def add_y_shift(motion):
    """Y轴偏移 - 加大范围"""
    distorted = motion.copy()
    shift = np.random.uniform(-0.08, 0.08)  # 更大偏移
    distorted[:, :, 1] += shift
    return distorted


def add_spatial_noise(motion):
    """空间噪声 - 加大"""
    noise_std = np.random.uniform(0.01, 0.05)  # 更大噪声
    noise = np.random.normal(0, noise_std, motion.shape)
    return motion + noise


def convert_to_joints(motion_263, mean, std):
    motion_denorm = motion_263 * std + mean
    motion_tensor = torch.from_numpy(motion_denorm).float().cuda()
    motion_joints = recover_from_ric(motion_tensor, 22)
    return motion_joints.cpu().numpy()


def main():
    print("=" * 60)
    print("MotionFix - Prepare Training Data V2")
    print("Stronger distortions for better correction")
    print("=" * 60)

    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)

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
        except:
            pass

    print(f"  Converted: {len(valid_motions)} motions")

    # 失真函数列表（重点是滑步相关的）
    distortion_funcs = [
        ('foot_skating', add_foot_skating),       # 40% 概率
        ('foot_skating', add_foot_skating),
        ('foot_drift', add_foot_drift),            # 20% 概率
        ('temporal_smooth', add_temporal_smoothing_strong),  # 20% 概率
        ('y_shift', add_y_shift),                  # 10% 概率
        ('noise', add_spatial_noise),              # 10% 概率
    ]

    print(f"\nGenerating training pairs...")
    pair_count = 0

    for i, real_motion in enumerate(valid_motions):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(valid_motions)}, pairs: {pair_count}")

        for _ in range(AUGMENT_PER_SAMPLE):
            # 随机选失真
            dist_name, dist_func = distortion_funcs[
                np.random.randint(len(distortion_funcs))
            ]
            distorted = dist_func(real_motion)

            np.save(f"{OUTPUT_DIR}/distorted_{pair_count:06d}.npy", distorted)
            np.save(f"{OUTPUT_DIR}/target_{pair_count:06d}.npy", real_motion)
            pair_count += 1

    print(f"\nDone! Training pairs: {pair_count}")
    print(f"Output: {OUTPUT_DIR}/")

    # 验证失真效果
    print("\nValidation - checking distortion magnitude:")
    sample_real = valid_motions[0]

    for name, func in [('foot_skating', add_foot_skating),
                        ('foot_drift', add_foot_drift),
                        ('temporal_smooth', add_temporal_smoothing_strong),
                        ('y_shift', add_y_shift),
                        ('noise', add_spatial_noise)]:
        dist = func(sample_real)
        diff = np.abs(dist - sample_real).mean()
        max_diff = np.abs(dist - sample_real).max()
        print(f"  {name:20s}: mean_diff={diff:.4f}, max_diff={max_diff:.4f}")


if __name__ == "__main__":
    main()
