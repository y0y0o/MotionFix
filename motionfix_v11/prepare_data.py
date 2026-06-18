"""
MotionFix V10 - Prepare Training Data

改进点（对标 FRDM）:
  1. 只对下半身关节加扰动，上半身保持干净 → 训推一致
  2. 预计算脚-地接触标签 b，存入训练数据
  3. 更具物理意义的瑕疵模拟:
     - 脚部抖动 (高频小振幅，模拟物理仿真产物)
     - 脚滑 (接触帧加水平位移)
     - 高斯噪声 (仅下半身)
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
OUTPUT_DIR = "training_data_v10"
NUM_SAMPLES = 5000
AUGMENT_PER_SAMPLE = 3

# ---- 下肢 & 脚部关节定义 (HumanML3D 22关节) ----
#  0: pelvis, 1: left_hip, 2: right_hip, 4: left_knee, 5: right_knee,
#  7: left_ankle, 8: right_ankle, 10: left_foot, 11: right_foot
LOWER_BODY_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]
FOOT_JOINTS = [7, 8, 10, 11]                     # 脚踝 + 脚掌


# ================================================================
#  接触标签计算
# ================================================================
def compute_contact_labels(motion, foot_joints=FOOT_JOINTS,
                           height_thresh=0.05, vel_thresh=0.5):
    """
    motion: (T, 22, 3)  关节世界坐标
    返回:   (T, 2)      左右脚着地标签 {0, 1}

    判定逻辑: 脚的高度 < 全局最低点 + height_thresh
              且 脚的速度 < vel_thresh
    """
    T = motion.shape[0]
    labels = np.zeros((T, 2), dtype=np.float32)

    for i, fj in enumerate(foot_joints[:2]):       # 7=左脚踝, 8=右脚踝
        foot_y = motion[:, fj, 1]
        ground = np.percentile(foot_y, 5)
        threshold = ground + height_thresh

        for t in range(T):
            if foot_y[t] < threshold:
                if t > 0:
                    vel = np.linalg.norm(motion[t, fj, [0, 2]]
                                         - motion[t-1, fj, [0, 2]])
                    if vel < vel_thresh:
                        labels[t, i] = 1.0
                else:
                    labels[t, i] = 1.0
    return labels


# ================================================================
#  瑕疵模拟（只作用于下半身）
# ================================================================
def add_foot_jitter(motion, jitter_std=0.008):
    """
    高频小振幅抖动 — 模拟物理仿真因摩擦不准产生的脚部抖动
    只作用于下半身关节
    """
    distorted = motion.copy()
    T = motion.shape[0]
    for j in LOWER_BODY_JOINTS:
        jitter = np.random.randn(T, 3).astype(np.float32) * jitter_std
        # 高频: 相邻帧符号随机翻转
        jitter = jitter * np.random.choice([-1, 1], size=(T, 1))
        distorted[:, j, :] += jitter
    return distorted


def add_foot_skating(motion, contact, skate_std=0.015):
    """
    脚滑模拟: 在接触帧 (b=1) 给脚的水平位置加随机偏移
    contact: (T, 2)
    """
    distorted = motion.copy()
    T = motion.shape[0]
    for i, fj in enumerate(FOOT_JOINTS[:2]):       # 7, 8 脚踝
        for t in range(T):
            if contact[t, i] > 0.5:
                offset = np.random.randn(2).astype(np.float32) * skate_std
                distorted[t, fj, 0] += offset[0]   # X
                distorted[t, fj, 2] += offset[1]   # Z
    return distorted


def add_spatial_noise_lower(motion, noise_std=0.01):
    """高斯噪声，只加在下半身"""
    distorted = motion.copy()
    for j in LOWER_BODY_JOINTS:
        noise = np.random.randn(motion.shape[0], 3).astype(np.float32) * noise_std
        distorted[:, j, :] += noise
    return distorted


def add_temporal_smoothing_lower(motion, window_size=5):
    """时序平滑，只作用在下半身"""
    distorted = motion.copy()
    for j in LOWER_BODY_JOINTS:
        for dim in range(3):
            distorted[:, j, dim] = uniform_filter1d(
                motion[:, j, dim], size=window_size, mode='nearest'
            )
    return distorted


# ================================================================
#  格式转换
# ================================================================
def convert_to_joints(motion_263, mean, std):
    motion_denorm = motion_263 * std + mean
    motion_tensor = torch.from_numpy(motion_denorm).float().cuda()
    motion_joints = recover_from_ric(motion_tensor, 22)
    return motion_joints.cpu().numpy()


# ================================================================
#  主流程
# ================================================================
def main():
    print("=" * 60)
    print("MotionFix V10 - Prepare Training Data")
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
                # 限制长度，避免超长序列
                if motion_joints.shape[0] > 300:
                    motion_joints = motion_joints[:300]
                valid_motions.append(motion_joints)
        except Exception as e:
            if i < 5:
                print(f"  Skip {os.path.basename(filepath)}: {e}")

    print(f"  Converted: {len(valid_motions)} motions")

    if len(valid_motions) == 0:
        print("No valid data!")
        sys.exit(1)

    print(f"\nSample shape: {valid_motions[0].shape}")

    print(f"\nGenerating training pairs ({AUGMENT_PER_SAMPLE} per motion)...")
    pair_count = 0

    for i, real_motion in enumerate(valid_motions):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(valid_motions)}, pairs: {pair_count}")

        # 每段动作预计算一次接触标签
        contact = compute_contact_labels(real_motion)

        for _ in range(AUGMENT_PER_SAMPLE):
            dist_type = np.random.choice(['jitter', 'skating', 'noise', 'smooth'])

            if dist_type == 'jitter':
                distorted = add_foot_jitter(real_motion,
                                            jitter_std=np.random.uniform(0.003, 0.015))
            elif dist_type == 'skating':
                distorted = add_foot_skating(real_motion, contact,
                                             skate_std=np.random.uniform(0.005, 0.025))
            elif dist_type == 'smooth':
                window = np.random.randint(3, 8)
                distorted = add_temporal_smoothing_lower(real_motion, window)
            else:
                distorted = add_spatial_noise_lower(real_motion,
                                                    noise_std=np.random.uniform(0.005, 0.02))

            np.save(f"{OUTPUT_DIR}/distorted_{pair_count:06d}.npy", distorted)
            np.save(f"{OUTPUT_DIR}/target_{pair_count:06d}.npy", real_motion)
            np.save(f"{OUTPUT_DIR}/contact_{pair_count:06d}.npy", contact)
            pair_count += 1

    print(f"\nDone!")
    print(f"  Training pairs: {pair_count}")
    print(f"  Total files: {pair_count * 3}  (distorted + target + contact)")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"\n  Distortion types: jitter(foot高频抖动) / skating(脚滑)")
    print(f"                    smooth(时序平滑) / noise(高斯噪声)")
    print(f"  All distortions ONLY on lower body joints: {LOWER_BODY_JOINTS}")
    print(f"  Contact labels saved for foot joints: {FOOT_JOINTS[:2]}")


if __name__ == "__main__":
    main()
