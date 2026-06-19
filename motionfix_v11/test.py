"""
MotionFix V11 (Fixed) - Test Script

修复要点:
  - 推理: foot_only=True, 只在滑步帧替换脚部（V8 机制）
  - 去掉 temporal_smooth 后处理（之前是二次伤害）
  - 评估脚滑率 (FSR) 和抖动 (Jitter) 的前后对比
"""

import torch
import numpy as np
import glob
import os

from motionfix_model import MotionFixNetwork


# ================================================================
#  指标计算
# ================================================================
def compute_contact_labels(motion, foot_joints=(7, 8),
                           height_thresh=0.05, vel_thresh=0.5):
    """
    motion: (T, 22, 3) 关节世界坐标
    返回:   (T, 2) 脚着地标签
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
                    vel = np.linalg.norm(motion[t, fj, [0, 2]]
                                         - motion[t-1, fj, [0, 2]])
                    if vel < vel_thresh:
                        labels[t, i] = 1.0
                else:
                    labels[t, i] = 1.0
    return labels


def compute_fsr(motion, foot_joints=(7, 8), vel_thresh=0.03):
    """
    Foot Skating Ratio: 脚着地时滑动的帧占比
    """
    contact = compute_contact_labels(motion, foot_joints)
    T = motion.shape[0]
    skating = 0
    contact_count = 0

    for i, fj in enumerate(foot_joints):
        for t in range(1, T):
            if contact[t, i] > 0.5:
                contact_count += 1
                vel = np.linalg.norm(motion[t, fj, [0, 2]]
                                     - motion[t-1, fj, [0, 2]])
                if vel > vel_thresh:
                    skating += 1

    if contact_count == 0:
        return 0.0, 0, 0
    return skating / contact_count, contact_count, skating


def compute_jitter(motion):
    """
    抖动指标: 加速度 RMS（越高=越抖）
    """
    vel = motion[1:] - motion[:-1]
    acc = vel[1:] - vel[:-1]
    return np.sqrt((acc ** 2).mean())


# ================================================================
#  修复函数
# ================================================================
def fix_motion(model, motion, device='cuda'):
    """
    motion: (T, 22, 3)  numpy
    返回:   (T, 22, 3)  numpy — 选择性替换脚部后的动作

    V11 Fixed: 使用 foot_only=True (selective replace),
    不再做 temporal_smooth 后处理。
    """
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1).astype(np.float32)        # (T, 66)
    motion_tensor = torch.from_numpy(motion_flat).unsqueeze(0).to(device)  # (1, T, 66)

    model.eval()
    with torch.no_grad():
        # 推理模式: 只在滑步帧替换脚部
        fixed_tensor = model(motion_tensor, foot_only=True)

    fixed_flat = fixed_tensor.squeeze(0).cpu().numpy()            # (T, 66)
    fixed = fixed_flat.reshape(T, 22, 3)

    return fixed


# ================================================================
#  主流程
# ================================================================
def main():
    print("=" * 60)
    print("MotionFix V11 (Fixed) - Test")
    print("  Selective foot replacement (V8 mechanism)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ---- 加载模型 ----
    model = MotionFixNetwork(blend_alpha=0.5).to(device)
    ckpt_path = "checkpoints_v11/best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Run train.py first.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

    # ---- 找测试文件 ----
    momask_dir = "../momask_50_results/no_ik"
    momask_files = sorted(glob.glob(f"{momask_dir}/*.npy"))

    if len(momask_files) == 0:
        print(f"\nNo MoMask files found in {momask_dir}")
        print("Skipping MoMask test.")
        return

    output_dir = "fixed_outputs_v11"
    os.makedirs(output_dir, exist_ok=True)

    # ---- 逐文件测试 ----
    header = (f"{'Name':<40} | {'FSR_before':>9} | {'FSR_after':>9} | "
              f"{'Jitter_before':>9} | {'Jitter_after':>9}")
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

        print(f"{name[:40]:<40} | {fsr_before:>8.1%} | {fsr_after:>8.1%} | "
              f"{jitter_before:>9.4f} | {jitter_after:>9.4f}")

        # 保存原始和修复后
        np.save(f"{output_dir}/{filename}", motion)
        np.save(f"{output_dir}/{filename.replace('.npy', '_fixed.npy')}", fixed)

        results.append({
            'name': name,
            'fsr_before': fsr_before,
            'fsr_after': fsr_after,
            'jitter_before': jitter_before,
            'jitter_after': jitter_after,
        })

    # ---- 汇总 ----
    if results:
        avg_fsr_b = np.mean([r['fsr_before'] for r in results])
        avg_fsr_a = np.mean([r['fsr_after'] for r in results])
        avg_jit_b = np.mean([r['jitter_before'] for r in results])
        avg_jit_a = np.mean([r['jitter_after'] for r in results])

        print("-" * len(header))
        print(f"{'AVERAGE':<40} | {avg_fsr_b:>8.1%} | {avg_fsr_a:>8.1%} | "
              f"{avg_jit_b:>9.4f} | {avg_jit_a:>9.4f}")
        print(f"{'CHANGE':<40} | {'':>9} | {avg_fsr_a-avg_fsr_b:>+8.1%} | "
              f"{'':>9} | {avg_jit_a-avg_jit_b:>+9.4f}")

    print(f"\nSaved to: {output_dir}/")


if __name__ == "__main__":
    main()
