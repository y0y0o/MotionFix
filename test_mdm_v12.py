"""
MotionFix V12 -> MDM (Motion Diffusion Model) 测试

测试 V12.1 模型在 MDM 扩散模型输出上的脚滑修复效果。
MDM 是 diffusion-based，MoMask 是 VQ-based — 对比 V12 在不同类型生成器上的表现。
"""

import torch
import numpy as np
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'motionfix_v12'))
from motionfix_model_v12 import MotionFixNetworkV12


def compute_contact_labels(motion, foot_joints=(7, 8),
                           height_thresh=0.05, vel_thresh=0.5):
    T = motion.shape[0]
    labels = np.zeros((T, 2), dtype=np.float32)
    for i, fj in enumerate(foot_joints):
        foot_y = motion[:, fj, 1]
        ground = np.percentile(foot_y, 5)
        threshold = ground + height_thresh
        for t in range(T):
            if foot_y[t] < threshold:
                if t > 0:
                    vel = np.linalg.norm(motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]])
                    if vel < vel_thresh:
                        labels[t, i] = 1.0
                else:
                    labels[t, i] = 1.0
    return labels


def compute_fsr(motion, foot_joints=(7, 8), vel_thresh=0.03):
    contact = compute_contact_labels(motion, foot_joints)
    T = motion.shape[0]
    skating = 0
    contact_count = 0
    for i, fj in enumerate(foot_joints):
        for t in range(1, T):
            if contact[t, i] > 0.5:
                contact_count += 1
                vel = np.linalg.norm(motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]])
                if vel > vel_thresh:
                    skating += 1
    if contact_count == 0:
        return 0.0, 0, 0
    return skating / contact_count, contact_count, skating


def compute_jitter(motion):
    foot_joints = [7, 8, 10, 11]
    foot_motion = motion[:, foot_joints, :]
    vel = foot_motion[1:] - foot_motion[:-1]
    acc = vel[1:] - vel[:-1]
    return np.sqrt((acc ** 2).mean())


def compute_mean_foot_error(original, fixed):
    foot_joints = [7, 8, 10, 11]
    diffs = []
    for fj in foot_joints:
        diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
        diffs.append(diff)
    return np.mean(diffs)


def fix_motion(model, motion, device='cuda'):
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1).astype(np.float32)
    motion_tensor = torch.from_numpy(motion_flat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        fixed_tensor = model(motion_tensor, foot_only=True)
    fixed_flat = fixed_tensor.squeeze(0).cpu().numpy()
    fixed = fixed_flat.reshape(T, 22, 3)
    return fixed


def main():
    print("=" * 60)
    print("MotionFix V12.1 -> MDM (Diffusion) 测试")
    print("  Model: V12.1 (FRDM-inspired, blend_alpha=0.7)")
    print("  Data:  MDM diffusion generated motions (50 prompts)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load V12 model
    model = MotionFixNetworkV12(
        blend_alpha=0.7,
        window_size=2,
        velocity_gate=1.0,
    ).to(device)
    ckpt_path = "motionfix_v12/checkpoints_v12/best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded V12.1: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

    # Find MDM test files
    mdm_dir = "mdm_raw_joints"
    mdm_files = sorted(glob.glob(f"{mdm_dir}/mdm_*_joints.npy"))

    if len(mdm_files) == 0:
        print(f"\nNo MDM files found in {mdm_dir}/")
        return

    output_dir = "fixed_outputs_mdm_v12"
    os.makedirs(output_dir, exist_ok=True)

    header = (f"{'Name':<40} | {'FSR_before':>9} | {'FSR_after':>9} | "
              f"{'Jitter_before':>9} | {'Jitter_after':>9} | {'FtErr(m)':>8}")
    print(f"\n{header}")
    print("-" * len(header))

    results = []

    for filepath in mdm_files:
        filename = os.path.basename(filepath)
        name = filename.replace('_joints.npy', '')

        motion = np.load(filepath).astype(np.float32)  # (T, 22, 3)

        fsr_before, _, _ = compute_fsr(motion)
        jitter_before = compute_jitter(motion)

        fixed = fix_motion(model, motion, device)

        fsr_after, _, _ = compute_fsr(fixed)
        jitter_after = compute_jitter(fixed)
        foot_error = compute_mean_foot_error(motion, fixed)

        print(f"{name[:40]:<40} | {fsr_before:>8.1%} | {fsr_after:>8.1%} | "
              f"{jitter_before:>9.4f} | {jitter_after:>9.4f} | {foot_error:>7.4f}")

        np.save(f"{output_dir}/{name}_fixed.npy", fixed)

        results.append({
            'name': name,
            'fsr_before': fsr_before,
            'fsr_after': fsr_after,
            'jitter_before': jitter_before,
            'jitter_after': jitter_after,
            'foot_error': foot_error,
        })

    # Summary
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

        max_jump = max(r['foot_error'] for r in results)
        max_jump_name = max(results, key=lambda r: r['foot_error'])['name']
        print(f"\nMax foot modification: {max_jump:.4f}m ({max_jump_name[:50]})")

    # Compare with V8 on MDM
    print(f"\n{'='*60}")
    print("MDM 对比: V8 vs V12.1")
    print(f"{'='*60}")
    print(f"  V8  on MoMask: FSR 14.1% → 11.2% (-2.9%)")
    print(f"  V12 on MoMask: FSR 14.1% → 16.7% (+2.7%)")
    print(f"  V12 on MDM:    FSR {avg_fsr_b:.1%} → {avg_fsr_a:.1%} ({avg_fsr_a-avg_fsr_b:+.1%})")
    print(f"\nSaved to: {output_dir}/")


if __name__ == "__main__":
    main()
