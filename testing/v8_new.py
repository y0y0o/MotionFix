"""
V8 + Root-Relative Fix — batch test on all 50 MoMask prompts.
"""
import torch
import numpy as np
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.v8 import MotionFixNetwork

FOOT_JOINTS = [7, 8, 10, 11]


def compute_contact(motion, foot_joints=(7,8), h_thresh=0.05, v_thresh=0.5):
    T = motion.shape[0]; labels = np.zeros((T, 2), dtype=np.float32)
    for i, fj in enumerate(foot_joints):
        foot_y = motion[:, fj, 1]; ground = np.percentile(foot_y, 5); thresh = ground + h_thresh
        for t in range(T):
            if foot_y[t] < thresh:
                if t > 0:
                    vel = np.linalg.norm(motion[t, fj, [0,2]] - motion[t-1, fj, [0,2]])
                    if vel < v_thresh: labels[t, i] = 1.0
                else: labels[t, i] = 1.0
    return labels


def compute_fsr(motion):
    contact = compute_contact(motion); T = motion.shape[0]
    skating = 0; contact_count = 0
    for i, fj in enumerate([7,8]):
        for t in range(1, T):
            if contact[t, i] > 0.5:
                contact_count += 1
                vel = np.linalg.norm(motion[t, fj, [0,2]] - motion[t-1, fj, [0,2]])
                if vel > 0.03: skating += 1
    if contact_count == 0: return 0.0, 0, 0
    return skating / contact_count, contact_count, skating


def compute_jitter(motion):
    foot = motion[:, FOOT_JOINTS, :]
    vel = foot[1:] - foot[:-1]; acc = vel[1:] - vel[:-1]
    return float(np.sqrt((acc**2).mean()))


def compute_foot_error(a, b):
    return float(np.mean([np.linalg.norm(a[:, fj, :] - b[:, fj, :], axis=1).mean()
                          for fj in FOOT_JOINTS]))


def fix_motion(model, motion, device='cuda'):
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1).astype(np.float32)
    motion_tensor = torch.from_numpy(motion_flat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        fixed_tensor = model(motion_tensor, foot_only=True, root_relative=True)
    return fixed_tensor.squeeze(0).cpu().numpy().reshape(T, 22, 3)


def main():
    print("=" * 70)
    print("MotionFix V8 + Root-Relative Fix — MoMask 50 批量测试")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = MotionFixNetwork(blend_alpha=0.5).to(device)
    ckpt = torch.load("checkpoints/v8/best.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Model: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")
    print(f"Mode: root_relative=True (coordinate fix)")

    input_dir  = "data/test_inputs/momask_50/momask_50_results/no_ik"
    output_dir = "outputs/fixed/v8_new"
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(f"{input_dir}/*.npy"))
    print(f"\nFound {len(files)} MoMask files\n")

    header = (f"{'#':<3} {'Name':<48} {'FSR_bef':>7} {'FSR_aft':>7} "
              f"{'ΔFSR':>8} {'Jit_bef':>8} {'Jit_aft':>8} {'FtErr':>7}")
    print(header)
    print("-" * len(header))

    results = []

    for i, filepath in enumerate(files):
        name = os.path.basename(filepath).replace('.npy', '')
        short = name[:47]

        motion = np.load(filepath).astype(np.float32)
        if motion.ndim == 4: motion = motion[0]

        fsr_bef, _, _ = compute_fsr(motion)
        jit_bef = compute_jitter(motion)

        fixed = fix_motion(model, motion, device)

        fsr_aft, _, _ = compute_fsr(fixed)
        jit_aft = compute_jitter(fixed)
        ft_err = compute_foot_error(motion, fixed)

        np.save(f"{output_dir}/{name}_fixed.npy", fixed)

        print(f"{i+1:<3} {short:<48} {fsr_bef:>6.1%} {fsr_aft:>6.1%} "
              f"{fsr_aft-fsr_bef:>+7.1%} {jit_bef:>8.4f} {jit_aft:>8.4f} {ft_err:>6.4f}")

        results.append({
            'name': name, 'fsr_before': fsr_bef, 'fsr_after': fsr_aft,
            'jitter_before': jit_bef, 'jitter_after': jit_aft,
            'foot_error': ft_err,
        })

    # ── Summary ────────────────────────────────────────
    print("-" * len(header))

    avg_fsr_b = np.mean([r['fsr_before'] for r in results])
    avg_fsr_a = np.mean([r['fsr_after'] for r in results])
    avg_jit_b = np.mean([r['jitter_before'] for r in results])
    avg_jit_a = np.mean([r['jitter_after'] for r in results])
    avg_ft_err = np.mean([r['foot_error'] for r in results])

    print(f"{'':<3} {'AVERAGE (n=50)':<48} {avg_fsr_b:>6.1%} {avg_fsr_a:>6.1%} "
          f"{avg_fsr_a-avg_fsr_b:>+7.1%} {avg_jit_b:>8.4f} {avg_jit_a:>8.4f} {avg_ft_err:>6.4f}")

    # Compare with old V8
    print(f"\n{'=' * 70}")
    print("对比: V8 OLD (无修复) vs V8 NEW (root_relative)")
    print(f"{'=' * 70}")
    print(f"  V8 NEW FSR:    {avg_fsr_a:.1%}  (原始 {avg_fsr_b:.1%} → {avg_fsr_a:.1%}, Δ={avg_fsr_a-avg_fsr_b:+.1%})")
    print(f"  V8 NEW Jitter: {avg_jit_a:.4f}  (原始 {avg_jit_b:.4f})")
    print(f"  V8 NEW FootErr: {avg_ft_err:.4f}m")
    print(f"  V8 OLD FSR:    16.6% (p000021 only)")
    print(f"  V8 OLD Jitter: 0.267  (p000021 only)")
    print(f"\n  Saved: {output_dir}/ ({len(files)} files)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
