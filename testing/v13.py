"""
MotionFix V13 — Test Script

Tests V13 on both MoMask (VQ) and MDM (Diffusion) generated motions.
"""
import torch
import numpy as np
import glob
import os
import sys

from models.v13 import MotionFixNetworkV13


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
    return fixed_flat.reshape(T, 22, 3)


def test_dataset(model, device, data_dir, file_pattern, output_dir, label):
    """Test one dataset and return results."""
    files = sorted(glob.glob(f"{data_dir}/{file_pattern}"))
    if not files:
        print(f"\n  No files found for {label} in {data_dir}/")
        return []

    os.makedirs(output_dir, exist_ok=True)

    header = (f"{'Name':<40} | {'FSR_before':>9} | {'FSR_after':>9} | "
              f"{'Jitter_before':>9} | {'Jitter_after':>9} | {'FtErr(m)':>8}")
    print(f"\n  {label} ({len(files)} motions)")
    print(f"  {header}")
    print(f"  {'-' * (len(header)-2)}")

    results = []
    for filepath in files:
        filename = os.path.basename(filepath)
        name = filename.replace('.npy', '').replace('_joints', '')

        motion = np.load(filepath).astype(np.float32)
        fsr_before, _, _ = compute_fsr(motion)
        jitter_before = compute_jitter(motion)

        fixed = fix_motion(model, motion, device)

        fsr_after, _, _ = compute_fsr(fixed)
        jitter_after = compute_jitter(fixed)
        foot_error = compute_mean_foot_error(motion, fixed)

        print(f"  {name[:40]:<40} | {fsr_before:>8.1%} | {fsr_after:>8.1%} | "
              f"{jitter_before:>9.4f} | {jitter_after:>9.4f} | {foot_error:>7.4f}")

        np.save(f"{output_dir}/{name}_fixed.npy", fixed)
        results.append({
            'name': name, 'fsr_before': fsr_before, 'fsr_after': fsr_after,
            'jitter_before': jitter_before, 'jitter_after': jitter_after,
            'foot_error': foot_error,
        })

    if results:
        avg_b = np.mean([r['fsr_before'] for r in results])
        avg_a = np.mean([r['fsr_after'] for r in results])
        avg_jb = np.mean([r['jitter_before'] for r in results])
        avg_ja = np.mean([r['jitter_after'] for r in results])
        avg_fe = np.mean([r['foot_error'] for r in results])
        print(f"  {'-' * (len(header)-2)}")
        print(f"  {'AVERAGE':<40} | {avg_b:>8.1%} | {avg_a:>8.1%} | "
              f"{avg_jb:>9.4f} | {avg_ja:>9.4f} | {avg_fe:>7.4f}")
        print(f"  {'CHANGE':<40} | {'':>9} | {avg_a-avg_b:>+8.1%} | "
              f"{'':>9} | {avg_ja-avg_jb:>+9.4f} |")

    return results


def main():
    print("=" * 60)
    print("MotionFix V13 — Test (V8 Architecture + Amplified Noise)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    model = MotionFixNetworkV13(blend_alpha=0.5).to(device)
    ckpt_path = "checkpoints/v13/best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found. Train first.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test 1: MoMask (VQ-based)
    momask_results = test_dataset(
        model, device,
        data_dir="data/test_inputs/momask_50/no_ik",
        file_pattern="*.npy",
        output_dir="outputs/fixed/v13_momask",
        label="MoMask (VQ)"
    )

    # Test 2: MDM (Diffusion-based)
    mdm_results = test_dataset(
        model, device,
        data_dir="data/test_inputs/mdm",
        file_pattern="mdm_*_joints.npy",
        output_dir="outputs/fixed/v13_mdm",
        label="MDM (Diffusion)"
    )

    # Summary
    print(f"\n{'='*60}")
    print("V13 Summary")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15} {'FSR Before':>10} {'FSR After':>10} {'Change':>10}")
    for name, res in [("MoMask (VQ)", momask_results), ("MDM (Diff)", mdm_results)]:
        if res:
            b = np.mean([r['fsr_before'] for r in res])
            a = np.mean([r['fsr_after'] for r in res])
            print(f"  {name:<15} {b:>9.1%} {a:>9.1%} {a-b:>+9.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
