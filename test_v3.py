import torch
import numpy as np
import glob
import os

from motionfix_model import MotionFixNetwork


def detect_skating(motion):
    skating = 0
    contact = 0
    for foot_idx in [10, 11]:
        foot = motion[:, foot_idx, :]
        heights = foot[:, 1]
        ground = np.percentile(heights, 5)
        threshold = ground + 0.05
        for t in range(len(motion) - 1):
            if heights[t] < threshold:
                contact += 1
                vel = np.linalg.norm(foot[t+1, [0,2]] - foot[t, [0,2]])
                if vel > 0.03:
                    skating += 1
    return skating / contact if contact > 0 else 0


def fix_motion(model, motion, device='cuda'):
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1)
    motion_tensor = torch.FloatTensor(motion_flat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        # V3: 直接用全部输出，不做选择性替换
        fixed_tensor = model(motion_tensor, foot_only=False)
    return fixed_tensor.squeeze(0).cpu().numpy().reshape(T, 22, 3)


def main():
    print("=" * 60)
    print("MotionFix V3 - Test (full reconstruction)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MotionFixNetwork(blend_alpha=0.5).to(device)
    ckpt = torch.load("checkpoints_v3/best.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

    momask_dir = os.path.expanduser("./momask_results")
    momask_files = sorted(glob.glob(f"{momask_dir}/momask_*_no_ik.npy"))

    output_dir = "fixed_outputs_v3"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nFound {len(momask_files)} MoMask files")
    print(f"{'Test':<50} | {'Before':>7} | {'After':>7} | {'Change':>8}")
    print("-" * 85)

    results = []
    for filepath in momask_files:
        filename = os.path.basename(filepath)
        name = filename.replace('momask_', '').replace('_no_ik.npy', '')

        data = np.load(filepath)
        motion = data[0] if len(data.shape) == 4 else data

        sr_before = detect_skating(motion)
        fixed = fix_motion(model, motion, device)
        sr_after = detect_skating(fixed)

        change = sr_after - sr_before
        print(f"{name[:50]:<50} | {sr_before:>6.1%} | {sr_after:>6.1%} | {change:>+7.1%}")

        np.save(f"{output_dir}/{filename}", motion)
        np.save(f"{output_dir}/{filename.replace('_no_ik.npy', '_fixed.npy')}", fixed)
        results.append({'name': name, 'before': sr_before, 'after': sr_after})

    if results:
        avg_b = np.mean([r['before'] for r in results])
        avg_a = np.mean([r['after'] for r in results])
        print("-" * 85)
        print(f"{'Average':<50} | {avg_b:>6.1%} | {avg_a:>6.1%} | {avg_a-avg_b:>+7.1%}")
        print("=" * 85)

    print(f"\n✓ Saved to: {output_dir}/")

if __name__ == "__main__":
    main()