"""
MotionFix 测试 MoMask 50 个 prompts 的输出

读取 momask_50_results/no_ik/ 目录，按分类输出结果
文件名格式: p{pid}_{category}_{text}.npy
"""
import torch
import numpy as np
import glob
import os

from models.v8 import MotionFixNetwork


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
        fixed_tensor = model(motion_tensor, foot_only=True)
    return fixed_tensor.squeeze(0).cpu().numpy().reshape(T, 22, 3)


def parse_filename(filename):
    """
    p000021_rotation_person_is_walking_normally_in_a.npy
    -> pid='000021', category='rotation', text='person is walking normally in a...'
    """
    name = filename.replace('.npy', '')
    # 去掉 p 前缀
    name = name[1:] if name.startswith('p') else name
    parts = name.split('_', 1)
    pid = parts[0]  # 000021
    rest = parts[1] if len(parts) > 1 else ''
    # 解析 category (第一个下划线之前)
    cat_parts = rest.split('_', 1)
    category = cat_parts[0]  # rotation
    text_hint = cat_parts[1] if len(cat_parts) > 1 else ''
    return pid, category, text_hint


def main():
    print("=" * 80)
    print("MotionFix V8 - MoMask 50 Prompts Test")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = MotionFixNetwork(blend_alpha=0.5).to(device)
    ckpt = torch.load("checkpoints/v8/best.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded MotionFix V8: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

    # 读取 MoMask no_ik 结果
    momask_dir = os.path.expanduser("data/test_inputs/momask_50/no_ik")
    momask_files = sorted(glob.glob(f"{momask_dir}/p*.npy"))

    if not momask_files:
        print(f"\nERROR: No .npy files found in {momask_dir}")
        return

    print(f"\nFound {len(momask_files)} MoMask files")

    output_dir = "outputs/fixed/momask"
    os.makedirs(output_dir, exist_ok=True)

    # 按分类排序显示
    print(f"\n{'PID':<8} {'Category':<12} {'Before':>8} {'After':>8} {'Change':>9}")
    print("-" * 55)

    results = []
    category_results = {}

    for filepath in momask_files:
        filename = os.path.basename(filepath)
        pid, category, text_hint = parse_filename(filename)

        data = np.load(filepath)
        motion = data[0] if len(data.shape) == 4 else data

        sr_before = detect_skating(motion)
        fixed = fix_motion(model, motion, device)
        sr_after = detect_skating(fixed)

        change = sr_after - sr_before
        print(f"{pid:<8} {category:<12} {sr_before:>7.1%} {sr_after:>7.1%} {change:>+8.1%}")

        # 保存结果
        np.save(f"{output_dir}/{filename}", motion)
        np.save(f"{output_dir}/{filename.replace('.npy', '_fixed.npy')}", fixed)

        results.append({'pid': pid, 'category': category, 'before': sr_before, 'after': sr_after})

        if category not in category_results:
            category_results[category] = {'before': [], 'after': []}
        category_results[category]['before'].append(sr_before)
        category_results[category]['after'].append(sr_after)

    # ---- 汇总统计 ----
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)

    if results:
        avg_b = np.mean([r['before'] for r in results])
        avg_a = np.mean([r['after'] for r in results])
        print(f"\n总体:")
        print(f"  平均滑步率 (修正前):  {avg_b:.1%}")
        print(f"  平均滑步率 (修正后):  {avg_a:.1%}")
        print(f"  绝对改善:            {avg_a - avg_b:+.1%}")
        if avg_b > 0:
            print(f"  相对改善:            {(avg_b - avg_a) / avg_b * 100:.1f}%")

        # 按分类
        print(f"\n按分类统计:")
        print(f"  {'Category':<14} {'Count':>5} {'Before':>8} {'After':>8} {'Change':>9}")
        print(f"  {'-'*50}")
        for cat in sorted(category_results.keys()):
            d = category_results[cat]
            avg_b_cat = np.mean(d['before'])
            avg_a_cat = np.mean(d['after'])
            print(f"  {cat:<14} {len(d['before']):>5} {avg_b_cat:>7.1%} {avg_a_cat:>7.1%} {avg_a_cat-avg_b_cat:>+8.1%}")

        # 改善 / 变差 汇总
        improved = [r for r in results if r['after'] < r['before']]
        worsened = [r for r in results if r['after'] > r['before']]
        unchanged = [r for r in results if r['after'] == r['before']]
        print(f"\n改善: {len(improved)}/{len(results)}, 变差: {len(worsened)}/{len(results)}, 不变: {len(unchanged)}/{len(results)}")

    print(f"\n输出文件: {output_dir}/")


if __name__ == "__main__":
    main()
