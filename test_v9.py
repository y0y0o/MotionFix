"""
MotionFix V9 测试脚本

对比 V8 的关键区别:
  - V9 内部已集成软门控+时间平滑, 调用 foot_only=True 即可
  - V9 会自动微调膝盖位置 (leg IK), 不需要额外处理
  - 同时测试 no_ik 和 ik 两个版本

用法:
  python test_v9.py                  # 测试 no_ik
  python test_v9.py --ik             # 测试 ik
  python test_v9.py --both           # 两个都测
"""
import torch
import numpy as np
import glob
import os
import sys
import argparse

from motionfix_model_v9 import MotionFixNetworkV9


def detect_skating(motion):
    """滑步率检测 — 与 test.py 保持一致"""
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


def fix_motion_v9(model, motion, device='cuda'):
    """
    V9 推理: foot_only=True 自动做软门控+时间平滑+骨骼链IK
    不需要像 V8 那样手动 _selective_replace
    """
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1)
    motion_tensor = torch.FloatTensor(motion_flat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        fixed_tensor = model(motion_tensor, foot_only=True)
    return fixed_tensor.squeeze(0).cpu().numpy().reshape(T, 22, 3)


def parse_filename(filename):
    """解析 MoMask 文件名: p000021_rotation_person_is...npy -> pid, category"""
    name = filename.replace('.npy', '')
    name = name[1:] if name.startswith('p') else name
    parts = name.split('_', 1)
    pid = parts[0]
    rest = parts[1] if len(parts) > 1 else ''
    cat_parts = rest.split('_', 1)
    category = cat_parts[0]
    return pid, category


def run_test(model, input_dir, output_dir, label, device):
    """对指定目录跑 MotionFix V9 测试"""
    files = sorted(glob.glob(f"{input_dir}/p*.npy"))

    if not files:
        print(f"  No .npy files in {input_dir}")
        return None

    print(f"\n{'─'*70}")
    print(f"  {label} — {len(files)} files")
    print(f"{'─'*70}")
    print(f"  {'PID':<8} {'Category':<12} {'Before':>7} {'After':>7} {'Change':>9}")
    print(f"  {'─'*50}")

    results = []
    cat_results = {}

    for fp in files:
        fn = os.path.basename(fp)
        pid, cat = parse_filename(fn)

        data = np.load(fp)
        motion = data[0] if len(data.shape) == 4 else data

        sr_before = detect_skating(motion)
        fixed = fix_motion_v9(model, motion, device)
        sr_after = detect_skating(fixed)

        change = sr_after - sr_before
        print(f"  {pid:<8} {cat:<12} {sr_before:>6.1%} {sr_after:>6.1%} {change:>+7.1%}")

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/{fn}", motion)
        np.save(f"{output_dir}/{fn.replace('.npy', '_fixed.npy')}", fixed)

        results.append({'pid': pid, 'cat': cat, 'before': sr_before, 'after': sr_after})
        if cat not in cat_results:
            cat_results[cat] = {'before': [], 'after': []}
        cat_results[cat]['before'].append(sr_before)
        cat_results[cat]['after'].append(sr_after)

    # 汇总
    print()
    avg_b = np.mean([r['before'] for r in results])
    avg_a = np.mean([r['after'] for r in results])
    print(f"  总体: {avg_b:.1%} → {avg_a:.1%}  ({avg_a-avg_b:+.1%})")
    if avg_b > 0:
        print(f"  相对改善: {(avg_b-avg_a)/avg_b*100:.1f}%")

    imp = sum(1 for r in results if r['after'] < r['before'])
    wor = sum(1 for r in results if r['after'] > r['before'])
    unc = sum(1 for r in results if r['after'] == r['before'])
    print(f"  改善:{imp} 变差:{wor} 不变:{unc}")

    print(f"\n  按分类:")
    print(f"  {'Category':<14} {'Count':>5} {'Before':>8} {'After':>8} {'Change':>9}")
    print(f"  {'─'*50}")
    for cat in sorted(cat_results.keys()):
        d = cat_results[cat]
        print(f"  {cat:<14} {len(d['before']):>5} {np.mean(d['before']):>7.1%} "
              f"{np.mean(d['after']):>7.1%} {np.mean(d['after'])-np.mean(d['before']):>+8.1%}")

    return {'dir': output_dir, 'avg_before': avg_b, 'avg_after': avg_a}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ik', action='store_true', help='只测 IK 版本')
    parser.add_argument('--both', action='store_true', help='同时测 no_ik 和 ik')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v9/best.pth',
                        help='V9 checkpoint 路径')
    args = parser.parse_args()

    # 默认测 no_ik
    if not args.ik and not args.both:
        args.both = True  # 两个都测

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("MotionFix V9 Test")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # 加载 V9 模型
    model = MotionFixNetworkV9(blend_alpha=0.5, temperature=0.01).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded V9: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")
    else:
        print(f"WARNING: {args.checkpoint} not found, using random weights!")
        print(f"  Train first: python train_v9.py")

    momask_dir = os.path.expanduser("./momask_50_results")
    all_summaries = []

    if args.both or not args.ik:
        summary = run_test(
            model=model,
            input_dir=f"{momask_dir}/no_ik",
            output_dir="fixed_outputs_v9_no_ik",
            label="MoMask no_ik + MotionFix V9",
            device=device,
        )
        if summary:
            all_summaries.append(summary)

    if args.both or args.ik:
        summary = run_test(
            model=model,
            input_dir=f"{momask_dir}/ik",
            output_dir="fixed_outputs_v9_ik",
            label="MoMask ik + MotionFix V9",
            device=device,
        )
        if summary:
            all_summaries.append(summary)

    # 最终对比
    if len(all_summaries) >= 2:
        print(f"\n{'='*70}")
        print("V9 no_ik vs ik 对比")
        print(f"{'='*70}")
        print(f"  no_ik: {all_summaries[0]['avg_before']:.1%} → {all_summaries[0]['avg_after']:.1%}")
        print(f"  ik:    {all_summaries[1]['avg_before']:.1%} → {all_summaries[1]['avg_after']:.1%}")


if __name__ == "__main__":
    main()
