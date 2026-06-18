"""
MotionFix 测试 MDM 输出

1. 加载 MDM 批量生成的 results.npy
2. MDM 输出已经是世界坐标 (22, 3, T)，转为 (T, 22, 3)
3. 运行 MotionFix V8 修正脚步滑动
4. 报告修正前后的滑步率
"""
import torch
import numpy as np
import os
import sys


def detect_skating(motion):
    """检测滑步率 - 与 eval_mdm_results.py 保持一致"""
    foot_indices = [7, 8, 10, 11]
    skating = 0
    contact = 0
    for t in range(len(motion) - 1):
        for foot_idx in foot_indices:
            if motion[t, foot_idx, 1] < 0.05:  # 固定高度阈值
                contact += 1
                vel = np.linalg.norm(motion[t+1, foot_idx, [0,2]] - motion[t, foot_idx, [0,2]])
                if vel > 0.03:
                    skating += 1
    return skating / contact if contact > 0 else 0


def fix_motion(model, motion, device='cuda'):
    """使用 MotionFix V8 修正脚部滑动"""
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1)
    motion_tensor = torch.FloatTensor(motion_flat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        fixed_tensor = model(motion_tensor, foot_only=True)
    return fixed_tensor.squeeze(0).cpu().numpy().reshape(T, 22, 3)


def main():
    print("=" * 70)
    print("MotionFix V8 -> MDM 批量测试")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # MDM 输出路径
    results_path = os.path.expanduser(
        "~/motion-diffusion-model-main/save/models_to_upload/"
        "humanml_trans_dec_512_bert/"
        "samples_humanml_trans_dec_512_bert_000200000_seed10_mdm_prompts_50/"
        "results.npy"
    )

    if not os.path.exists(results_path):
        print(f"\nERROR: results.npy not found at:")
        print(f"  {results_path}")
        print("\nPlease run MDM generation first:")
        print(f"  cd ~/motion-diffusion-model-main && bash generate_mdm_batch50.sh")
        sys.exit(1)

    # 加载 MDM 结果
    print(f"\nLoading MDM results...")
    data = np.load(results_path, allow_pickle=True).item()
    all_motions = data['motion']   # (N, 22, 3, T)
    all_texts = data['text']       # list of N text prompts
    all_lengths = data['lengths']  # (N,)

    N = all_motions.shape[0]
    print(f"Found {N} motions")
    print(f"Motion shape: {all_motions.shape}")

    # 加载 MotionFix V8 模型
    from motionfix_model import MotionFixNetwork
    model = MotionFixNetwork(blend_alpha=0.5).to(device)
    ckpt = torch.load("checkpoints_v8/best.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded MotionFix V8: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

    # 解析prompts以获取分类信息
    prompts_file = os.path.expanduser("~/HumanML3D/HumanML3D/test_prompts_50.txt")
    text_to_info = {}
    with open(prompts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 3:
                text_to_info[parts[2]] = {'pid': parts[0], 'category': parts[1]}

    # 创建输出目录
    output_dir = "fixed_outputs_mdm"
    raw_dir = "mdm_raw_joints"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"\n{'ID':<8} {'Category':<12} {'Before':>8} {'After':>8} {'Change':>9}  Prompt")
    print("-" * 100)

    results = []
    category_results = {}

    for i in range(N):
        # MDM 输出格式: (22, 3, T) -> (T, 22, 3)
        motion_raw = all_motions[i]  # (22, 3, T)
        length = int(all_lengths[i])

        # Transpose: (22, 3, T) -> (T, 22, 3)
        motion = motion_raw.transpose(2, 0, 1)[:length]

        text = all_texts[i]
        if isinstance(text, list):
            text = text[0] if text else ''

        info = text_to_info.get(text, {'pid': f'mdm_{i:02d}', 'category': 'unknown'})
        pid = info['pid']

        # 保存原始关节坐标
        np.save(f"{raw_dir}/mdm_{pid}_joints.npy", motion)

        # 计算修正前滑步率
        sr_before = detect_skating(motion)

        # MotionFix 修正
        fixed = fix_motion(model, motion, device)

        # 计算修正后滑步率
        sr_after = detect_skating(fixed)

        # 保存修正后结果
        np.save(f"{output_dir}/mdm_{pid}_fixed.npy", fixed)

        change = sr_after - sr_before
        category = info['category']
        print(f"{pid:<8} {category:<12} {sr_before:>7.1%} {sr_after:>7.1%} {change:>+8.1%}  {text[:45]}")

        results.append({
            'pid': pid,
            'category': category,
            'text': text,
            'before': sr_before,
            'after': sr_after
        })

        if category not in category_results:
            category_results[category] = {'before': [], 'after': []}
        category_results[category]['before'].append(sr_before)
        category_results[category]['after'].append(sr_after)

    # ---- 汇总统计 ----
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    if results:
        avg_b = np.mean([r['before'] for r in results])
        avg_a = np.mean([r['after'] for r in results])
        print(f"\n总体:")
        print(f"  平均滑步率 (修正前): {avg_b:.1%}")
        print(f"  平均滑步率 (修正后): {avg_a:.1%}")
        print(f"  绝对改善:           {avg_a - avg_b:+.1%}")
        print(f"  相对改善:           {(avg_b - avg_a) / avg_b * 100:.1f}%" if avg_b > 0 else "  (N/A)")

        print(f"\n按分类统计:")
        print(f"  {'Category':<14} {'Count':>5} {'Before':>8} {'After':>8} {'Change':>9}")
        print(f"  {'-'*50}")
        for cat in sorted(category_results.keys()):
            d = category_results[cat]
            avg_b_cat = np.mean(d['before'])
            avg_a_cat = np.mean(d['after'])
            print(f"  {cat:<14} {len(d['before']):>5} {avg_b_cat:>7.1%} {avg_a_cat:>7.1%} {avg_a_cat-avg_b_cat:>+8.1%}")

    print(f"\n输出文件:")
    print(f"  原始关节坐标: {raw_dir}/")
    print(f"  修正后关节坐标: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
