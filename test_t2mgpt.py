"""
MotionFix 测试 T2M-GPT 输出

1. 加载 T2M-GPT 批量生成的 .npy 文件 (output_batch50/)
2. 将 (1, T, 263) 格式转为 (T, 22, 3) 关节坐标
3. 运行 MotionFix V8 修正脚步滑动
4. 报告修正前后的滑步率
"""
import torch
import numpy as np
import glob
import os
import sys

# 添加 T2M-GPT 的 utils 路径
sys.path.insert(0, os.path.expanduser("~/Project/T2M-GPT"))
from utils.motion_process import recover_from_ric

from motionfix_model import MotionFixNetwork


def detect_skating(motion):
    """检测滑步率 - 与 eval_t2mgpt_results_fixed.py 保持一致"""
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


def convert_t2mgpt_to_joints(npy_path, mean, std, device='cuda'):
    """
    将 T2M-GPT 输出 (1, T, 263) 转为关节坐标 (T, 22, 3)
    参考 eval_t2mgpt_results.py 和 verify_t2mgpt.py
    """
    motion_features = np.load(npy_path)[0]  # (T, 263)

    # 反归一化
    motion_denorm = motion_features * std + mean

    # 使用 recover_from_ric 转为关节坐标
    motion_tensor = torch.from_numpy(motion_denorm).float().to(device)
    joints = recover_from_ric(motion_tensor, 22)
    joints = joints.cpu().numpy()  # (T, 22, 3)

    return joints


def fix_motion(model, motion, device='cuda'):
    """使用 MotionFix V8 修正脚部滑动"""
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1)
    motion_tensor = torch.FloatTensor(motion_flat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        # V8: 选择性替换模式
        fixed_tensor = model(motion_tensor, foot_only=True)
    return fixed_tensor.squeeze(0).cpu().numpy().reshape(T, 22, 3)


def main():
    print("=" * 70)
    print("MotionFix V8 -> T2M-GPT 批量测试")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载归一化参数 (使用 MDM 的 t2m_mean/std，与 eval_t2mgpt_results_fixed.py 一致)
    mean_path = os.path.expanduser("~/motion-diffusion-model-main/dataset/t2m_mean.npy")
    std_path = os.path.expanduser("~/motion-diffusion-model-main/dataset/t2m_std.npy")
    mean = np.load(mean_path)
    std = np.load(std_path)
    print(f"Loaded Mean/Std from MDM dataset (t2m_mean/t2m_std)")

    # 加载 MotionFix V8 模型
    model = MotionFixNetwork(blend_alpha=0.5).to(device)
    ckpt = torch.load("checkpoints_v8/best.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded MotionFix V8: epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

    # 读取所有 T2M-GPT 输出
    t2mgpt_dir = os.path.expanduser("~/Project/T2M-GPT/output_batch50")
    t2mgpt_files = sorted(glob.glob(f"{t2mgpt_dir}/t2mgpt_*.npy"))

    if not t2mgpt_files:
        print(f"\nERROR: No T2M-GPT output files found in {t2mgpt_dir}")
        print("Please run generate_t2mgpt_batch50.py first.")
        sys.exit(1)

    print(f"\nFound {len(t2mgpt_files)} T2M-GPT output files")

    # 创建输出目录
    output_dir = "fixed_outputs_t2mgpt"
    raw_dir = "t2mgpt_raw_joints"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # 解析prompts以获取分类信息
    prompts_file = os.path.expanduser("~/HumanML3D/HumanML3D/test_prompts_50.txt")
    prompt_info = {}
    with open(prompts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 3:
                prompt_info[parts[0]] = {'category': parts[1], 'text': parts[2]}

    print(f"\n{'ID':<8} {'Category':<12} {'Before':>8} {'After':>8} {'Change':>9}  Prompt")
    print("-" * 100)

    results = []
    category_results = {}  # 按分类统计

    for filepath in t2mgpt_files:
        filename = os.path.basename(filepath)
        pid = filename.replace('t2mgpt_', '').replace('.npy', '')
        info = prompt_info.get(pid, {'category': 'unknown', 'text': filename})

        # Step 1: 转换 T2M-GPT 输出为关节坐标
        try:
            joints = convert_t2mgpt_to_joints(filepath, mean, std, device)
        except Exception as e:
            print(f"  [{pid}] ERROR converting: {e}")
            continue

        # 保存原始关节坐标
        np.save(f"{raw_dir}/t2mgpt_{pid}_joints.npy", joints)

        # Step 2: 计算滑步率（修正前）
        sr_before = detect_skating(joints)

        # Step 3: MotionFix 修正
        fixed = fix_motion(model, joints, device)

        # Step 4: 计算滑步率（修正后）
        sr_after = detect_skating(fixed)

        # 保存修正后的结果
        np.save(f"{output_dir}/t2mgpt_{pid}_fixed.npy", fixed)

        change = sr_after - sr_before
        category = info['category']
        print(f"{pid:<8} {category:<12} {sr_before:>7.1%} {sr_after:>7.1%} {change:>+8.1%}  {info['text'][:45]}")

        results.append({
            'pid': pid,
            'category': category,
            'text': info['text'],
            'before': sr_before,
            'after': sr_after
        })

        # 按分类统计
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
