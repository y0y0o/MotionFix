"""
MotionFix V6: 基于速度衰减的滑步修正

原理：
1. 检测哪些帧的脚在地面（高度<阈值）
2. 对于地面帧，降低脚的水平速度
3. 从修改后的速度重建位置
4. 空中的帧完全不改

不需要训练！直接用规则修正。
"""

import numpy as np
import glob
import os


def fix_foot_skating(motion, foot_idx, height_thresh=0.05, 
                     velocity_reduction=0.8, smooth_window=3):
    """
    修正单只脚的滑步
    
    motion: (T, 22, 3)
    foot_idx: 脚的关节索引
    velocity_reduction: 速度衰减比例 (0.8 = 减少80%的速度)
    smooth_window: 平滑窗口
    """
    fixed = motion.copy()
    T = len(motion)
    
    foot = motion[:, foot_idx, :].copy()  # (T, 3)
    heights = foot[:, 1]
    
    # 找到地面高度
    ground_level = np.percentile(heights, 5)
    contact_threshold = ground_level + height_thresh
    
    # 检测接触帧
    is_contact = heights < contact_threshold
    
    # 对接触帧，减小水平速度
    for t in range(1, T):
        if is_contact[t]:
            # 计算当前帧的水平位移
            dx = foot[t, 0] - foot[t-1, 0]  # X方向
            dz = foot[t, 2] - foot[t-1, 2]  # Z方向
            
            horizontal_speed = np.sqrt(dx**2 + dz**2)
            
            if horizontal_speed > 0.01:  # 有明显移动
                # 减小水平位移
                foot[t, 0] = foot[t-1, 0] + dx * (1 - velocity_reduction)
                foot[t, 2] = foot[t-1, 2] + dz * (1 - velocity_reduction)
    
    # 平滑过渡（避免突变）
    from scipy.ndimage import uniform_filter1d
    
    # 只平滑接触段和非接触段的边界
    for dim in [0, 2]:  # 只平滑XZ（水平方向）
        original = motion[:, foot_idx, dim]
        corrected = foot[:, dim]
        
        # 混合：接触帧用修正值，非接触帧用原始值
        blended = np.where(is_contact, corrected, original)
        
        # 轻微平滑过渡
        blended = uniform_filter1d(blended, size=smooth_window, mode='nearest')
        
        # 非接触帧强制恢复原始值
        blended = np.where(is_contact, blended, original)
        
        fixed[:, foot_idx, dim] = blended
    
    return fixed


def detect_skating(motion):
    """检测滑步率"""
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


def fix_motion(motion, velocity_reduction=0.8):
    """修正所有脚的滑步"""
    fixed = motion.copy()
    
    # 修正4个脚部关节
    for foot_idx in [7, 8, 10, 11]:
        fixed = fix_foot_skating(
            fixed, foot_idx, 
            velocity_reduction=velocity_reduction
        )
    
    return fixed


def main():
    print("=" * 60)
    print("MotionFix V6 (velocity reduction)")
    print("No training needed!")
    print("=" * 60)
    
    momask_dir = os.path.expanduser("~/momask_batch_output")
    momask_files = sorted(glob.glob(f"{momask_dir}/momask_*_no_ik.npy"))
    
    output_dir = "fixed_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试不同的衰减比例
    reductions = [0.5, 0.7, 0.8, 0.9]
    
    print(f"\n测试不同的速度衰减比例:")
    print(f"{'Test':<40}", end="")
    print(f" | {'原始':>6}", end="")
    for r in reductions:
        print(f" | r={r}:>6}", end="")
    print()
    print("-" * 90)
    
    best_reduction = 0.8
    best_avg = 1.0
    
    for r in reductions:
        results = []
        for filepath in momask_files:
            data = np.load(filepath)
            motion = data[0] if len(data.shape) == 4 else data
            
            sr_before = detect_skating(motion)
            fixed = fix_motion(motion, velocity_reduction=r)
            sr_after = detect_skating(fixed)
            
            results.append({'before': sr_before, 'after': sr_after})
        
        avg_after = np.mean([x['after'] for x in results])
        if avg_after < best_avg:
            best_avg = avg_after
            best_reduction = r
    
    # 用最佳参数生成最终结果
    print(f"\n最佳衰减比例: {best_reduction}")
    print(f"\n{'Test':<50} | {'Before':>7} | {'After':>7} | {'Change':>8}")
    print("-" * 85)
    
    all_results = []
    
    for filepath in momask_files:
        filename = os.path.basename(filepath)
        name = filename.replace('momask_', '').replace('_no_ik.npy', '')
        
        data = np.load(filepath)
        motion = data[0] if len(data.shape) == 4 else data
        
        sr_before = detect_skating(motion)
        fixed = fix_motion(motion, velocity_reduction=best_reduction)
        sr_after = detect_skating(fixed)
        
        change = sr_after - sr_before
        print(f"{name[:50]:<50} | {sr_before:>6.1%} | {sr_after:>6.1%} | {change:>+7.1%}")
        
        # 保存
        np.save(f"{output_dir}/{filename}", motion)
        np.save(f"{output_dir}/{filename.replace('_no_ik.npy', '_fixed.npy')}", fixed)
        
        all_results.append({'name': name, 'before': sr_before, 'after': sr_after})
    
    avg_before = np.mean([r['before'] for r in all_results])
    avg_after = np.mean([r['after'] for r in all_results])
    print("-" * 85)
    print(f"{'Average':<50} | {avg_before:>6.1%} | {avg_after:>6.1%} | {avg_after-avg_before:>+7.1%}")
    print("=" * 85)
    
    print(f"\n✓ Saved to: {output_dir}/")
    print(f"\n下载到Windows后用MoMask生成视频对比！")


if __name__ == "__main__":
    main()
