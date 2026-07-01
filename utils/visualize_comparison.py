import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

from models.v8 import MotionFixNetwork

# HumanML3D骨架连接
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),      # 骨盆 → 左右髋
    (1, 4), (2, 5),      # 髋 → 膝
    (4, 7), (5, 8),      # 膝 → 踝
    (7, 10), (8, 11),    # 踝 → 脚
    (0, 3), (3, 6),      # 骨盆 → 脊柱
    (6, 9), (9, 12),     # 脊柱 → 颈 → 头(15)
    (9, 13), (9, 14),    # 颈 → 左右肩
    (13, 16), (14, 17),  # 肩 → 肘
    (16, 18), (17, 19),  # 肘 → 腕
    (18, 20), (19, 21),  # 腕 → 手
    (12, 15),            # 颈 → 头
]


def fix_motion(model, motion, device='cuda'):
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1)
    motion_tensor = torch.FloatTensor(motion_flat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        fixed_tensor = model(motion_tensor)
    fixed = fixed_tensor.squeeze(0).cpu().numpy()
    return fixed.reshape(T, 22, 3)


def detect_skating_per_frame(motion):
    """返回每一帧的滑步状态"""
    T = len(motion)
    skating_frames = np.zeros(T, dtype=bool)

    for foot_idx in [10, 11]:
        foot = motion[:, foot_idx, :]
        heights = foot[:, 1]
        ground = np.percentile(heights, 5)
        threshold = ground + 0.05

        for t in range(T - 1):
            if heights[t] < threshold:
                vel = np.linalg.norm(foot[t+1, [0,2]] - foot[t, [0,2]])
                if vel > 0.03:
                    skating_frames[t] = True

    return skating_frames


def create_comparison_video(original, fixed, test_name, output_path):
    """生成并排对比视频"""
    T = min(len(original), len(fixed))

    # 计算滑步
    skating_before = detect_skating_per_frame(original)
    skating_after = detect_skating_per_frame(fixed)

    sr_before = skating_before.sum() / T
    sr_after = skating_after.sum() / T

    # 计算坐标范围
    all_data = np.concatenate([original[:T], fixed[:T]], axis=0)
    x_range = [all_data[:, :, 0].min() - 0.3, all_data[:, :, 0].max() + 0.3]
    y_range = [all_data[:, :, 1].min() - 0.1, all_data[:, :, 1].max() + 0.3]
    z_range = [all_data[:, :, 2].min() - 0.3, all_data[:, :, 2].max() + 0.3]

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    def update(frame):
        for ax, motion, skating, title, sr in [
            (ax1, original, skating_before, 'Before MotionFix', sr_before),
            (ax2, fixed, skating_after, 'After MotionFix', sr_after)
        ]:
            ax.cla()
            joints = motion[frame]

            # 画骨架
            for j1, j2 in SKELETON_CONNECTIONS:
                ax.plot(
                    [joints[j1, 0], joints[j2, 0]],
                    [joints[j1, 1], joints[j2, 1]],
                    [joints[j1, 2], joints[j2, 2]],
                    'b-', linewidth=2, alpha=0.7
                )

            # 画关节
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                      c='blue', s=20, alpha=0.6)

            # 突出脚部
            for foot_idx in [10, 11]:
                color = 'red' if skating[frame] else 'green'
                ax.scatter(
                    joints[foot_idx, 0],
                    joints[foot_idx, 1],
                    joints[foot_idx, 2],
                    c=color, s=200, marker='o',
                    edgecolors='black', linewidths=2
                )

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            status = "🔴 SKATING" if skating[frame] else "🟢 OK"
            ax.set_title(f'{title}\nSR: {sr:.1%} | Frame {frame}/{T} | {status}',
                        fontsize=12, fontweight='bold')

        fig.suptitle(f'{test_name}', fontsize=14, fontweight='bold')
        return []

    anim = animation.FuncAnimation(fig, update, frames=T, interval=50, blit=False)
    anim.save(output_path, writer='ffmpeg', fps=20, dpi=100)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_comparison_plot(original, fixed, test_name, output_path):
    """生成静态对比图（脚部高度+速度）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, (motion, label) in enumerate([(original, 'Before'), (fixed, 'After')]):
        foot = motion[:, 10, :]  # Left Toe
        heights = foot[:, 1]
        T = len(motion)

        velocities = np.zeros(T)
        for t in range(T-1):
            velocities[t] = np.linalg.norm(foot[t+1, [0,2]] - foot[t, [0,2]])

        ground = np.percentile(heights, 5)
        threshold = ground + 0.05

        # 检测滑步帧
        skating = np.zeros(T, dtype=bool)
        contact = 0
        skating_count = 0
        for t in range(T-1):
            if heights[t] < threshold:
                contact += 1
                if velocities[t] > 0.03:
                    skating[t] = True
                    skating_count += 1

        sr = skating_count / contact if contact > 0 else 0

        # 高度图
        ax_h = axes[0, col]
        ax_h.plot(heights, 'b-', linewidth=2)
        ax_h.axhline(threshold, color='gray', linestyle='--', alpha=0.5)
        for t in range(T):
            if skating[t]:
                ax_h.axvspan(t, t+1, alpha=0.3, color='red')
        ax_h.set_ylabel('Height (m)')
        ax_h.set_title(f'{label} - Left Toe Height (SR: {sr:.1%})',
                       fontsize=12, fontweight='bold')
        ax_h.grid(True, alpha=0.3)

        # 速度图
        ax_v = axes[1, col]
        ax_v.plot(velocities[:-1], 'g-', linewidth=2)
        ax_v.axhline(0.03, color='gray', linestyle='--', alpha=0.5)
        for t in range(T):
            if skating[t]:
                ax_v.axvspan(t, t+1, alpha=0.5, color='red')
        ax_v.set_xlabel('Frame')
        ax_v.set_ylabel('Velocity (m/frame)')
        ax_v.set_title(f'{label} - Velocity', fontsize=12, fontweight='bold')
        ax_v.grid(True, alpha=0.3)

    fig.suptitle(f'{test_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def main():
    print("=" * 60)
    print("MotionFix - Generate Comparison Visualizations")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MotionFixNetwork().to(device)

    checkpoint = torch.load("checkpoints/v3/best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model (epoch {checkpoint['epoch']+1})")

    momask_dir = os.path.expanduser("~/momask_batch_output")
    momask_files = sorted(glob.glob(f"{momask_dir}/momask_*_no_ik.npy"))

    output_dir = "outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # 选择最有代表性的测试
    key_tests = [
        'spins_around',
        'backward_in_a_circle',
        'backward_in_a_curved',
        'forward_slowly',
        'zigzag',
    ]

    for filepath in momask_files:
        name = os.path.basename(filepath).replace('momask_', '').replace('_no_ik.npy', '')
        short_name = name[:40]

        # 只处理关键测试
        is_key = any(key in name for key in key_tests)
        if not is_key:
            continue

        print(f"\nProcessing: {short_name}")

        # 加载
        data = np.load(filepath)
        motion = data[0] if len(data.shape) == 4 else data

        # 修正
        fixed = fix_motion(model, motion, device)

        # 保存修正后的npy
        np.save(f"{output_dir}/{short_name}_fixed.npy", fixed)
        np.save(f"{output_dir}/{short_name}_original.npy", motion)

        # 生成静态对比图
        create_comparison_plot(
            motion, fixed, short_name,
            f"{output_dir}/{short_name}_comparison.png"
        )

        # 生成对比视频
        try:
            create_comparison_video(
                motion, fixed, short_name,
                f"{output_dir}/{short_name}_comparison.mp4"
            )
        except Exception as e:
            print(f"  ⚠️ Video failed: {e}")
            print(f"  (ffmpeg might not be installed)")

    print(f"\n✓ All visualizations saved in: {output_dir}/")


if __name__ == "__main__":
    main()
