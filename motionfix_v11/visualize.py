"""
V11 Test Results Visualization
Generates side-by-side comparison videos: original (left) vs fixed (right)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import glob

# HumanML3D skeleton connections (bones to draw)
BONES = [
    # Spine
    (0, 3), (3, 6), (6, 9), (9, 12),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20),
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21),
    # Neck to head
    (12, 15),
]

FOOT_JOINTS = {7, 8, 10, 11}  # highlight these

def load_original_fixed(filepath):
    """Load original .npy file and its _fixed counterpart"""
    fixed_path = filepath.replace('.npy', '_fixed.npy')
    orig = np.load(filepath)
    fixed = np.load(fixed_path)
    # Shape: (T, 22, 3) or (1, T, 22, 3)
    if len(orig.shape) == 4:
        orig = orig[0]
    if len(fixed.shape) == 4:
        fixed = fixed[0]
    return orig, fixed

def create_comparison_video(orig, fixed, name, output_dir):
    """Create side-by-side 3D skeleton animation"""
    T = min(orig.shape[0], 200)  # cap at 200 frames
    orig = orig[:T]
    fixed = fixed[:T]

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Find consistent axis limits
    all_data = np.concatenate([orig.reshape(-1, 3), fixed.reshape(-1, 3)], axis=0)
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    z_min, z_max = all_data[:, 2].min(), all_data[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    mid_x, mid_y, mid_z = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2

    for ax, title in [(ax1, 'Original'), (ax2, 'Fixed (V11)')]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=15, azim=-60)

    # Pre-compute contact/FSR info
    def get_fsr_info(motion):
        foot_vels = []
        for fj in [7, 8]:
            heights = motion[:, fj, 1]
            ground = np.percentile(heights, 5)
            for t in range(1, len(motion)):
                if heights[t] < ground + 0.05:
                    vel = np.linalg.norm(motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]])
                    foot_vels.append(vel)
        return np.mean(foot_vels) if foot_vels else 0

    orig_fsr = get_fsr_info(orig)
    fixed_fsr = get_fsr_info(fixed)

    # Lines for each bone
    lines1, lines2 = [], []
    scatters1, scatters2 = [], []

    for _ in BONES:
        l1, = ax1.plot([], [], [], 'b-', linewidth=1.5, alpha=0.7)
        l2, = ax2.plot([], [], [], 'r-', linewidth=1.5, alpha=0.7)
        lines1.append(l1)
        lines2.append(l2)

    # Foot scatter
    sf1 = ax1.scatter([], [], [], c='red', s=30, marker='o')
    sf2 = ax2.scatter([], [], [], c='red', s=30, marker='o')
    scatters1.append(sf1)
    scatters2.append(sf2)

    # Other joints scatter
    so1 = ax1.scatter([], [], [], c='blue', s=10, marker='o')
    so2 = ax2.scatter([], [], [], c='blue', s=10, marker='o')
    scatters1.append(so1)
    scatters2.append(so2)

    def update(frame):
        for side, (motion, lines, scatters) in enumerate([
            (orig, lines1, scatters1),
            (fixed, lines2, scatters2)
        ]):
            pos = motion[frame]  # (22, 3)
            for i, (j1, j2) in enumerate(BONES):
                x = [pos[j1, 0], pos[j2, 0]]
                y = [pos[j1, 1], pos[j2, 1]]
                z = [pos[j1, 2], pos[j2, 2]]
                lines[i].set_data(x, y)
                lines[i].set_3d_properties(z)

            # Foot joints
            foot_pos = pos[list(FOOT_JOINTS)]
            scatters[0]._offsets3d = (foot_pos[:, 0], foot_pos[:, 1], foot_pos[:, 2])

            # Other joints
            other = [j for j in range(22) if j not in FOOT_JOINTS]
            other_pos = pos[other]
            scatters[1]._offsets3d = (other_pos[:, 0], other_pos[:, 1], other_pos[:, 2])

        fig.suptitle(f'{name}\nOrig foot-vel: {orig_fsr:.3f} | Fixed foot-vel: {fixed_fsr:.3f}',
                     fontsize=11, y=0.98)
        return lines1 + lines2 + scatters1 + scatters2

    ani = FuncAnimation(fig, update, frames=T, interval=33, blit=False)

    output_path = os.path.join(output_dir, f'{name}.mp4')
    writer = FFMpegWriter(fps=30, metadata=dict(artist='MotionFix'), bitrate=2000)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f'  Saved: {output_path}')
    return output_path

def main():
    output_dir = 'comparison_videos'
    os.makedirs(output_dir, exist_ok=True)

    # Find all original files (not _fixed)
    src_dir = 'fixed_outputs_v11'
    all_files = sorted(glob.glob(f'{src_dir}/*.npy'))
    orig_files = [f for f in all_files if '_fixed.npy' not in f]

    # Pick representatives: best FSR improvements + worst jitter cases
    # Best FSR improvements (from test results):
    # p009161: 21.4%→8.3%,  p004822: 40.3%→24.0%, p007767: 18.8%→9.1%, p011028: 14.0%→5.5%
    # Worst jitter: p009613: x29, p009958: x46
    priority = [
        'p009161', 'p004822', 'p007767', 'p011028',  # good FSR reduction
        'p009613', 'p009958',                          # worst jitter
    ]

    selected = []
    for prefix in priority:
        for f in orig_files:
            bn = os.path.basename(f)
            if bn.startswith(prefix):
                selected.append(bn)
                break

    # If some not found, fill with first few available
    if len(selected) < 4:
        for f in orig_files:
            bn = os.path.basename(f)
            if bn not in selected:
                selected.append(bn)
            if len(selected) >= 6:
                break

    for fn in selected[:6]:
        filepath = os.path.join(src_dir, fn)
        name = fn.replace('.npy', '')
        print(f'Processing: {name}')
        orig, fixed = load_original_fixed(filepath)
        create_comparison_video(orig, fixed, name, output_dir)

    print(f'\nDone! Videos in {output_dir}/')

if __name__ == '__main__':
    main()
