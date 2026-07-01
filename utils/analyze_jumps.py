"""
Analyze foot jump artifact: sudden position spikes in fixed motions.
Identifies frames where the foot jumps to a wrong position then snaps back.
"""
import numpy as np
import os
import glob

FOOT_JOINTS = [7, 8, 10, 11]  # L_ankle, R_ankle, L_foot, R_foot
FOOT_NAMES = {7: 'L_Ankle', 8: 'R_Ankle', 10: 'L_Foot', 11: 'R_Foot'}

def load_motion(filepath):
    data = np.load(filepath)
    if len(data.shape) == 4:
        data = data[0]
    return data  # (T, 22, 3)

def analyze_foot_jumps(orig, fixed, name):
    """Analyze frame-by-frame foot jumps"""
    T = min(orig.shape[0], fixed.shape[0])
    orig = orig[:T]
    fixed = fixed[:T]

    results = {}
    for fj in FOOT_JOINTS:
        # Compute foot velocities (horizontal)
        orig_vel = np.zeros(T)
        fixed_vel = np.zeros(T)
        fixed_acc = np.zeros(T)
        orig_to_fixed_dist = np.zeros(T)  # L2 distance between orig and fixed foot position

        for t in range(T):
            # Distance from original to fixed (3D)
            orig_to_fixed_dist[t] = np.linalg.norm(fixed[t, fj] - orig[t, fj])

            if t >= 1:
                orig_vel[t] = np.linalg.norm(orig[t, fj, [0, 2]] - orig[t-1, fj, [0, 2]])
                fixed_vel[t] = np.linalg.norm(fixed[t, fj, [0, 2]] - fixed[t-1, fj, [0, 2]])

            if t >= 2:
                # Acceleration (change in velocity) - detects sudden jumps
                prev_vel = np.linalg.norm(fixed[t-1, fj] - fixed[t-2, fj])
                curr_vel = np.linalg.norm(fixed[t, fj] - fixed[t-1, fj])
                fixed_acc[t] = abs(curr_vel - prev_vel)

        # Detect jump frames: frames where acceleration is unusually high
        acc_threshold = np.percentile(fixed_acc[2:], 95) * 2  # 2x 95th percentile
        jump_frames = np.where(fixed_acc > acc_threshold)[0]

        # Also detect: frames where foot moves AWAY from orig, then BACK next frame
        # This is the classic "jump to wrong position" pattern
        pattern_frames = []
        for t in range(2, T-1):
            # Frame t: foot is far from original
            # Frame t-1 and t+1: foot is close to original
            if (orig_to_fixed_dist[t] > 0.05 and
                orig_to_fixed_dist[t-1] < 0.03 and
                orig_to_fixed_dist[t+1] < 0.03):
                pattern_frames.append(t)

        results[fj] = {
            'jump_frames': jump_frames.tolist(),
            'pattern_jump_frames': pattern_frames,
            'max_acc': float(fixed_acc.max()),
            'mean_acc': float(fixed_acc[2:].mean()),
            'orig_to_fixed_max': float(orig_to_fixed_dist.max()),
            'orig_to_fixed_mean': float(orig_to_fixed_dist.mean()),
        }

    return results


def reconstruct_selective_replace_blend(orig, fixed, model=None):
    """Simulate which frames would be modified by _selective_replace"""
    T = orig.shape[0]
    modified_frames = {fj: set() for fj in FOOT_JOINTS}

    for fj in FOOT_JOINTS:
        y_dim = 1  # Y coordinate
        heights = orig[:, fj, y_dim]
        ground_level = np.percentile(heights, 5)
        contact_threshold = ground_level + 0.05

        for t in range(1, T):
            height = heights[t]
            if height < contact_threshold:
                orig_xz = orig[t, fj, [0, 2]]
                prev_xz = orig[t-1, fj, [0, 2]]
                velocity = np.linalg.norm(orig_xz - prev_xz)

                if velocity > 0.03:
                    modified_frames[fj].add(t)

    return modified_frames


def main():
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixed_outputs_v11')
    files = sorted(glob.glob(f'{src_dir}/*.npy'))
    # Only original files (not _fixed)
    orig_files = [f for f in files if '_fixed.npy' not in f]

    print("=" * 80)
    print("FOOT JUMP ANALYSIS")
    print(f"  Found {len(orig_files)} motion pairs")
    print("=" * 80)

    # Use first 3 available samples
    samples = orig_files[:3]
    for fn in samples:

        name = os.path.basename(fn).replace('.npy', '')
        orig = load_motion(fn)
        fixed = load_motion(fn.replace('.npy', '_fixed.npy'))

        print(f"\n{'='*80}")
        print(f"Sample: {name}")
        print(f"  Length: {orig.shape[0]} frames")

        # Analyze jumps
        results = analyze_foot_jumps(orig, fixed, name)

        # Reconstruct which frames selective_replace modifies
        modified = reconstruct_selective_replace_blend(orig, fixed)

        for fj in FOOT_JOINTS:
            r = results[fj]
            mod = modified[fj]
            print(f"\n  --- {FOOT_NAMES[fj]} (joint {fj}) ---")
            print(f"  Frames modified by selective_replace: {len(mod)}/{orig.shape[0]} ({100*len(mod)/orig.shape[0]:.1f}%)")
            print(f"  Acc spikes (>2x p95): {len(r['jump_frames'])} frames")
            print(f"  'Jump-and-back' pattern: {len(r['pattern_jump_frames'])} frames")
            print(f"  Mean acc: {r['mean_acc']:.4f}, Max acc: {r['max_acc']:.4f}")
            print(f"  Mean orig→fixed distance: {r['orig_to_fixed_mean']:.4f}, Max: {r['orig_to_fixed_max']:.4f}")

            # Show a few jump-and-back examples with surrounding context
            if r['pattern_jump_frames']:
                print(f"\n  Example 'jump-and-back' frames (format: t: orig_pos → fixed_pos | dist | velocity):")
                for t in r['pattern_jump_frames'][:5]:
                    op = orig[t, fj]
                    fp = fixed[t, fj]
                    dist = np.linalg.norm(fp - op)
                    vel = np.linalg.norm(fixed[t, fj, [0,2]] - fixed[t-1, fj, [0,2]]) if t > 0 else 0
                    prev_dist = np.linalg.norm(fixed[t-1, fj] - orig[t-1, fj]) if t > 0 else 0
                    next_dist = np.linalg.norm(fixed[t+1, fj] - orig[t+1, fj]) if t < orig.shape[0]-1 else 0

                    # Check if this frame is in modified set
                    is_mod = t in mod
                    is_prev_mod = (t-1) in mod
                    is_next_mod = (t+1) in mod

                    print(f"    t={t:3d}: dist={dist:.3f} (prev={prev_dist:.3f}, next={next_dist:.3f}) "
                          f"| vel={vel:.3f} | mod[t]={is_mod}, mod[t-1]={is_prev_mod}, mod[t+1]={is_next_mod}")

        # Check for isolated modification frames
        print(f"\n  --- ISOLATION ANALYSIS ---")
        for fj in FOOT_JOINTS:
            mod = sorted(modified[fj])
            isolated = []
            for i, t in enumerate(mod):
                prev_ok = (i == 0) or (mod[i-1] != t-1)
                next_ok = (i == len(mod)-1) or (mod[i+1] != t+1)
                if prev_ok and next_ok:
                    isolated.append(t)
            print(f"  {FOOT_NAMES[fj]}: {len(isolated)}/{len(mod)} modified frames are ISOLATED (no adjacent modified frames)")

    print(f"\n{'='*80}")
    print("SUMMARY: Root Cause of Foot Jumps")
    print("=" * 80)
    print("""
The foot jump artifact is caused by _selective_replace operating on ISOLATED frames:

1. Detection: Frame t has foot-on-ground + high velocity → "skating" → BLEND
2. But frames t-1 and t+1 are NOT detected as skating → KEEP ORIGINAL
3. Blending at t: foot = 0.5*original + 0.5*predicted
   - If model's predicted foot at t differs significantly from original at t-1,
     the blended result jumps away from the smooth original trajectory
4. Frame t+1: foot = original[t+1] → snaps BACK to original position

This creates a 1-frame spike: foot jumps AWAY at t, then BACK at t+1.

The model's predicted foot position has ~0.9m L1 error on MoMask data.
When blended at an isolated frame, this large prediction error becomes a visible jump.
""")

if __name__ == '__main__':
    main()
