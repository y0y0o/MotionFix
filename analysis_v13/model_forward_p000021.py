"""
Inference-only analysis: run V13 model forward on p000021 to get both
full_output (raw prediction) and foot_only (selective_replace) for comparison.
"""
import torch
import numpy as np
import sys
sys.path.insert(0, 'motionfix_v13')
from motionfix_model_v13 import MotionFixNetworkV13

# Load p000021 original MoMask result
original = np.load("momask_50_results/no_ik/p000021_rotation_person_is_walking_normally_in_a.npy").astype(np.float32)
if len(original.shape) == 4:
    original = original[0]

T = original.shape[0]
print(f"Motion: T={T}, shape={original.shape}")

# Load V13 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MotionFixNetworkV13(blend_alpha=0.5).to(device)
ckpt = torch.load("motionfix_v13/checkpoints_v13/latest.pth", map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Flatten to (T, 66) as model expects
original_flat = original.reshape(T, -1).astype(np.float32)
x = torch.from_numpy(original_flat).unsqueeze(0).to(device)  # (1, T, 66)
print(f"Input shape: {x.shape}")

with torch.no_grad():
    full_output = model(x, foot_only=False)  # Raw prediction (1, T, 66)
    foot_output = model(x, foot_only=True)   # Selective replace output (1, T, 66)

full_flat = full_output.cpu().numpy()[0]   # (T, 66)
foot_flat = foot_output.cpu().numpy()[0]   # (T, 66)
full_output = full_flat.reshape(T, 22, 3)  # (T, 22, 3)
foot_output = foot_flat.reshape(T, 22, 3)  # (T, 22, 3)

# Compare
FOOT_JOINTS = [7, 8, 10, 11]
JOINT_NAMES = {7: "Left Ankle", 8: "Right Ankle", 10: "Left Foot", 11: "Right Foot"}

print(f"\n{'='*70}")
print(f"{'Joint':<15} {'full_output mean err':<20} {'full_output max err':<20} {'foot_output mean err':<20} {'foot_output max err':<20}")
print(f"{'='*70}")

for fj in FOOT_JOINTS:
    full_err = np.linalg.norm(full_output[:, fj, :] - original[:, fj, :], axis=1)
    foot_err = np.linalg.norm(foot_output[:, fj, :] - original[:, fj, :], axis=1)
    print(f"{JOINT_NAMES[fj]:<15} {full_err.mean():<20.4f} {full_err.max():<20.4f} {foot_err.mean():<20.4f} {foot_err.max():<20.4f}")

# Check: how many frames does full_output displace feet by >1m?
print(f"\n{'='*70}")
print("FULL OUTPUT: frames with foot joint displacement > 1m")
print(f"{'='*70}")
for fj in FOOT_JOINTS:
    err = np.linalg.norm(full_output[:, fj, :] - original[:, fj, :], axis=1)
    big_err_frames = np.where(err > 1.0)[0]
    print(f"\n{JOINT_NAMES[fj]}: {len(big_err_frames)}/{T} frames have >1m error")
    for t in big_err_frames[:5]:
        print(f"  frame {t}: {err[t]:.3f}m, full=({full_output[t,fj,0]:.3f},{full_output[t,fj,1]:.3f},{full_output[t,fj,2]:.3f}) orig=({original[t,fj,0]:.3f},{original[t,fj,1]:.3f},{original[t,fj,2]:.3f})")

# Check non-foot joints (to verify they're untouched in foot_only)
print(f"\n{'='*70}")
print("NON-FOOT JOINT CHECK: are they correctly untouched in foot_only mode?")
print(f"{'='*70}")
max_diff_non_foot = 0
for j in range(22):
    if j not in FOOT_JOINTS:
        diff = np.abs(foot_output[:, j, :] - original[:, j, :]).max()
        if diff > max_diff_non_foot:
            max_diff_non_foot = diff
print(f"Max non-foot joint diff in foot_only mode: {max_diff_non_foot:.6f} (should be 0)")

# How many foot joint frames are actually modified in foot_only?
print(f"\n{'='*70}")
print("FOOT_ONLY: frames modified per joint")
print(f"{'='*70}")
for fj in FOOT_JOINTS:
    diff = np.linalg.norm(foot_output[:, fj, :] - original[:, fj, :], axis=1)
    modified = np.sum(diff > 0.0001)
    print(f"{JOINT_NAMES[fj]}: {modified}/{T} frames modified")

# Print the full_output X coordinates to understand the model's "intent"
print(f"\n{'='*70}")
print("FULL OUTPUT vs ORIGINAL X-COORDINATE comparison (Left Ankle)")
print(f"{'='*70}")
print(f"{'frame':<8} {'orig_X':<12} {'full_X':<12} {'delta_X':<12} {'orig_Y':<12}")
for t in range(0, T, 10):
    ox = original[t, 7, 0]
    fx = full_output[t, 7, 0]
    oy = original[t, 7, 1]
    delta = fx - ox
    marker = " ***" if abs(delta) > 1.0 else ""
    print(f"{t:<8} {ox:<12.3f} {fx:<12.3f} {delta:<+12.3f} {oy:<12.3f}{marker}")
