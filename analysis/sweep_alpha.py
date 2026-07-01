"""
Quick test: sweep blend_alpha on V14 to find optimal FSR vs Jitter tradeoff.
Runs on a single motion (p000021) then on all 50 if promising.
"""
import torch
import numpy as np
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v14 import MotionFixNetworkV14
from utils.metrics import compute_fsr, compute_jitter, compute_floating, compute_foot_error

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOTION_NAME = "p000021_rotation_person_is_walking_normally_in_a"
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"

def infer(model, motion, device, alpha=None):
    """Run inference, optionally overriding blend_alpha."""
    if alpha is not None:
        old_alpha = model.blend_alpha
        model.blend_alpha = alpha

    T = motion.shape[0]
    flat = motion.reshape(T, -1).astype(np.float32)
    tensor = torch.from_numpy(flat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        result = model(tensor, foot_only=True, root_relative=True)
    fixed = result.squeeze(0).cpu().numpy().reshape(T, 22, 3)

    if alpha is not None:
        model.blend_alpha = old_alpha
    return fixed

# Load
model = MotionFixNetworkV14(blend_alpha=0.5).to(DEVICE)
ckpt = torch.load("checkpoints/v14/best.pth", map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

motion = np.load(f"{INPUT_DIR}/{MOTION_NAME}.npy").astype(np.float32)
if motion.ndim == 4: motion = motion[0]

print(f"Model: V14 (epoch {ckpt['epoch']+1})")
print(f"Motion: {MOTION_NAME[:50]}, T={motion.shape[0]}")
print()
print(f"{'Alpha':>6} {'FSR_bef':>7} {'FSR_aft':>7} {'ΔFSR':>7} {'Jit_bef':>7} {'Jit_aft':>7} {'Ratio':>6} {'Float':>6} {'FtErr':>6}")
print("-" * 73)

fsr_bef, _, _ = compute_fsr(motion)
jit_bef = compute_jitter(motion)

best_alpha = 0.5
best_fsr = fsr_bef

for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
    fixed = infer(model, motion, DEVICE, alpha=alpha)
    fsr_aft, _, _ = compute_fsr(fixed)
    jit_aft = compute_jitter(fixed)
    floating, _, _ = compute_floating(fixed)
    ft_err = compute_foot_error(fixed, motion)

    print(f"{alpha:>6.2f} {fsr_bef:>6.1%} {fsr_aft:>6.1%} "
          f"{fsr_aft-fsr_bef:>+6.1%} {jit_bef:>7.4f} {jit_aft:>7.4f} "
          f"{jit_aft/max(jit_bef,1e-8):>5.1f}x {floating:>5.1%} {ft_err:>5.4f}")

    if fsr_aft < best_fsr:
        best_fsr = fsr_aft
        best_alpha = alpha

print(f"\n✅ Best alpha: {best_alpha} (FSR: {best_fsr:.1%})")

# Quick check on top 3 candidates across more motions
print(f"\n{'='*70}")
print(f"Testing top candidates (alpha={best_alpha}, 0.7, 0.9) on 10 motions...")
print(f"{'='*70}")

files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))[:10]
for test_alpha in sorted(set([best_alpha, 0.7, 0.9])):
    fsrs, jits, floats, ferrs = [], [], [], []
    for fp in files:
        m = np.load(fp).astype(np.float32)
        if m.ndim == 4: m = m[0]
        fixed = infer(model, m, DEVICE, alpha=test_alpha)
        fsrs.append(compute_fsr(fixed)[0])
        jits.append(compute_jitter(fixed))
        floats.append(compute_floating(fixed)[0])
        ferrs.append(compute_foot_error(fixed, m))

    print(f"  alpha={test_alpha:.2f}: FSR={np.mean(fsrs):.1%}, "
          f"Jitter={np.mean(jits):.4f}, Float={np.mean(floats):.1%}, "
          f"FtErr={np.mean(ferrs):.4f}")
