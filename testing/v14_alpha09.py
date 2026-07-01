"""V14 batch test with blend_alpha=0.9"""
import torch, numpy as np, glob, os, sys, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v14 import MotionFixNetworkV14
from utils.metrics import (compute_fsr, compute_jitter, compute_floating,
    compute_foot_error, compute_bone_length_consistency,
    compute_ground_penetration, compute_contact_accuracy)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
OUTPUT_DIR = "outputs/fixed/v14_alpha09"
BLEND_ALPHA = 0.9

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = MotionFixNetworkV14(blend_alpha=BLEND_ALPHA).to(DEVICE)
ckpt = torch.load("checkpoints/v14/best.pth", map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))
print(f"{'='*70}")
print(f"  V14 alpha={BLEND_ALPHA} — 50 MoMask test")
print(f"{'='*70}")

hdr = f"{'#':>3} {'Name':<32} {'FSR_bef':>7} {'FSR_aft':>7} {'ΔFSR':>7} {'Jit_bef':>7} {'Jit_aft':>7} {'Float':>6} {'FtErr':>6} {'CtAcc':>6}"
print(hdr); print("-"*len(hdr))

all_res = []
t0 = time.time()
for i, fp in enumerate(files):
    name = os.path.basename(fp).replace('.npy','')
    m = np.load(fp).astype(np.float32)
    if m.ndim == 4: m = m[0]
    T = m.shape[0]

    fsr_b, _, _ = compute_fsr(m); jit_b = compute_jitter(m)

    flat = m.reshape(T,-1).astype(np.float32)
    tensor = torch.from_numpy(flat).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        fixed = model(tensor, foot_only=True, root_relative=True)
    fixed = fixed.squeeze(0).cpu().numpy().reshape(T,22,3)

    fsr_a, _, _ = compute_fsr(fixed)
    jit_a = compute_jitter(fixed)
    flt, _, _ = compute_floating(fixed)
    fte = compute_foot_error(fixed, m)
    cta = compute_contact_accuracy(fixed, m)

    print(f"{i+1:>3} {name[:31]:<32} {fsr_b:>6.1%} {fsr_a:>6.1%} {fsr_a-fsr_b:>+6.1%} "
          f"{jit_b:>7.4f} {jit_a:>7.4f} {flt:>5.1%} {fte:>5.4f} {cta:>5.1%}")

    np.save(f"{OUTPUT_DIR}/{name}_fixed.npy", fixed)
    all_res.append({'name':name,'FSR_b':fsr_b,'FSR_a':fsr_a,'Jit_b':jit_b,
        'Jit_a':jit_a,'Float':flt,'FtErr':fte,'CtAcc':cta,
        'BoneCV':compute_bone_length_consistency(fixed),
        'PenMean':compute_ground_penetration(fixed)[0],
        'PenMax':compute_ground_penetration(fixed)[1]})

# Summary
n = len(all_res)
avg = {k: np.mean([r[k] for r in all_res]) for k in
    ['FSR_b','FSR_a','Jit_b','Jit_a','Float','FtErr','CtAcc','BoneCV','PenMean','PenMax']}

print("-"*len(hdr))
print(f"\n{'='*70}")
print(f"  V14 alpha={BLEND_ALPHA} Summary (n={n})")
print(f"  Time: {time.time()-t0:.1f}s")
print(f"  ── Priority 1 ──")
print(f"  FSR:        {avg['FSR_b']:.1%} → {avg['FSR_a']:.1%}  (Δ={avg['FSR_a']-avg['FSR_b']:+.1%})")
print(f"  Jitter:     {avg['Jit_b']:.4f} → {avg['Jit_a']:.4f}  (×{avg['Jit_a']/max(avg['Jit_b'],1e-8):.1f})")
print(f"  Floating:   {avg['Float']:.1%}")
print(f"  Foot Error: {avg['FtErr']:.4f}m")
print(f"  ── Priority 2 ──")
print(f"  Contact Acc: {avg['CtAcc']:.1%}")
print(f"  Bone CV:     {avg['BoneCV']:.4f}")
print(f"  Penetration: mean={avg['PenMean']:.4f}m max={avg['PenMax']:.4f}m")

# Comparison table
print(f"\n  ── Comparison ──")
print(f"  {'Version':<15} {'Alpha':>6} {'FSR':>8} {'Jitter':>8} {'Float':>8} {'FtErr':>8}")
print(f"  {'Original':<15} {'-':>6} {avg['FSR_b']:>7.1%} {avg['Jit_b']:>8.4f} {'-':>8} {'-':>8}")
print(f"  {'V8_new':<15} {'0.5':>6} {'15.6%':>8} {'0.0278':>8} {'-':>8} {'0.0098':>8}")
print(f"  {'V14':<15} {'0.5':>6} {'15.6%':>8} {'0.0286':>8} {'0.0%':>8} {'0.0102':>8}")
print(f"  {'V14':<15} {'0.9':>6} {avg['FSR_a']:>7.1%} {avg['Jit_a']:>8.4f} {avg['Float']:>7.1%} {avg['FtErr']:>7.4f}")
print(f"{'='*70}")
