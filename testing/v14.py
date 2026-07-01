"""
MotionFix V14 — Batch Test with 7-Metric Evaluation
====================================================
Tests V14 on all 50 MoMask prompts using the unified 7-metric framework.

Metrics (Priority 1+2):
  1. FSR      — Foot Skating Ratio
  2. Jitter   — Foot Acceleration RMS
  3. Floating — Foot hovering above ground during contact
  4. FootErr  — Mean foot position error vs original
  5. BoneCV   — Bone Length Consistency (CV)
  6. Penetration — Ground penetration depth
  7. ContactAcc — Contact label preservation accuracy

Output:
  - outputs/fixed/v14/*.npy (50 fixed motions)
  - outputs/fixed/v14/VERSION.md (version log with all metrics)
  - logs/v14_test.log (test log)
"""

import torch
import numpy as np
import glob
import os
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.v14 import MotionFixNetworkV14
from utils.metrics import (
    evaluate_all, compute_fsr, compute_jitter, compute_floating,
    compute_foot_error, compute_bone_length_consistency,
    compute_ground_penetration, compute_contact_accuracy,
    print_summary_table,
)

# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
OUTPUT_DIR = "outputs/fixed/v14"
LOG_PATH = "logs/v14_test.log"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def log(msg: str, also_print: bool = True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line)


def infer(model, motion: np.ndarray, device: str) -> np.ndarray:
    """Run V14 inference with root_relative conversion."""
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1).astype(np.float32)
    motion_tensor = torch.from_numpy(motion_flat).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        fixed_tensor = model(motion_tensor, foot_only=True, root_relative=True)

    return fixed_tensor.squeeze(0).cpu().numpy().reshape(T, 22, 3)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    log("=" * 70)
    log("  MotionFix V14 — MoMask 50 Batch Test (7-Metric Evaluation)")
    log("=" * 70)
    log(f"  Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Device: {DEVICE}")
    log(f"")

    # ── Load model ──
    ckpt_path = "checkpoints/v14/best.pth"
    if not os.path.exists(ckpt_path):
        log(f"  ❌ ERROR: {ckpt_path} not found. Train V14 first.")
        log(f"     Run: python training/v14.py")
        return

    model = MotionFixNetworkV14(blend_alpha=0.5).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    log(f"  Model:  V14 Transformer Encoder ×6")
    log(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    log(f"  Ckpt:   {ckpt_path} (epoch {ckpt.get('epoch', '?')+1}, loss {ckpt.get('loss', 0):.4f})")
    log(f"  Config: root_relative=True, blend_alpha=0.5, Y-protected")
    log(f"")

    # ── Find test files ──
    files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))
    log(f"  Test motions: {len(files)}")
    log(f"")

    if not files:
        log(f"  ❌ No .npy files found in {INPUT_DIR}")
        return

    # ── Run inference ──
    t_start = time.time()

    # Table header (condensed for display)
    header = (f"{'#':>3} {'Name':<32} {'FSR_bef':>7} {'FSR_aft':>7} "
              f"{'ΔFSR':>7} {'Jit_bef':>7} {'Jit_aft':>7} "
              f"{'Float':>6} {'FtErr':>6} {'CtAcc':>6}")
    log(header)
    log("-" * len(header))

    all_results = []
    all_summary = []

    for i, filepath in enumerate(files):
        name = os.path.basename(filepath).replace('.npy', '')
        short = name[:31]

        # Load
        motion = np.load(filepath).astype(np.float32)
        if motion.ndim == 4:
            motion = motion[0]

        # ── Before metrics ──
        fsr_bef, _, _ = compute_fsr(motion)
        jit_bef = compute_jitter(motion)

        # ── Inference ──
        fixed = infer(model, motion, DEVICE)

        # ── After metrics ──
        fsr_aft, _, _ = compute_fsr(fixed)
        jit_aft = compute_jitter(fixed)
        floating, _, _ = compute_floating(fixed)
        ft_err = compute_foot_error(fixed, motion)
        cont_acc = compute_contact_accuracy(fixed, motion)

        # ── Save ──
        np.save(f"{OUTPUT_DIR}/{name}_fixed.npy", fixed)

        # ── Log ──
        log(f"{i+1:>3} {short:<32} {fsr_bef:>6.1%} {fsr_aft:>6.1%} "
            f"{fsr_aft-fsr_bef:>+6.1%} {jit_bef:>7.4f} {jit_aft:>7.4f} "
            f"{floating:>5.1%} {ft_err:>5.4f} {cont_acc:>5.1%}")

        all_summary.append({
            'name': name,
            'FSR': fsr_aft,
            'Jitter': jit_aft,
            'Floating': floating,
            'FootErr': ft_err,
            'ContactAcc': cont_acc,
            'BoneCV': compute_bone_length_consistency(fixed),
            'PenetrationMean': compute_ground_penetration(fixed)[0],
            'PenetrationMax': compute_ground_penetration(fixed)[1],
            # Before metrics
            'FSR_before': fsr_bef,
            'Jitter_before': jit_bef,
        })

    elapsed = time.time() - t_start
    n = len(files)

    # ── Summary ──
    log("-" * len(header))
    log(f"")

    avg = {k: np.mean([r[k] for r in all_summary]) for k in [
        'FSR', 'Jitter', 'Floating', 'FootErr', 'ContactAcc', 'BoneCV',
        'PenetrationMean', 'PenetrationMax', 'FSR_before', 'Jitter_before'
    ]}
    avg_float_bef, _, _ = compute_floating(motion) if False else (0,0,0)

    log("=" * 70)
    log(f"  📊 V14 Test Summary — {n} MoMask motions")
    log("=" * 70)
    log(f"  Time: {elapsed:.1f}s ({elapsed/n:.2f}s/motion)")
    log(f"")

    # Priority 1
    log(f"  ── Priority 1 (核心) ──")
    log(f"  FSR:           {avg['FSR_before']:>6.2%} → {avg['FSR']:>6.2%}  "
        f"(Δ={avg['FSR']-avg['FSR_before']:+.2%})")
    log(f"  Jitter:        {avg['Jitter_before']:>8.4f} → {avg['Jitter']:>8.4f}  "
        f"(×{avg['Jitter']/max(avg['Jitter_before'], 1e-8):.1f})")
    log(f"  Floating:      {avg['Floating']:>6.2%}")
    log(f"  Foot Error:    {avg['FootErr']:>7.4f} m")
    log(f"")

    # Priority 2
    log(f"  ── Priority 2 (强烈建议) ──")
    log(f"  Contact Acc:   {avg['ContactAcc']:>6.2%}")
    log(f"  Bone CV:       {avg['BoneCV']:>8.4f}")
    log(f"  Penetration:   mean={avg['PenetrationMean']:.4f}m, max={avg['PenetrationMax']:.4f}m")
    log(f"")

    # Comparison with V8
    log(f"  ── Cross-Version Comparison ──")
    log(f"  {'Version':<10} {'FSR':>8} {'Jitter':>8} {'Float':>8} {'FtErr':>8} {'CtAcc':>8} {'BoneCV':>8}")
    log(f"  {'V8_new':<10} {'15.6%':>8} {'0.0278':>8} {'N/A':>8} {'0.0098':>8} {'N/A':>8} {'N/A':>8}")
    log(f"  {'V14':<10} {avg['FSR']:>7.1%} {avg['Jitter']:>8.4f} {avg['Floating']:>7.1%} "
        f"{avg['FootErr']:>7.4f} {avg['ContactAcc']:>7.1%} {avg['BoneCV']:>8.4f}")
    log(f"")

    # Improvement
    fsr_improvement = avg['FSR_before'] - avg['FSR']
    if fsr_improvement > 0.01:
        log(f"  ✅ FSR improved by {fsr_improvement:.1%} — model successfully fixes foot skating")
    elif fsr_improvement > -0.01:
        log(f"  ⚠️  FSR change within ±1% — model has minimal impact")
    else:
        log(f"  ❌ FSR worsened by {-fsr_improvement:.1%} — model introduces new skating")

    jitter_ratio = avg['Jitter'] / max(avg['Jitter_before'], 1e-8)
    if jitter_ratio < 2.0:
        log(f"  ✅ Jitter ratio {jitter_ratio:.1f}× — acceptable (<2×)")
    else:
        log(f"  ⚠️  Jitter ratio {jitter_ratio:.1f}× — high (>2×), check for artifacts")

    log("=" * 70)
    log(f"")
    log(f"  💾 Results saved to: {OUTPUT_DIR}/ ({n} .npy files)")
    log(f"")

    # ── Generate VERSION.md ──
    version_path = f"{OUTPUT_DIR}/VERSION.md"
    with open(version_path, 'w') as f:
        f.write(f"# V14 — Version Log\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model:** V14 Transformer Encoder ×6, d_model=512, nhead=8, 19.1M params\n")
        f.write(f"**Checkpoint:** `{ckpt_path}` (epoch {ckpt.get('epoch', '?')+1})\n\n")

        f.write(f"## Training Paradigm\n\n")
        f.write(f"- **Input:** HumanML3D + simulated foot skating (horizontal drift at contact frames)\n")
        f.write(f"- **Target:** Clean HumanML3D\n")
        f.write(f"- **Distortion:** Only foot joints [7,8,10,11], only XZ plane, only contact frames\n")
        f.write(f"- **Loss:** V14Loss (foot-XZ-focused, λ_foot=3.0, λ_foot_y=0.5)\n\n")

        f.write(f"## Inference Settings\n\n")
        f.write(f"- `foot_only=True`: selective foot replacement\n")
        f.write(f"- `root_relative=True`: world→root-relative→world conversion\n")
        f.write(f"- `blend_alpha=0.5`: 50% original + 50% predicted on XZ only\n")
        f.write(f"- Y-axis protected: height never modified\n\n")

        f.write(f"## Results — MoMask 50 Prompts\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **FSR** | {avg['FSR']:.1%} (was {avg['FSR_before']:.1%}, Δ={avg['FSR']-avg['FSR_before']:+.1%}) |\n")
        f.write(f"| **Jitter** | {avg['Jitter']:.4f} m/frame² (was {avg['Jitter_before']:.4f}) |\n")
        f.write(f"| **Floating** | {avg['Floating']:.1%} |\n")
        f.write(f"| **Foot Error** | {avg['FootErr']:.4f} m |\n")
        f.write(f"| **Contact Accuracy** | {avg['ContactAcc']:.1%} |\n")
        f.write(f"| **Bone Length CV** | {avg['BoneCV']:.4f} |\n")
        f.write(f"| **Penetration (mean)** | {avg['PenetrationMean']:.4f} m |\n")
        f.write(f"| **Penetration (max)** | {avg['PenetrationMax']:.4f} m |\n")
        f.write(f"\n### Per-Motion Results\n\n")
        f.write(f"| # | Name | FSR_bef | FSR_aft | ΔFSR | Jit_bef | Jit_aft | Float | FtErr | CtAcc |\n")
        f.write(f"|---|------|---------|---------|------|---------|---------|-------|-------|-------|\n")
        for i, r in enumerate(all_summary):
            name_short = r['name'][:40]
            f.write(f"| {i+1} | {name_short} | {r['FSR_before']:.1%} | {r['FSR']:.1%} | "
                    f"{r['FSR']-r['FSR_before']:+.1%} | {r['Jitter_before']:.4f} | "
                    f"{r['Jitter']:.4f} | {r['Floating']:.1%} | {r['FootErr']:.4f} | "
                    f"{r['ContactAcc']:.1%} |\n")
        f.write(f"\n**Average** | | {avg['FSR_before']:.1%} | {avg['FSR']:.1%} | "
                f"{avg['FSR']-avg['FSR_before']:+.1%} | {avg['Jitter_before']:.4f} | "
                f"{avg['Jitter']:.4f} | {avg['Floating']:.1%} | {avg['FootErr']:.4f} | "
                f"{avg['ContactAcc']:.1%} |\n")

    log(f"  📝 Version log: {version_path}")
    log(f"")
    log("  ✅ V14 test complete.")


if __name__ == "__main__":
    main()
