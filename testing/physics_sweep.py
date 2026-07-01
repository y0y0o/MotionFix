"""
Physics-based foot skating correction — damp_factor sweep + 50-motion test.

Finds optimal damp_factor balancing FSR reduction vs Jitter control.
Logs all results to logs/physics_fix.log
"""
import numpy as np
import glob, os, sys, json, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.physics_fix import physics_foot_fix, print_stats
from utils.metrics import (
    compute_fsr, compute_jitter, compute_floating, compute_foot_error,
    compute_bone_length_consistency, compute_ground_penetration,
    compute_contact_accuracy,
)

INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
OUTPUT_DIR = "outputs/fixed/physics"
LOG_PATH = "logs/physics_fix.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg, also_print=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line)


# ═══════════════════════════════════════════════════════════════
# Phase 1: Sweep damp_factor on a single motion
# ═══════════════════════════════════════════════════════════════

log("=" * 70)
log("  Physics-Based Foot Skating Correction — Sweep + Test")
log("=" * 70)
log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("")

# Load test data
files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))
n_files = len(files)
log(f"  Test motions: {n_files}")

# Load one motion for sweep
MOTION_NAME = "p000021_rotation_person_is_walking_normally_in_a"
motion = np.load(f"{INPUT_DIR}/{MOTION_NAME}.npy").astype(np.float32)
if motion.ndim == 4:
    motion = motion[0]

# Baseline
fsr_baseline, _, _ = compute_fsr(motion)
jit_baseline = compute_jitter(motion)
log(f"\n  Sweep baseline ({MOTION_NAME[:50]}):")
log(f"    FSR={fsr_baseline:.1%}, Jitter={jit_baseline:.4f}")

# Sweep damp_factor
log(f"\n  {'damp':>7} {'FSR':>7} {'ΔFSR':>7} {'Jitter':>8} {'Ratio':>6} {'Float':>6} {'FtErr':>6} {'CtAcc':>6}")
log(f"  {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

best_fsr = fsr_baseline
best_factor = 1.0

for damp in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]:
    fixed, stats = physics_foot_fix(motion, damp_factor=damp, return_stats=True)

    fsr, c_n, s_n = compute_fsr(fixed)
    jit = compute_jitter(fixed)
    flt, _, _ = compute_floating(fixed)
    fte = compute_foot_error(fixed, motion)
    cta = compute_contact_accuracy(fixed, motion)

    log(f"  {damp:>7.2f} {fsr:>6.1%} {fsr-fsr_baseline:>+6.1%} "
        f"{jit:>8.4f} {jit/max(jit_baseline,1e-8):>5.1f}x "
        f"{flt:>5.1%} {fte:>5.4f} {cta:>5.1%}")

    if fsr < best_fsr:
        best_fsr = fsr
        best_factor = damp

log(f"\n  ✅ Best damp_factor: {best_factor} (FSR: {best_fsr:.1%})")

# ═══════════════════════════════════════════════════════════════
# Phase 2: Test top candidates on all 50 motions
# ═══════════════════════════════════════════════════════════════

candidates = sorted(set([best_factor, 0.05, 0.1, 0.15, 0.2, 0.3]))
log(f"\n{'=' * 70}")
log(f"  Phase 2: Testing top candidates on {n_files} motions")
log(f"  Candidates: {candidates}")
log(f"{'=' * 70}")

all_versions = {}

for damp in candidates:
    log(f"\n  ── damp_factor={damp} ──")

    results = []
    for i, fp in enumerate(files):
        name = os.path.basename(fp).replace('.npy', '')
        m = np.load(fp).astype(np.float32)
        if m.ndim == 4:
            m = m[0]

        fsr_b, _, _ = compute_fsr(m)
        jit_b = compute_jitter(m)

        fixed, stats = physics_foot_fix(m, damp_factor=damp, return_stats=True)

        fsr_a, _, _ = compute_fsr(fixed)
        jit_a = compute_jitter(fixed)
        flt, _, _ = compute_floating(fixed)
        fte = compute_foot_error(fixed, m)
        cta = compute_contact_accuracy(fixed, m)
        bcv = compute_bone_length_consistency(fixed)
        pen_m, pen_x, _ = compute_ground_penetration(fixed)

        # Save best (at best_factor)
        if damp == best_factor:
            np.save(f"{OUTPUT_DIR}/{name}_fixed.npy", fixed)

        results.append({
            'name': name,
            'FSR_before': fsr_b, 'FSR_after': fsr_a,
            'Jitter_before': jit_b, 'Jitter_after': jit_a,
            'Floating': flt, 'FootErr': fte, 'ContactAcc': cta,
            'BoneCV': bcv, 'PenMean': pen_m, 'PenMax': pen_x,
        })

        if (i + 1) % 25 == 0:
            log(f"    {i+1}/{n_files} done")

    all_versions[damp] = results

    # Per-candidate summary
    avg = {}
    for k in ['FSR_before', 'FSR_after', 'Jitter_before', 'Jitter_after',
              'Floating', 'FootErr', 'ContactAcc', 'BoneCV', 'PenMean']:
        vals = [r[k] for r in results]
        avg[k] = float(np.mean(vals))

    log(f"    Summary (n={len(results)}):")
    log(f"    FSR: {avg['FSR_before']:.1%} → {avg['FSR_after']:.1%}  "
        f"(Δ={avg['FSR_after']-avg['FSR_before']:+.1%})")
    log(f"    Jitter: {avg['Jitter_before']:.4f} → {avg['Jitter_after']:.4f}  "
        f"(×{avg['Jitter_after']/max(avg['Jitter_before'],1e-8):.1f})")
    log(f"    Floating: {avg['Floating']:.1%}  |  FootErr: {avg['FootErr']:.4f}m")
    log(f"    ContactAcc: {avg['ContactAcc']:.1%}  |  BoneCV: {avg['BoneCV']:.4f}")

# ═══════════════════════════════════════════════════════════════
# Phase 3: Final comparison
# ═══════════════════════════════════════════════════════════════

log(f"\n{'=' * 80}")
log(f"  📊 FINAL COMPARISON — All Versions on 50 MoMask Motions")
log(f"{'=' * 80}")

log(f"  {'Version':<18} {'FSR':>8} {'Jitter':>8} {'Float':>7} {'FtErr':>7} {'CtAcc':>7} {'BoneCV':>7} {'PenMean':>7}")
log(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

orig_avg = {}
for k in ['FSR_before', 'Jitter_before']:
    orig_avg[k] = float(np.mean([r[k] for r in results]))

log(f"  {'Original':<18} {orig_avg['FSR_before']:>7.1%} {orig_avg['Jitter_before']:>8.4f} "
    f"{'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7}")

for damp in candidates:
    r = all_versions[damp]
    avg = {}
    for k in ['FSR_after', 'Jitter_after', 'Floating', 'FootErr',
              'ContactAcc', 'BoneCV', 'PenMean']:
        vals = [rr[k] for rr in r]
        avg[k] = float(np.mean(vals))

    label = f"Physics damp={damp}"
    log(f"  {label:<18} {avg['FSR_after']:>7.1%} {avg['Jitter_after']:>8.4f} "
        f"{avg['Floating']:>6.1%} {avg['FootErr']:>6.4f} "
        f"{avg['ContactAcc']:>6.1%} {avg['BoneCV']:>7.4f} {avg['PenMean']:>6.4f}")

# Previous versions for context
log(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
log(f"  {'V8_new (α=0.5)':<18} {'15.6%':>8} {'0.0278':>8} {'—':>7} {'0.0098':>7} {'—':>7} {'—':>7} {'—':>7}")
log(f"  {'V14 (α=0.5)':<18} {'15.6%':>8} {'0.0286':>8} {'0.0%':>7} {'0.0102':>7} {'100%':>7} {'0.0212':>7} {'0.0046m':>7}")
log(f"  {'V14 (α=0.9)':<18} {'14.1%':>8} {'0.0476':>8} {'0.0%':>7} {'0.0183':>7} {'100%':>7} {'0.0343':>7} {'0.0046m':>7}")

# Best physics
best_res = all_versions[best_factor]
best_avg = {}
for k in ['FSR_after', 'Jitter_after', 'Floating', 'FootErr',
          'ContactAcc', 'BoneCV', 'PenMean']:
    vals = [rr[k] for rr in best_res]
    best_avg[k] = float(np.mean(vals))

fsr_orig_avg = float(np.mean([rr['FSR_before'] for rr in best_res]))
jit_orig_avg = float(np.mean([rr['Jitter_before'] for rr in best_res]))

log(f"\n{'=' * 80}")
log(f"  🏆 BEST: damp_factor={best_factor}")
log(f"  FSR:      {fsr_orig_avg:.1%} → {best_avg['FSR_after']:.1%}  "
    f"(Δ={best_avg['FSR_after']-fsr_orig_avg:+.1%})")
log(f"  Jitter:   {jit_orig_avg:.4f} → {best_avg['Jitter_after']:.4f}  "
    f"(×{best_avg['Jitter_after']/max(jit_orig_avg,1e-8):.1f})")
log(f"  Floating: {best_avg['Floating']:.1%}")
log(f"  FootErr:  {best_avg['FootErr']:.4f}m")
log(f"  Contact:  {best_avg['ContactAcc']:.1%}")
log(f"  BoneCV:   {best_avg['BoneCV']:.4f}")
log(f"  PenMean:  {best_avg['PenMean']:.4f}m")
log(f"{'=' * 80}")

log(f"\n  💾 Results saved to: {OUTPUT_DIR}/")
log(f"  📝 Log: {LOG_PATH}")
log(f"  ✅ Done.")
