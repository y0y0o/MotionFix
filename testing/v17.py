"""
V17 — Test the Hybrid (3-way Ablation + Semantic Proxy + Plots)
================================================================
Held-out 10 motions. Three-way comparison:
  Original  →  Physics-only  →  Physics + Smoother (full hybrid)

Reports:
  - Physical plausibility: 7 metrics (FSR, Jitter, Floating, FootErr, ContactAcc, BoneCV, Penetration)
  - Semantic preservation proxy: % joint-frames modified, non-foot zero-change, foot displacement
  - Ablation: isolates the smoother's contribution (jitter reduction without FSR regrowth)
  - Plots: trajectories + velocity (visual twitch check)
"""
import torch, numpy as np, os, sys, json, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v17 import FootSmoother, hybrid_fix, FOOT_JOINTS
from utils.physics_fix import physics_foot_fix
from utils.metrics import (
    compute_fsr, compute_jitter, compute_floating, compute_foot_error,
    compute_bone_length_consistency, compute_ground_penetration,
    compute_contact_accuracy,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
TEST_NAMES = "data/training/v15/test_names.json"
OUTPUT_DIR = "outputs/fixed/v17"
VIZ_DIR = "analysis/v17_viz"
LOG_PATH = "logs/v17_test.log"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

JOINT_NAMES = {7: "L-Ankle", 8: "R-Ankle", 10: "L-Foot", 11: "R-Foot"}
NON_FOOT = [j for j in range(22) if j not in FOOT_JOINTS]


def log(msg, p=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if p:
        print(line)


def metrics(motion, original):
    return {
        'FSR': compute_fsr(motion)[0],
        'Jitter': compute_jitter(motion),
        'Floating': compute_floating(motion)[0],
        'FootErr': compute_foot_error(motion, original),
        'ContactAcc': compute_contact_accuracy(motion, original),
        'BoneCV': compute_bone_length_consistency(motion),
        'PenMean': compute_ground_penetration(motion)[0],
    }


def semantic_proxy(fixed, original):
    """Quantify how little the motion changed (semantic preservation by construction)."""
    diff = np.linalg.norm(fixed - original, axis=2)          # (T, 22) per-joint displacement
    nonfoot_max = float(diff[:, NON_FOOT].max())            # should be ~0
    foot_disp = float(diff[:, FOOT_JOINTS].mean())          # avg foot displacement
    # % of joint-frames modified beyond 1mm
    modified = float((diff > 0.001).sum()) / diff.size
    return {'nonfoot_max': nonfoot_max, 'foot_disp': foot_disp, 'pct_modified': modified}


def main():
    log("=" * 74)
    log("  V17 — Hybrid Test (Original / Physics / Physics+Smoother)")
    log("=" * 74)

    ck = torch.load("checkpoints/v17/best.pth", map_location=DEVICE)
    smoother = FootSmoother().to(DEVICE)
    smoother.load_state_dict(ck['model_state_dict'])
    smoother.eval()
    log(f"  Smoother: {smoother.n_params():,} params (epoch {ck['epoch']+1}, loss {ck['loss']:.4f})")

    with open(TEST_NAMES) as f:
        test_names = json.load(f)
    log(f"  Held-out motions: {len(test_names)}")
    log("")

    hdr = (f"{'#':>2} {'Name':<30} {'FSR_o':>6} {'FSR_p':>6} {'FSR_h':>6} | "
           f"{'Jit_o':>6} {'Jit_p':>6} {'Jit_h':>6}")
    log(hdr); log("-" * len(hdr))

    R = {'orig': [], 'phys': [], 'hyb': []}
    SP = []
    viz = []

    for i, name in enumerate(test_names):
        orig = np.load(f"{INPUT_DIR}/{name}.npy").astype(np.float32)
        if orig.ndim == 4:
            orig = orig[0]

        hyb, phys = hybrid_fix(orig, smoother, DEVICE, physics_foot_fix, damp_factor=0.0)

        mo, mp, mh = metrics(orig, orig), metrics(phys, orig), metrics(hyb, orig)
        R['orig'].append(mo); R['phys'].append(mp); R['hyb'].append(mh)
        SP.append(semantic_proxy(hyb, orig))

        np.save(f"{OUTPUT_DIR}/{name}_hybrid.npy", hyb)

        log(f"{i+1:>2} {name[:29]:<30} "
            f"{mo['FSR']:>5.1%} {mp['FSR']:>5.1%} {mh['FSR']:>5.1%} | "
            f"{mo['Jitter']:>6.4f} {mp['Jitter']:>6.4f} {mh['Jitter']:>6.4f}")

        if i < 4:
            viz.append((name, orig, phys, hyb))

    log("-" * len(hdr))

    def avg(lst): return {k: float(np.mean([m[k] for m in lst])) for k in lst[0]}
    ao, ap, ah = avg(R['orig']), avg(R['phys']), avg(R['hyb'])
    asp = {k: float(np.mean([s[k] for s in SP])) for k in SP[0]}

    log("")
    log("=" * 74)
    log(f"  📊 ABLATION — Physical Plausibility (n={len(test_names)})")
    log("=" * 74)
    log(f"  {'Metric':<16} {'Original':>11} {'Physics':>11} {'Hybrid(V17)':>12}")
    log(f"  {'─'*16} {'─'*11} {'─'*11} {'─'*12}")
    for k, lbl, pct in [('FSR','FSR ↓',True), ('Jitter','Jitter ↓',False),
                        ('Floating','Floating ↓',True), ('FootErr','FootErr',False),
                        ('ContactAcc','ContactAcc ↑',True), ('BoneCV','BoneCV ↓',False),
                        ('PenMean','Penetration ↓',False)]:
        if pct:
            log(f"  {lbl:<16} {ao[k]:>10.1%} {ap[k]:>10.1%} {ah[k]:>11.1%}")
        else:
            log(f"  {lbl:<16} {ao[k]:>11.4f} {ap[k]:>11.4f} {ah[k]:>12.4f}")

    log("")
    log("  ── Semantic Preservation Proxy (Hybrid vs Original) ──")
    log(f"  Non-foot joints max change: {asp['nonfoot_max']:.6f} m  (≈0 → upper body untouched)")
    log(f"  Avg foot displacement:      {asp['foot_disp']:.4f} m")
    log(f"  Joint-frames modified:      {asp['pct_modified']:.1%}  (only feet at skating frames)")

    log("")
    log("  ── What the smoother adds (Physics → Hybrid) ──")
    d_fsr = ah['FSR'] - ap['FSR']
    d_jit = ah['Jitter'] - ap['Jitter']
    jit_red = (1 - ah['Jitter']/ap['Jitter']) * 100 if ap['Jitter'] > 0 else 0
    log(f"  Jitter: {ap['Jitter']:.4f} → {ah['Jitter']:.4f}  ({jit_red:+.0f}% vs physics)")
    log(f"  FSR:    {ap['FSR']:.1%} → {ah['FSR']:.1%}  (Δ={d_fsr:+.1%} — should stay low)")
    if ah['Jitter'] < ap['Jitter'] and ah['FSR'] <= ap['FSR'] + 0.02:
        log(f"  ✅ Smoother reduces jitter while keeping FSR gain — hybrid works")
    elif ah['Jitter'] < ap['Jitter']:
        log(f"  ⚠️  Smoother reduces jitter but FSR rose {d_fsr:+.1%}")
    else:
        log(f"  ❌ Smoother did not reduce jitter")

    # Full summary vs original
    log("")
    log(f"  🎯 Hybrid vs Original: FSR {ao['FSR']:.1%}→{ah['FSR']:.1%} "
        f"({ah['FSR']-ao['FSR']:+.1%}), Jitter {ao['Jitter']:.4f}→{ah['Jitter']:.4f}")

    # ── Plots ──
    log("")
    log("  Generating plots...")
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle('V17 Hybrid — Foot X Trajectory (Original / Physics / Hybrid)',
                 fontsize=15, fontweight='bold')
    for row, (name, orig, phys, hyb) in enumerate(viz):
        for col, fj in enumerate(FOOT_JOINTS):
            ax = axes[row][col]
            ax.plot(orig[:, fj, 0], color='#bbbbbb', lw=1.2, label='Original')
            ax.plot(phys[:, fj, 0], color='#FF9800', lw=1.0, ls='--', alpha=0.8, label='Physics')
            ax.plot(hyb[:, fj, 0], color='#2196F3', lw=1.6, alpha=0.9, label='Hybrid')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            ax.set_title(f'{name[:20]} {JOINT_NAMES[fj]} X', fontsize=8)
            ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'{VIZ_DIR}/01_trajectories.png', dpi=140, bbox_inches='tight')
    plt.close()
    log(f"  ✅ {VIZ_DIR}/01_trajectories.png")

    # Velocity (twitch check) for first motion
    fig, axes = plt.subplots(4, 1, figsize=(18, 14))
    name, orig, phys, hyb = viz[0]
    fig.suptitle(f'V17 Foot Speed — {name[:40]} (spikes=twitch)', fontsize=14, fontweight='bold')
    for idx, fj in enumerate(FOOT_JOINTS):
        ax = axes[idx]
        for data, c, lbl in [(orig,'#bbbbbb','Original'),(phys,'#FF9800','Physics'),(hyb,'#2196F3','Hybrid')]:
            v = np.linalg.norm(data[1:, fj, :] - data[:-1, fj, :], axis=1)
            ax.plot(v, color=c, lw=1.3, alpha=0.85, label=lbl)
        ax.axhline(0.03, color='red', ls=':', alpha=0.5, label='skate thresh')
        ax.set_title(f'{JOINT_NAMES[fj]} speed', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'{VIZ_DIR}/02_velocity.png', dpi=140, bbox_inches='tight')
    plt.close()
    log(f"  ✅ {VIZ_DIR}/02_velocity.png")

    # Save summary
    with open(f'{VIZ_DIR}/summary.json', 'w') as f:
        json.dump({'original': ao, 'physics': ap, 'hybrid': ah,
                   'semantic_proxy': asp, 'n': len(test_names)}, f, indent=2)
    log(f"  ✅ {VIZ_DIR}/summary.json")
    log("=" * 74)
    log(f"  💾 Outputs: {OUTPUT_DIR}/  |  Plots+summary: {VIZ_DIR}/")
    log("  ✅ Done.")


if __name__ == "__main__":
    main()
