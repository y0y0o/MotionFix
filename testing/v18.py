"""
V18 — Test: 5-way Ablation + Semantic Proxy + Plots
====================================================
Original / Physics(damp0) / Analytical(V18 res=0) / V18(learned) / [V17 for context]

The decisive questions:
  1. Does V18 keep FSR + Jitter below original (break the antagonism)?
  2. Does the LEARNED model control the 38cm drift of the analytical baseline?
"""
import torch, numpy as np, os, sys, json, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import FootRefiner, v18_fix, analytical_fix, FOOT_JOINTS
from utils.physics_fix import physics_foot_fix
from utils.metrics import (compute_fsr, compute_jitter, compute_floating,
    compute_foot_error, compute_bone_length_consistency,
    compute_ground_penetration, compute_contact_accuracy)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
TEST_NAMES = "data/training/v15/test_names.json"
OUTPUT_DIR = "outputs/fixed/v18"
VIZ_DIR = "analysis/v18_viz"
LOG_PATH = "logs/v18_test.log"
for d in (OUTPUT_DIR, VIZ_DIR, os.path.dirname(LOG_PATH)):
    os.makedirs(d, exist_ok=True)

JOINT_NAMES = {7: "L-Ankle", 8: "R-Ankle", 10: "L-Foot", 11: "R-Foot"}


def log(msg, p=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if p:
        print(line)


def M(motion, original):
    return {
        'FSR': compute_fsr(motion)[0], 'Jitter': compute_jitter(motion),
        'Floating': compute_floating(motion)[0], 'FootErr': compute_foot_error(motion, original),
        'ContactAcc': compute_contact_accuracy(motion, original),
        'BoneCV': compute_bone_length_consistency(motion),
        'PenMean': compute_ground_penetration(motion)[0],
    }


def main():
    log("=" * 76)
    log("  V18 — 4-way Ablation (Original / Physics / Analytical / V18-learned)")
    log("=" * 76)

    ck = torch.load("checkpoints/v18/best.pth", map_location=DEVICE)
    model = FootRefiner().to(DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    log(f"  V18 FootRefiner: {model.n_params():,} params (epoch {ck['epoch']+1}, loss {ck['loss']:.5f})")

    with open(TEST_NAMES) as f:
        names = json.load(f)
    log(f"  Held-out: {len(names)} motions")
    log("")

    hdr = (f"{'#':>2} {'Name':<26} {'FSR_o':>5} {'FSR_p':>5} {'FSR_a':>5} {'FSR_v':>5} | "
           f"{'Jit_o':>6} {'Jit_v':>6} | {'FtE_a':>5} {'FtE_v':>5}")
    log(hdr); log("-" * len(hdr))

    R = {'orig': [], 'phys': [], 'ana': [], 'v18': []}
    viz = []

    for i, nm in enumerate(names):
        m = np.load(f"{INPUT_DIR}/{nm}.npy").astype(np.float32)
        if m.ndim == 4:
            m = m[0]
        phys, _ = physics_foot_fix(m, damp_factor=0.0, return_stats=True)
        ana = analytical_fix(m)
        v18 = v18_fix(m, model, DEVICE)

        mo, mp, ma, mv = M(m, m), M(phys, m), M(ana, m), M(v18, m)
        R['orig'].append(mo); R['phys'].append(mp); R['ana'].append(ma); R['v18'].append(mv)

        np.save(f"{OUTPUT_DIR}/{nm}_v18.npy", v18)
        log(f"{i+1:>2} {nm[:25]:<26} {mo['FSR']:>4.1%} {mp['FSR']:>4.1%} {ma['FSR']:>4.1%} {mv['FSR']:>4.1%} | "
            f"{mo['Jitter']:>6.4f} {mv['Jitter']:>6.4f} | {ma['FootErr']:>5.3f} {mv['FootErr']:>5.3f}")
        if i < 4:
            viz.append((nm, m, phys, ana, v18))

    log("-" * len(hdr))

    def avg(L): return {k: float(np.mean([x[k] for x in L])) for k in L[0]}
    ao, ap, aa, av = avg(R['orig']), avg(R['phys']), avg(R['ana']), avg(R['v18'])

    log("")
    log("=" * 76)
    log(f"  📊 4-WAY ABLATION (n={len(names)})")
    log("=" * 76)
    log(f"  {'Metric':<14} {'Original':>10} {'Physics':>10} {'Analytical':>11} {'V18(learn)':>11}")
    log(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*11} {'─'*11}")
    for k, lbl, pct in [('FSR','FSR ↓',True),('Jitter','Jitter ↓',False),
                        ('Floating','Floating ↓',True),('FootErr','FootErr ↓',False),
                        ('ContactAcc','ContactAcc ↑',True),('BoneCV','BoneCV ↓',False),
                        ('PenMean','Penetration↓',False)]:
        if pct:
            log(f"  {lbl:<14} {ao[k]:>9.1%} {ap[k]:>9.1%} {aa[k]:>10.1%} {av[k]:>10.1%}")
        else:
            log(f"  {lbl:<14} {ao[k]:>10.4f} {ap[k]:>10.4f} {aa[k]:>11.4f} {av[k]:>11.4f}")

    log("")
    log("  ── Verdict ──")
    # Q1: both FSR and Jitter below original?
    if av['FSR'] < ao['FSR'] and av['Jitter'] < ao['Jitter']:
        log(f"  ✅ V18 beats Original on BOTH: FSR {ao['FSR']:.1%}→{av['FSR']:.1%}, "
            f"Jitter {ao['Jitter']:.4f}→{av['Jitter']:.4f} — antagonism BROKEN")
    elif av['FSR'] < ao['FSR']:
        log(f"  ⚠️  V18 FSR↓ ({av['FSR']:.1%}) but Jitter {av['Jitter']:.4f} vs orig {ao['Jitter']:.4f}")
    else:
        log(f"  ❌ V18 FSR {av['FSR']:.1%} not below original {ao['FSR']:.1%}")
    # Q2: did learning control the drift?
    log(f"  Drift control (learned vs analytical): FootErr {aa['FootErr']:.3f}m → {av['FootErr']:.3f}m "
        f"({(1-av['FootErr']/max(aa['FootErr'],1e-6))*100:+.0f}%)")
    # Pareto vs physics
    if av['FSR'] <= ap['FSR'] and av['Jitter'] < ap['Jitter']:
        log(f"  ✅ V18 Pareto-dominates Physics (FSR≤ and Jitter<)")
    # vs SOTA
    log(f"  vs SOTA: OmniControl/MaskControl FSR≈5.5% | V18 FSR={av['FSR']:.1%}")

    # ── Plots ──
    log("")
    log("  Generating plots...")
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle('V18 — Foot X Trajectory (Original/Physics/Analytical/V18)',
                 fontsize=15, fontweight='bold')
    for row, (nm, m, phys, ana, v18) in enumerate(viz):
        for col, fj in enumerate(FOOT_JOINTS):
            ax = axes[row][col]
            ax.plot(m[:, fj, 0], color='#bbbbbb', lw=1.3, label='Original')
            ax.plot(phys[:, fj, 0], color='#FF9800', lw=0.9, ls='--', alpha=0.7, label='Physics')
            ax.plot(ana[:, fj, 0], color='#9C27B0', lw=0.9, ls=':', alpha=0.7, label='Analytical')
            ax.plot(v18[:, fj, 0], color='#2196F3', lw=1.7, alpha=0.9, label='V18')
            if row == 0 and col == 0: ax.legend(fontsize=7)
            ax.set_title(f'{nm[:18]} {JOINT_NAMES[fj]} X', fontsize=8)
            ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(f'{VIZ_DIR}/01_trajectories.png', dpi=140, bbox_inches='tight'); plt.close()
    log(f"  ✅ {VIZ_DIR}/01_trajectories.png")

    fig, axes = plt.subplots(4, 1, figsize=(18, 14))
    nm, m, phys, ana, v18 = viz[0]
    fig.suptitle(f'V18 Foot Speed — {nm[:40]} (spikes=twitch)', fontsize=14, fontweight='bold')
    for idx, fj in enumerate(FOOT_JOINTS):
        ax = axes[idx]
        for data, c, lbl in [(m,'#bbbbbb','Original'),(phys,'#FF9800','Physics'),(v18,'#2196F3','V18')]:
            v = np.linalg.norm(data[1:, fj, :] - data[:-1, fj, :], axis=1)
            ax.plot(v, color=c, lw=1.3, alpha=0.85, label=lbl)
        ax.axhline(0.03, color='red', ls=':', alpha=0.5)
        ax.set_title(f'{JOINT_NAMES[fj]} speed', fontsize=10); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(f'{VIZ_DIR}/02_velocity.png', dpi=140, bbox_inches='tight'); plt.close()
    log(f"  ✅ {VIZ_DIR}/02_velocity.png")

    with open(f'{VIZ_DIR}/summary.json', 'w') as f:
        json.dump({'original': ao, 'physics': ap, 'analytical': aa, 'v18': av, 'n': len(names)}, f, indent=2)
    log(f"  ✅ {VIZ_DIR}/summary.json")
    log("=" * 76)
    log(f"  💾 {OUTPUT_DIR}/  |  {VIZ_DIR}/")
    log("  ✅ Done.")


if __name__ == "__main__":
    main()
