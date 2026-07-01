"""
V18+IK — FINAL Test: 5-way Ablation + Plots
===========================================
Original / Physics / De-skate+IK / Gaussian+IK / Learned+IK

The full method = de-skate (plant-at-mean, no drift) → smoother → 2-bone IK.
Ablation isolates each stage:
  - De-skate+IK : no smoothing (jitter high, FSR/bones good)        [physics only]
  - Gaussian+IK : fixed global σ=1.5 smoothing                       [non-adaptive]
  - Learned+IK  : adaptive smoother (smooth in air, hold at contact) [our learning]

Decisive questions:
  1. Bones intact?  BoneCV ≈ original (leg-tear fixed).
  2. FSR + Jitter both below original?
  3. Does the LEARNED smoother beat the fixed Gaussian (FSR at equal Jitter)?
"""
import torch, numpy as np, os, sys, json
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import (FootRefiner, smooth_fix, deskate_xz, FOOT_JOINTS,
                        FOOT_XZ_DIMS, compute_contact_weight_np, deskated_target, FOOT_Y_DIMS)
from models.v18_ik import apply_ik, LEGS
from utils.physics_fix import physics_foot_fix
from utils.metrics import (compute_fsr, compute_jitter, compute_floating,
    compute_foot_error, compute_bone_length_consistency,
    compute_ground_penetration, compute_contact_accuracy)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
TEST_NAMES = "data/training/v15/test_names.json"
OUTPUT_DIR = "outputs/fixed/v18_ik"
VIZ_DIR = "analysis/v18_ik_viz"
LOG_PATH = "logs/v18_ik_test.log"
GAUSS_SIGMA = 1.5
for d in (OUTPUT_DIR, VIZ_DIR, os.path.dirname(LOG_PATH)):
    os.makedirs(d, exist_ok=True)


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


def deskate_only(m, sigma=0.0):
    """Plant-at-mean de-skating (+ optional global Gaussian). Foot XZ only."""
    T = m.shape[0]; flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    if sigma > 0:
        tgt = gaussian_filter1d(tgt, sigma=sigma, axis=0, mode='nearest')
    out = flat.copy(); out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def maxshin(m):
    return max(np.linalg.norm(m[:, l['ankle']] - m[:, l['knee']], axis=1).max() for l in LEGS)


def main():
    log("=" * 84)
    log("  V18+IK FINAL — 5-way (Original/Physics/DeSkate+IK/Gaussian+IK/Learned+IK)")
    log("=" * 84)

    ck = torch.load("checkpoints/v18_ik/best.pth", map_location=DEVICE)
    model = FootRefiner().to(DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    log(f"  Learned smoother: {model.n_params():,} params (epoch {ck['epoch']+1}, loss {ck['loss']:.5f})")

    with open(TEST_NAMES) as f:
        names = json.load(f)
    log(f"  Held-out: {len(names)} motions   (σ_gauss={GAUSS_SIGMA})")
    log("")

    hdr = (f"{'#':>2} {'Name':<22} {'FSRo':>5} {'FSRds':>5} {'FSRg':>5} {'FSRL':>5} | "
           f"{'Jo':>6} {'Jds':>6} {'Jg':>6} {'JL':>6} | {'BoneL':>6} {'shin':>5}")
    log(hdr); log("-" * len(hdr))

    R = {'orig': [], 'phys': [], 'ds': [], 'gauss': [], 'learn': []}
    viz = []

    for i, nm in enumerate(names):
        m = np.load(f"{INPUT_DIR}/{nm}.npy").astype(np.float32)
        if m.ndim == 4:
            m = m[0]
        phys, _ = physics_foot_fix(m, damp_factor=0.0, return_stats=True)
        ds = apply_ik(m, deskate_only(m, 0.0))
        gauss = apply_ik(m, deskate_only(m, GAUSS_SIGMA))
        learn = apply_ik(m, smooth_fix(m, model, DEVICE))

        mo, mp, md, mg, ml = M(m, m), M(phys, m), M(ds, m), M(gauss, m), M(learn, m)
        R['orig'].append(mo); R['phys'].append(mp); R['ds'].append(md)
        R['gauss'].append(mg); R['learn'].append(ml)

        np.save(f"{OUTPUT_DIR}/{nm}_v18ik.npy", learn)
        log(f"{i+1:>2} {nm[:21]:<22} {mo['FSR']:>4.1%} {md['FSR']:>4.1%} {mg['FSR']:>4.1%} {ml['FSR']:>4.1%} | "
            f"{mo['Jitter']:>6.4f} {md['Jitter']:>6.4f} {mg['Jitter']:>6.4f} {ml['Jitter']:>6.4f} | "
            f"{ml['BoneCV']:>6.4f} {maxshin(learn):>5.3f}")
        if i < 4:
            viz.append((nm, m, ds, learn))

    log("-" * len(hdr))

    def avg(L): return {k: float(np.mean([x[k] for x in L])) for k in L[0]}
    ao, ap, ad, ag, al = (avg(R['orig']), avg(R['phys']), avg(R['ds']),
                          avg(R['gauss']), avg(R['learn']))

    log("")
    log("=" * 84)
    log(f"  📊 5-WAY ABLATION (n={len(names)})")
    log("=" * 84)
    log(f"  {'Metric':<13} {'Original':>9} {'Physics':>9} {'DeSk+IK':>9} {'Gauss+IK':>9} {'Learn+IK':>9}")
    log(f"  {'─'*13} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")
    for k, lbl, pct in [('FSR','FSR ↓',True),('Jitter','Jitter ↓',False),
                        ('FootErr','FootErr ↓',False),('ContactAcc','ContactAcc ↑',True),
                        ('BoneCV','BoneCV ↓',False),('PenMean','Penetrat ↓',False)]:
        if pct:
            log(f"  {lbl:<13} {ao[k]:>8.1%} {ap[k]:>8.1%} {ad[k]:>8.1%} {ag[k]:>8.1%} {al[k]:>8.1%}")
        else:
            log(f"  {lbl:<13} {ao[k]:>9.4f} {ap[k]:>9.4f} {ad[k]:>9.4f} {ag[k]:>9.4f} {al[k]:>9.4f}")

    log("")
    log("  ── Verdict ──")
    if al['FSR'] < ao['FSR'] and al['Jitter'] < ao['Jitter']:
        log(f"  ✅ Learned+IK beats Original on BOTH: FSR {ao['FSR']:.1%}→{al['FSR']:.1%}, "
            f"Jitter {ao['Jitter']:.4f}→{al['Jitter']:.4f}")
    log(f"  🦴 BoneCV {al['BoneCV']:.4f} vs orig {ao['BoneCV']:.4f} — leg-tear {'FIXED' if al['BoneCV']<0.01 else 'NOT fixed'}")
    log(f"  📏 FootErr {al['FootErr']:.3f}m, ContactAcc {al['ContactAcc']:.1%}")
    # learned vs gaussian
    log(f"  Learned vs Gaussian: FSR {ag['FSR']:.1%}→{al['FSR']:.1%}, Jitter {ag['Jitter']:.4f}→{al['Jitter']:.4f}")
    if al['FSR'] <= ag['FSR'] and al['Jitter'] <= ag['Jitter'] + 1e-4:
        log(f"  ✅ Learned Pareto-dominates fixed Gaussian (adaptive smoothing pays off)")
    else:
        log(f"  ℹ️  Learned trades on the same frontier as Gaussian")
    log(f"  vs SOTA: OmniControl/MaskControl FSR≈5.5% | Learned+IK FSR={al['FSR']:.1%}")

    # ── Plots ──
    log("")
    log("  Generating plots...")
    # shin length (leg-tear fixed)
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('V18+IK — Shin length (knee→ankle) [flat=bone intact]', fontsize=15, fontweight='bold')
    for row, (nm, m, ds, learn) in enumerate(viz):
        for col, leg in enumerate(LEGS):
            ax = axes[row][col]
            for data, c, lbl in [(m,'#bbbbbb','Original'),(learn,'#2196F3','Learned+IK')]:
                sh = np.linalg.norm(data[:, leg['ankle'], :] - data[:, leg['knee'], :], axis=1)
                ax.plot(sh, color=c, lw=1.6, alpha=0.9, label=lbl)
            if row == 0 and col == 0: ax.legend(fontsize=8)
            side = 'L' if leg['ankle'] == 7 else 'R'
            ax.set_title(f'{nm[:18]} {side}-shin', fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(f'{VIZ_DIR}/01_shin_length.png', dpi=140, bbox_inches='tight'); plt.close()
    log(f"  ✅ {VIZ_DIR}/01_shin_length.png")

    # foot speed (jitter / planting)
    fig, axes = plt.subplots(4, 1, figsize=(18, 14))
    nm, m, ds, learn = viz[0]
    fig.suptitle(f'V18+IK Foot speed — {nm[:40]} (spikes=jitter; flat-low=planted)', fontsize=14, fontweight='bold')
    for idx, fj in enumerate(FOOT_JOINTS):
        ax = axes[idx]
        for data, c, lbl in [(m,'#bbbbbb','Original'),(ds,'#F44336','DeSk+IK (raw)'),(learn,'#2196F3','Learned+IK')]:
            v = np.linalg.norm(data[1:, fj, :] - data[:-1, fj, :], axis=1)
            ax.plot(v, color=c, lw=1.3, alpha=0.85, label=lbl)
        ax.axhline(0.03, color='red', ls=':', alpha=0.5)
        nmj = {7:'L-Ankle',8:'R-Ankle',10:'L-Foot',11:'R-Foot'}[fj]
        ax.set_title(f'{nmj} speed', fontsize=10); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(f'{VIZ_DIR}/02_foot_speed.png', dpi=140, bbox_inches='tight'); plt.close()
    log(f"  ✅ {VIZ_DIR}/02_foot_speed.png")

    with open(f'{VIZ_DIR}/summary.json', 'w') as f:
        json.dump({'original': ao, 'physics': ap, 'deskate_ik': ad,
                   'gauss_ik': ag, 'learn_ik': al, 'n': len(names)}, f, indent=2)
    log(f"  ✅ {VIZ_DIR}/summary.json")
    log("=" * 84)
    log(f"  💾 {OUTPUT_DIR}/  |  {VIZ_DIR}/")
    log("  ✅ Done.")


if __name__ == "__main__":
    main()
