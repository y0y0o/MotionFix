"""
V16 — Test on 10 Held-Out Motions + Visual Verification
========================================================
Three-way comparison: V16 model vs Physics teacher vs MoMask original.
Full 7-metric panel + trajectory plots to verify NO twitching.

The user's concern: did V16 game FSR/Jitter at the cost of real quality?
Guards checked here: Floating, BoneCV, Penetration (catch gaming) + visual plots.
"""
import torch, numpy as np, pickle, os, sys, json, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v16 import MotionFixNetworkV16
from utils.physics_fix import physics_foot_fix
from utils.metrics import (
    compute_fsr, compute_jitter, compute_floating, compute_foot_error,
    compute_bone_length_consistency, compute_ground_penetration,
    compute_contact_accuracy,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
TEST_NAMES = "data/training/v15/test_names.json"
OUTPUT_DIR = "outputs/fixed/v16"
VIZ_DIR = "analysis/v16_viz"
LOG_PATH = "logs/v16_test.log"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

JOINT_NAMES = {7: "L-Ankle", 8: "R-Ankle", 10: "L-Foot", 11: "R-Foot"}
FOOT_JOINTS = [7, 8, 10, 11]


def log(msg, also_print=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line)


def infer_v16(model, motion_world):
    T = motion_world.shape[0]
    flat = motion_world.reshape(T, -1).astype(np.float32)
    tensor = torch.from_numpy(flat).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        # foot_only=True: selective foot replacement at skating frames
        result = model(tensor, foot_only=True, root_relative=True)
    return result.squeeze(0).cpu().numpy().reshape(T, 22, 3)


def all_metrics(motion, original):
    return {
        'FSR': compute_fsr(motion)[0],
        'Jitter': compute_jitter(motion),
        'Floating': compute_floating(motion)[0],
        'FootErr': compute_foot_error(motion, original),
        'ContactAcc': compute_contact_accuracy(motion, original),
        'BoneCV': compute_bone_length_consistency(motion),
        'PenMean': compute_ground_penetration(motion)[0],
        'PenMax': compute_ground_penetration(motion)[1],
    }


def main():
    log("=" * 72)
    log("  V16 — Test on Held-Out (3-way: V16 / Physics / Original)")
    log("=" * 72)

    ckpt = torch.load("checkpoints/v16/best.pth", map_location=DEVICE)
    model = MotionFixNetworkV16(blend_alpha=0.5).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    log(f"  Model: V16 (epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f})")

    with open(TEST_NAMES) as f:
        test_names = json.load(f)
    log(f"  Held-out motions: {len(test_names)}")
    log("")

    hdr = (f"{'#':>2} {'Name':<32} {'Orig':>6} {'Phys':>6} {'V16':>6} "
           f"{'Jit_o':>6} {'Jit_p':>6} {'Jit_v':>6} {'Float':>6} {'BoneCV':>7}")
    log(hdr); log("-" * len(hdr))

    R = {'orig': [], 'phys': [], 'v16': []}
    viz_motions = []

    for i, name in enumerate(test_names):
        fp = f"{INPUT_DIR}/{name}.npy"
        orig = np.load(fp).astype(np.float32)
        if orig.ndim == 4:
            orig = orig[0]

        phys, _ = physics_foot_fix(orig, damp_factor=0.0, return_stats=True)
        v16 = infer_v16(model, orig)

        m_o = all_metrics(orig, orig)
        m_p = all_metrics(phys, orig)
        m_v = all_metrics(v16, orig)

        np.save(f"{OUTPUT_DIR}/{name}_v16.npy", v16)

        log(f"{i+1:>2} {name[:31]:<32} "
            f"{m_o['FSR']:>5.1%} {m_p['FSR']:>5.1%} {m_v['FSR']:>5.1%} "
            f"{m_o['Jitter']:>6.4f} {m_p['Jitter']:>6.4f} {m_v['Jitter']:>6.4f} "
            f"{m_v['Floating']:>5.1%} {m_v['BoneCV']:>7.4f}")

        R['orig'].append(m_o); R['phys'].append(m_p); R['v16'].append(m_v)
        if i < 4:
            viz_motions.append((name, orig, phys, v16, m_o, m_p, m_v))

    log("-" * len(hdr))

    def avg(lst): return {k: float(np.mean([m[k] for m in lst])) for k in lst[0]}
    ao, ap, av = avg(R['orig']), avg(R['phys']), avg(R['v16'])

    log("")
    log("=" * 72)
    log(f"  📊 Summary (n={len(test_names)})")
    log("=" * 72)
    log(f"  {'Metric':<16} {'Original':>10} {'Physics':>10} {'V16':>10}")
    log(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*10}")
    for k, lbl, pct in [('FSR','FSR',True), ('Jitter','Jitter',False),
                        ('Floating','Floating',True), ('FootErr','FootErr',False),
                        ('ContactAcc','ContactAcc',True), ('BoneCV','BoneCV',False),
                        ('PenMean','Penetration',False)]:
        if pct:
            log(f"  {lbl:<16} {ao[k]:>9.1%} {ap[k]:>9.1%} {av[k]:>9.1%}")
        else:
            log(f"  {lbl:<16} {ao[k]:>10.4f} {ap[k]:>10.4f} {av[k]:>10.4f}")

    log("")
    log("  ── Verdict (anti-gaming checks) ──")
    # 1. FSR improved?
    d_fsr = av['FSR'] - ao['FSR']
    if d_fsr < -0.005:
        log(f"  ✅ FSR reduced: {ao['FSR']:.1%} → {av['FSR']:.1%} ({d_fsr:+.1%})")
    elif d_fsr < 0.005:
        log(f"  ➖ FSR unchanged: {ao['FSR']:.1%} → {av['FSR']:.1%}")
    else:
        log(f"  ❌ FSR worsened: {ao['FSR']:.1%} → {av['FSR']:.1%}")
    # 2. Jitter vs physics (the twitch test)
    if av['Jitter'] < ap['Jitter']:
        log(f"  ✅ Jitter {av['Jitter']:.4f} < Physics {ap['Jitter']:.4f} — SMOOTHER than physics (no twitch)")
    else:
        log(f"  ⚠️  Jitter {av['Jitter']:.4f} ≥ Physics {ap['Jitter']:.4f}")
    # 3. Floating gaming check
    if av['Floating'] < 0.02:
        log(f"  ✅ Floating {av['Floating']:.1%} — not lifting feet to game FSR")
    else:
        log(f"  ❌ Floating {av['Floating']:.1%} — GAMING by lifting feet!")
    # 4. Bone gaming check
    if av['BoneCV'] < ap['BoneCV']:
        log(f"  ✅ BoneCV {av['BoneCV']:.4f} < Physics {ap['BoneCV']:.4f} — skeleton more stable")
    else:
        log(f"  ⚠️  BoneCV {av['BoneCV']:.4f} ≥ Physics {ap['BoneCV']:.4f}")

    # ── Visualization ──
    log("")
    log("  Generating trajectory plots (visual twitch check)...")
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle('V16 vs Physics vs Original — Foot X Trajectories (twitch check)',
                 fontsize=15, fontweight='bold')
    for row, (name, orig, phys, v16, mo, mp, mv) in enumerate(viz_motions):
        for col, fj in enumerate(FOOT_JOINTS):
            ax = axes[row][col]
            ax.plot(orig[:, fj, 0], color='#bbbbbb', lw=1.2, label='Original')
            ax.plot(phys[:, fj, 0], color='#FF9800', lw=1.0, ls='--', alpha=0.8, label='Physics')
            ax.plot(v16[:, fj, 0], color='#2196F3', lw=1.6, alpha=0.9, label='V16')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            ax.set_title(f'{name[:22]} {JOINT_NAMES[fj]} X', fontsize=8)
            ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'{VIZ_DIR}/01_trajectories.png', dpi=140, bbox_inches='tight')
    plt.close()
    log(f"  ✅ Saved {VIZ_DIR}/01_trajectories.png")

    # Velocity plot (twitch = velocity spikes)
    fig, axes = plt.subplots(4, 1, figsize=(18, 14))
    fig.suptitle('V16 Foot Velocity — spikes = twitching', fontsize=14, fontweight='bold')
    name, orig, phys, v16, mo, mp, mv = viz_motions[0]
    for idx, fj in enumerate(FOOT_JOINTS):
        ax = axes[idx]
        for data, color, lbl in [(orig,'#bbbbbb','Original'), (phys,'#FF9800','Physics'), (v16,'#2196F3','V16')]:
            vel = np.linalg.norm(data[1:, fj, :] - data[:-1, fj, :], axis=1)
            ax.plot(vel, color=color, lw=1.3, alpha=0.85, label=lbl)
        ax.axhline(y=0.03, color='red', ls=':', alpha=0.5, label='skate thresh')
        ax.set_title(f'{JOINT_NAMES[fj]} speed', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'{VIZ_DIR}/02_velocity.png', dpi=140, bbox_inches='tight')
    plt.close()
    log(f"  ✅ Saved {VIZ_DIR}/02_velocity.png")

    # Save summary json
    with open(f'{VIZ_DIR}/summary.json', 'w') as f:
        json.dump({'original': ao, 'physics': ap, 'v16': av}, f, indent=2)

    log("=" * 72)
    log(f"  💾 Outputs: {OUTPUT_DIR}/  |  Plots: {VIZ_DIR}/")
    log("  ✅ Done.")


if __name__ == "__main__":
    main()
