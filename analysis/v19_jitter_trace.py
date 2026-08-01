"""
What does Jitter 0.0173 actually LOOK like?
===========================================
`v19_045` delivers FSR 7.34% on T2M-GPT but Jitter 0.01733 — ABOVE the original
(0.01388). The scalar does not say whether that reads as twitching to a viewer.

This script does the part that can be judged from numbers: it plots ankle speed
and acceleration over time for each method, and counts acceleration spikes.
Twitching is a SPIKE phenomenon — a few large accelerations — not a raised mean,
so a method can have higher RMS jitter and still look smooth if the extra energy
is spread rather than spiky (and vice versa).

The visual call itself needs eyes on `utils/render_v19.py` output.

Outputs: analysis/v19/jitter_trace.png, analysis/v19/jitter_stats.json
"""
import os, sys, json, glob
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import FootRefiner, smooth_fix, deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SRC = "data/test_inputs/t2mgpt/t2mgpt_raw_joints"
ANA = "analysis/v19"
ANKLES = [7, 8]
# Categorical slots from the reference palette, fixed order (never cycled).
COLORS = {'original': '#52514e', 'deskate_ik': '#eb6834',
          'gauss_ik': '#008300', 'v19_ik': '#2a78d6'}
C_INK, C_MUTED = '#0b0b0b', '#52514e'
os.makedirs(ANA, exist_ok=True)


def deskate_only(m, sigma=0.0):
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    if sigma > 0:
        tgt = gaussian_filter1d(tgt, sigma=sigma, axis=0, mode='nearest')
    out = flat.copy(); out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def build(mo, m19):
    return {
        'original': mo,
        'deskate_ik': apply_ik(mo, deskate_only(mo, 0.0)),
        'gauss_ik': apply_ik(mo, deskate_only(mo, 1.5)),
        'v19_ik': apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE)),
    }


def traces(m):
    """Ankle horizontal speed and full 3D acceleration magnitude, per frame."""
    a = m[:, ANKLES, :]
    v = a[1:] - a[:-1]
    speed = np.linalg.norm(v[..., [0, 2]], axis=-1)      # (T-1, 2)
    acc = np.linalg.norm(v[1:] - v[:-1], axis=-1)        # (T-2, 2)
    return speed, acc


def main():
    m19 = V19Smoother().to(DEVICE)
    m19.load_state_dict(torch.load("checkpoints/v19_045/best.pth",
                                   map_location=DEVICE)['model_state_dict'])
    m19.eval()

    files = sorted(glob.glob(f"{SRC}/*.npy"))
    stats = {k: {'acc_rms': [], 'acc_p99': [], 'acc_max': [], 'spikes': []}
             for k in COLORS}
    SPIKE = 0.01          # accel magnitude counted as a visible twitch (m/frame^2)

    for fp in files:
        mo = np.load(fp).astype(np.float32)
        if mo.ndim == 4:
            mo = mo[0]
        for k, mm in build(mo, m19).items():
            _, acc = traces(mm)
            f = acc.ravel()
            stats[k]['acc_rms'].append(float(np.sqrt((f ** 2).mean())))
            stats[k]['acc_p99'].append(float(np.percentile(f, 99)))
            stats[k]['acc_max'].append(float(f.max()))
            stats[k]['spikes'].append(float((f > SPIKE).mean()))

    summary = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in stats.items()}
    with open(f"{ANA}/jitter_stats.json", 'w') as f:
        json.dump({'spike_threshold': SPIKE, 'n': len(files), 'summary': summary}, f, indent=2)

    print(f"Ankle acceleration character over n={len(files)} T2M-GPT motions")
    print(f"{'method':<12} {'RMS':>9} {'p99':>9} {'max':>9} {'frames>':>9}")
    print(f"{'':12} {'':>9} {'':>9} {'':>9} {SPIKE:>9}")
    for k in COLORS:
        s = summary[k]
        print(f"{k:<12} {s['acc_rms']:9.5f} {s['acc_p99']:9.5f} "
              f"{s['acc_max']:9.5f} {s['spikes']*100:8.2f}%")

    plot(files, m19)
    print(f"\n-> {ANA}/jitter_trace.png  |  {ANA}/jitter_stats.json")


def plot(files, m19):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    picks = [f for f in files if any(t in f for t in ('001120', '000818', '012498'))][:3]
    if len(picks) < 3:
        picks = files[:3]

    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.7 * len(picks)),
                             facecolor='#fcfcfb', sharex=False)
    if len(picks) == 1:
        axes = [axes]

    for ax, fp in zip(axes, picks):
        mo = np.load(fp).astype(np.float32)
        if mo.ndim == 4:
            mo = mo[0]
        ax.set_facecolor('#fcfcfb')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#d9d8d3')
        ax.grid(True, color='#ebeae5', linewidth=0.8)
        ax.set_axisbelow(True)

        for k, mm in build(mo, m19).items():
            _, acc = traces(mm)
            ax.plot(acc[:, 0], color=COLORS[k], linewidth=1.6, label=k, alpha=0.9)

        name = os.path.basename(fp).replace('t2mgpt_', '').replace('_joints.npy', '')
        ax.set_title(f"{name} — left-ankle acceleration magnitude",
                     fontsize=10.5, color=C_INK, loc='left', pad=8)
        ax.set_ylabel('|accel|', fontsize=9.5, color=C_MUTED)
        ax.tick_params(colors=C_MUTED, labelsize=9)
    axes[-1].set_xlabel('frame', fontsize=9.5, color=C_MUTED)

    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9.5, labelcolor=C_MUTED, ncol=4,
               loc='upper left', bbox_to_anchor=(0.006, 0.968))
    fig.suptitle('Twitching is spikes, not a raised mean — where does each method '
                 'put its acceleration?',
                 fontsize=12.5, color=C_INK, x=0.006, ha='left', y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{ANA}/jitter_trace.png", dpi=160, facecolor='#fcfcfb')


if __name__ == "__main__":
    main()
