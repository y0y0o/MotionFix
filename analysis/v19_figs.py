"""
Thesis Figures 2 & 3 — qualitative ankle-trajectory and velocity plots.
=======================================================================
Fig 2: one ankle's X and Z position over time, overlaying original / de-skate /
       full (V19), with contact windows shaded and reach-clamp frames marked.
Fig 3: the same ankle's horizontal speed over time, showing de-skate introduces
       boundary jitter that the smoother repairs.

Auto-selects a T2M-GPT clip with a clear contact window, visible original skating,
and at least one reach-clamp event. Metric defs match utils.metrics.
Output: analysis/v19/fig2_ankle_trajectory.png, analysis/v19/fig3_ankle_speed.png
"""
import os, sys, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from utils.metrics import compute_contact_labels

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SRC = "data/test_inputs/t2mgpt/t2mgpt_raw_joints"
ANA = "analysis/v19"
ANK = 7                      # left ankle
COL = {'original': '#52514e', 'deskate': '#eb6834', 'v19': '#2a78d6'}
C_INK, C_MUT, GRID = '#1a1a1a', '#6b6b6b', '#ebeae5'


def load(fp):
    m = np.load(fp).astype(np.float32)
    return m[0] if m.ndim == 4 else m


def deskate_only(m):
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    out = flat.copy()
    out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def contact_windows(labels_col):
    """List of (start, end) inclusive index ranges where labels==1."""
    wins, t, T = [], 0, len(labels_col)
    while t < T:
        if labels_col[t] > 0.5:
            s = t
            while t < T and labels_col[t] > 0.5:
                t += 1
            wins.append((s, t - 1))
        else:
            t += 1
    return wins


def pick_clip(m19, files):
    """Choose a clip with a decent contact window, original skating, a clamp event."""
    best = None
    for fp in files[:120]:
        mo = load(fp)
        dk = deskate_only(mo)
        v19 = apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE))
        lab = compute_contact_labels(mo, (7, 8))[:, 0]
        wins = [w for w in contact_windows(lab) if w[1] - w[0] >= 8]
        if not wins:
            continue
        # reach-clamp frames on left ankle within contact
        clamp = (np.linalg.norm(dk[:, ANK][:, [0, 2]] - v19[:, ANK][:, [0, 2]], axis=-1) > 1e-3) & (lab > 0.5)
        # original horizontal speed inside contact (skating amount)
        sp = np.linalg.norm(np.diff(mo[:, ANK][:, [0, 2]], axis=0), axis=-1)
        skate = (sp[:-0 or None] > 0.03)
        score = clamp.sum() + 0.5 * (lab.astype(bool) & np.r_[skate, False]).sum() + 0.01 * sum(w[1]-w[0] for w in wins)
        if best is None or score > best[0]:
            best = (score, fp, mo, dk, v19, lab, wins, clamp)
    return best


def shade(ax, wins, clamp):
    for s, e in wins:
        ax.axvspan(s, e, color='#2a78d6', alpha=0.06, lw=0)
    cf = np.where(clamp)[0]
    for i, f in enumerate(cf):
        ax.axvline(f, color='#c026d3', alpha=0.22, lw=0.6,
                   label='reach-clamp frame' if i == 0 else None)


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d9d8d3')
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_MUT, labelsize=9)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    m19 = V19Smoother().to(DEVICE)
    m19.load_state_dict(torch.load("checkpoints/v19_088a10/best.pth",
                                   map_location=DEVICE)['model_state_dict'])
    m19.eval()
    files = sorted(glob.glob(f"{SRC}/*.npy"))
    _, fp, mo, dk, v19, lab, wins, clamp = pick_clip(m19, files)
    name = os.path.basename(fp).replace('t2mgpt_', '').replace('_joints.npy', '')
    T = mo.shape[0]
    fr = np.arange(T)
    print(f"selected clip {name}  T={T}  contact windows={wins}  clamp frames={int(clamp.sum())}")

    # ── Figure 2: ankle X and Z position over time ──
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), facecolor='#fcfcfb', sharex=True)
    for ax, coord, ci in [(axes[0], 'X', 0), (axes[1], 'Z', 2)]:
        ax.set_facecolor('#fcfcfb'); style(ax)
        shade(ax, wins, clamp)
        ax.plot(fr, mo[:, ANK, ci], color=COL['original'], lw=1.6, label='original', alpha=0.9)
        ax.plot(fr, dk[:, ANK, ci], color=COL['deskate'], lw=1.6, label='de-skate', alpha=0.9)
        ax.plot(fr, v19[:, ANK, ci], color=COL['v19'], lw=1.8, label='full (V19)', alpha=0.95)
        ax.set_ylabel(f'left-ankle {coord} (m)', fontsize=10, color=C_MUT)
    axes[1].set_xlabel('frame', fontsize=10, color=C_MUT)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9.5, labelcolor=C_MUT, ncol=4,
               loc='upper left', bbox_to_anchor=(0.008, 0.925))
    fig.suptitle(f'Left-ankle trajectory (clip {name}) — shaded: contact window,  '
                 'magenta: reach-clamp',
                 fontsize=10.5, color=C_INK, x=0.008, ha='left', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(f"{ANA}/fig2_ankle_trajectory.png", dpi=170, facecolor='#fcfcfb')
    plt.close(fig)

    # ── Figure 3: ankle horizontal speed over time ──
    def hspeed(x):
        return np.r_[0, np.linalg.norm(np.diff(x[:, ANK][:, [0, 2]], axis=0), axis=-1)]
    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb'); style(ax)
    shade(ax, wins, clamp)
    ax.axhline(0.03, color='#b8860b', lw=1.0, ls='--', alpha=0.8, label='FSR threshold (0.03)')
    ax.plot(fr, hspeed(mo), color=COL['original'], lw=1.5, label='original', alpha=0.9)
    ax.plot(fr, hspeed(apply_ik(mo, dk)), color=COL['deskate'], lw=1.5, label='de-skate+IK', alpha=0.9)
    ax.plot(fr, hspeed(v19), color=COL['v19'], lw=1.8, label='full (V19)', alpha=0.95)
    ax.set_ylabel('left-ankle horizontal speed (m/frame)', fontsize=10, color=C_MUT)
    ax.set_xlabel('frame', fontsize=10, color=C_MUT)
    ax.legend(frameon=False, fontsize=9, labelcolor=C_MUT, ncol=2)
    fig.suptitle(f'Left-ankle horizontal speed (clip {name}): de-skate spikes at contact boundaries, '
                 'smoother flattens them',
                 fontsize=10, color=C_INK, x=0.008, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{ANA}/fig3_ankle_speed.png", dpi=170, facecolor='#fcfcfb')
    plt.close(fig)
    print(f"-> {ANA}/fig2_ankle_trajectory.png\n-> {ANA}/fig3_ankle_speed.png")


if __name__ == "__main__":
    main()
