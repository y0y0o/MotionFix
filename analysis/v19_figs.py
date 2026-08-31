"""
Thesis Figures — qualitative ankle-trajectory (Fig T) and speed (Fig S).
========================================================================
Addresses figure-review points: serif fonts, semantic labels (Ours (full)),
reach-clamp as a bottom rug (not full-height lines), light contact shading,
zoom inset on one contact window, legend off the data, split-y speed panel with
a labelled threshold, no in-figure title (caption carries the conclusion).

Auto-selects a representative T2M-GPT clip. Metric defs match utils.metrics.
Output: analysis/v19/fig_trajectory.png, analysis/v19/fig_speed.png
"""
import os, sys, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8.5,
})

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from utils.metrics import compute_contact_labels

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SRC = "data/test_inputs/t2mgpt/t2mgpt_raw_joints"
ANA = "analysis/v19"
ANK = 7
COL = {'orig': '#52514e', 'deskate': '#eb6834', 'ours': '#2a78d6'}
CLAMP, SHADE, C_MUT, GRIDC = '#c026d3', '#2a78d6', '#6b6b6b', '#ebeae5'


def load(fp):
    m = np.load(fp).astype(np.float32)
    return m[0] if m.ndim == 4 else m


def deskate_only(m):
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    out = flat.copy(); out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def windows(lab):
    w, t, T = [], 0, len(lab)
    while t < T:
        if lab[t] > 0.5:
            s = t
            while t < T and lab[t] > 0.5:
                t += 1
            w.append((s, t - 1))
        else:
            t += 1
    return w


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d9d8d3')
    ax.grid(True, color=GRIDC, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_MUT)


def shade(ax, wins):
    for s, e in wins:
        ax.axvspan(s, e, color=SHADE, alpha=0.10, lw=0)


def rug(ax, cf):
    ylo, yhi = ax.get_ylim()
    h = (yhi - ylo) * 0.045
    ax.vlines(cf, ylo, ylo + h, color=CLAMP, lw=1.0, alpha=0.8)
    ax.set_ylim(ylo, yhi)


def pick(m19, files):
    best = None
    for fp in files[:120]:
        mo = load(fp)
        dk = deskate_only(mo)
        v = apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE))
        lab = compute_contact_labels(mo, (7, 8))[:, 0]
        wins = [w for w in windows(lab) if w[1] - w[0] >= 8]
        if len(wins) < 3:
            continue
        clamp = (np.linalg.norm(dk[:, ANK][:, [0, 2]] - v[:, ANK][:, [0, 2]], axis=-1) > 1e-3) & (lab > 0.5)
        sp = np.r_[0, np.linalg.norm(np.diff(mo[:, ANK][:, [0, 2]], axis=0), axis=-1)]
        score = clamp.sum() + 0.5 * (lab.astype(bool) & (sp > 0.03)).sum()
        if best is None or score > best[0]:
            best = (score, fp, mo, dk, v, lab, wins, np.where(clamp)[0])
    return best


def main():
    m19 = V19Smoother().to(DEVICE)
    m19.load_state_dict(torch.load("checkpoints/v19_088a10/best.pth",
                                   map_location=DEVICE)['model_state_dict'])
    m19.eval()
    files = sorted(glob.glob(f"{SRC}/*.npy"))
    _, fp, mo, dk, v, lab, wins, cf = pick(m19, files)
    name = os.path.basename(fp).replace('t2mgpt_', '').replace('_joints.npy', '')
    T = mo.shape[0]; fr = np.arange(T)
    # zoom window: the middle contact window, padded
    wz = wins[len(wins) // 2]
    z0, z1 = max(0, wz[0] - 15), min(T, wz[1] + 16)
    print(f"clip {name} T={T} windows={len(wins)} clamp={len(cf)} zoom={z0}-{z1}")

    # ── Figure T: trajectory, full-length (left) + zoom (right), X row / Z row ──
    fig = plt.figure(figsize=(9.2, 4.8), facecolor='white')
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], hspace=0.28, wspace=0.22)
    for r, (coord, ci) in enumerate([('X', 0), ('Z', 2)]):
        for c, (lo, hi, zoom) in enumerate([(0, T, False), (z0, z1, True)]):
            ax = fig.add_subplot(gs[r, c]); ax.set_facecolor('white'); style(ax)
            shade(ax, wins)
            ax.plot(fr, mo[:, ANK, ci], color=COL['orig'], lw=1.4, label='original')
            ax.plot(fr, dk[:, ANK, ci], color=COL['deskate'], lw=1.4, label='de-skate')
            ax.plot(fr, v[:, ANK, ci], color=COL['ours'], lw=1.7, label='Ours (full)')
            ax.set_xlim(lo, hi)
            rug(ax, cf[(cf >= lo) & (cf < hi)])
            if c == 0:
                ax.set_ylabel(f'left ankle {coord} (m)')
            if zoom:
                ax.set_title('zoom: one contact window', color=C_MUT, fontsize=8)
            if r == 1:
                ax.set_xlabel('frame')
    handles = [plt.Line2D([], [], color=COL['orig'], lw=1.6, label='original'),
               plt.Line2D([], [], color=COL['deskate'], lw=1.6, label='de-skate'),
               plt.Line2D([], [], color=COL['ours'], lw=1.8, label='Ours (full)'),
               plt.Line2D([], [], color=CLAMP, lw=1.4, label='reach-clamp (rug)'),
               matplotlib.patches.Patch(color=SHADE, alpha=0.10, label='contact window')]
    fig.legend(handles=handles, frameon=False, ncol=5, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), labelcolor=C_MUT)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{ANA}/fig_trajectory.png", dpi=170, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    # ── Figure S: horizontal speed, split-y (spikes on top, decision band below) ──
    def hspeed(x):
        return np.r_[0, np.linalg.norm(np.diff(x[:, ANK][:, [0, 2]], axis=0), axis=-1)]
    so, sd, sv = hspeed(mo), hspeed(apply_ik(mo, dk)), hspeed(v)
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(9.2, 4.4), facecolor='white',
                                   sharex=True, gridspec_kw={'height_ratios': [1, 1.4], 'hspace': 0.12})
    for ax, (lo, hi) in [(axT, (0.10, 0.52)), (axB, (0.0, 0.10))]:
        ax.set_facecolor('white'); style(ax)
        shade(ax, wins)
        ax.plot(fr, so, color=COL['orig'], lw=1.3, label='original')
        ax.plot(fr, sd, color=COL['deskate'], lw=1.3, label='de-skate+IK')
        ax.plot(fr, sv, color=COL['ours'], lw=1.7, label='Ours (full)')
        ax.set_ylim(lo, hi)
    axB.axhline(0.03, color='#1a1a1a', lw=1.0, ls='--', alpha=0.8)
    axB.text(T * 0.995, 0.033, r'$\tau = 0.03$ (FSR threshold)', ha='right', fontsize=8, color='#1a1a1a')
    axT.spines['bottom'].set_visible(False); axT.tick_params(bottom=False)
    axB.set_ylabel('left-ankle horizontal speed (m/frame)')
    axB.yaxis.set_label_coords(-0.07, 1.0)
    axB.set_xlabel('frame')
    rug(axB, cf)
    handles = [plt.Line2D([], [], color=COL['orig'], lw=1.5, label='original'),
               plt.Line2D([], [], color=COL['deskate'], lw=1.5, label='de-skate+IK'),
               plt.Line2D([], [], color=COL['ours'], lw=1.8, label='Ours (full)'),
               matplotlib.patches.Patch(color=SHADE, alpha=0.10, label='contact window'),
               plt.Line2D([], [], color=CLAMP, lw=1.4, label='reach-clamp (rug)')]
    fig.legend(handles=handles, frameon=False, ncol=5, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), labelcolor=C_MUT)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{ANA}/fig_speed.png", dpi=170, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"-> {ANA}/fig_trajectory.png\n-> {ANA}/fig_speed.png")


if __name__ == "__main__":
    main()
