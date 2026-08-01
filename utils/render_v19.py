"""
V19 — side-by-side comparison videos for the visual call
========================================================
`analysis/v19_jitter_trace.py` shows that v19_045's extra jitter is SPIKY
(p99 1.54x the original, max 1.79x) rather than a uniformly raised floor. That
predicts it reads as twitching — but the prediction needs eyes to confirm.

Renders Original | De-skate+IK | Gaussian+IK | V19(js=.45) | V19(js=.88) as one
5-panel video so the same frame can be compared across methods AND across the two
delivery candidates. Shins are drawn thick (leg-tear indicator) and foot joints
highlighted (the joints every metric is about).

Default motion list = the 6 WORST cases by v19_045 spike ratio (p99 vs original)
plus one clean control. Picking worst-case is deliberate: a mid-range clip looking
fine says nothing about whether the method twitches.

Usage:  python utils/render_v19.py [motion_id ...]
Output: outputs/videos/v19/<id>.mp4
"""
import os, sys, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19

BONES = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]
SHIN = {(4, 7), (5, 8)}
FEET = {7, 8, 10, 11}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SRC = "data/test_inputs/t2mgpt/t2mgpt_raw_joints"
OUT = "outputs/videos/v19"
C_BONE, C_SHIN, C_FOOT = '#52514e', '#2a78d6', '#eb6834'
os.makedirs(OUT, exist_ok=True)


def deskate_only(m, sigma=0.0):
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    if sigma > 0:
        tgt = gaussian_filter1d(tgt, sigma=sigma, axis=0, mode='nearest')
    out = flat.copy(); out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def render(mid, model):
    hits = glob.glob(f"{SRC}/*{mid}*.npy")
    if not hits:
        print(f"  !! {mid} not found"); return
    mo = np.load(hits[0]).astype(np.float32)
    if mo.ndim == 4:
        mo = mo[0]

    variants = [
        ('Original', mo),
        ('De-skate + IK', apply_ik(mo, deskate_only(mo, 0.0))),
        ('Gaussian + IK', apply_ik(mo, deskate_only(mo, 1.5))),
        ('V19 js=.45 (low FSR)', apply_ik(mo, smooth_fix_v19(mo, model['045'], DEVICE))),
        ('V19 js=.88 (low spike)', apply_ik(mo, smooth_fix_v19(mo, model['088a10'], DEVICE))),
    ]
    T = mo.shape[0]
    allp = np.concatenate([v for _, v in variants], 0)
    lo, hi = allp.reshape(-1, 3).min(0), allp.reshape(-1, 3).max(0)
    pad = 0.25

    fig, axes = plt.subplots(1, 5, figsize=(21, 5), facecolor='#fcfcfb',
                             subplot_kw={'projection': '3d'})
    lines, pts = [], []
    for ax, (name, _) in zip(axes, variants):
        ax.set_title(name, fontsize=11, color='#0b0b0b', pad=4)
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[2] - pad, hi[2] + pad)
        ax.set_zlim(0, hi[1] + pad)
        ax.set_box_aspect((1, 1, 1.1))
        ax.view_init(elev=12, azim=-70)
        ax.set_facecolor('#fcfcfb')
        ax.grid(False)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_ticklabels([])
            a.pane.set_alpha(0.03)
        ls = [ax.plot([], [], [], color=C_SHIN if b in SHIN else C_BONE,
                      linewidth=3.2 if b in SHIN else 1.7)[0] for b in BONES]
        p = ax.plot([], [], [], 'o', color=C_FOOT, markersize=5)[0]
        lines.append(ls); pts.append(p)

    def upd(t):
        for ls, p, (_, m) in zip(lines, pts, variants):
            j = m[min(t, len(m) - 1)]
            for ln, (a, b) in zip(ls, BONES):
                ln.set_data([j[a, 0], j[b, 0]], [j[a, 2], j[b, 2]])
                ln.set_3d_properties([j[a, 1], j[b, 1]])
            f = j[sorted(FEET)]
            p.set_data(f[:, 0], f[:, 2]); p.set_3d_properties(f[:, 1])
        return [x for ls in lines for x in ls] + pts

    fig.suptitle(f"{mid} — thick blue = shin (leg-tear), orange = foot joints",
                 fontsize=12, color='#0b0b0b', x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ani = FuncAnimation(fig, upd, frames=T, interval=50, blit=False)
    path = f"{OUT}/{mid}.mp4"
    ani.save(path, writer=FFMpegWriter(fps=20, bitrate=2400))
    plt.close(fig)
    print(f"  -> {path}  ({T} frames)")


# worst 6 by v19_045 ankle-accel p99 ratio vs original (3.36x .. 2.20x), plus a
# clean control (011028, 0.84x) so the comparison is not selected only on failures
WORST = ['003111', '009377', '002530', '001168', '003784', '004311']
CONTROL = ['011028']


def main():
    ids = sys.argv[1:] or (WORST + CONTROL)
    model = {}
    for tag in ('045', '088a10'):
        m = V19Smoother().to(DEVICE)
        m.load_state_dict(torch.load(f"checkpoints/v19_{tag}/best.pth",
                                     map_location=DEVICE)['model_state_dict'])
        m.eval()
        model[tag] = m
    for mid in ids:
        render(mid, model)


if __name__ == "__main__":
    main()
