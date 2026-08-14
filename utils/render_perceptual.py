"""
Perceptual study stimuli — single-condition, unlabelled, identical-camera clips.
========================================================================
For each motion, renders 4 SEPARATE clips (no titles, shared axis limits and
camera) so they can be shown blind & pairwise:
    orig    = raw generator output
    deskate = de-skate + IK          (physics only)
    gauss   = de-skate + Gaussian + IK
    v19     = de-skate + V19(088a10) + IK   (delivery point)

Neutral skeleton (uniform colour, subtle foot dots) so nothing cues the rater
toward the feet. Output: outputs/perceptual/clips/<mid>_<cond>.mp4

Usage: python utils/render_perceptual.py            # ids from perceptual_ids.txt
       python utils/render_perceptual.py 003111 ...  # explicit ids
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
FEET = [7, 8, 10, 11]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SRC = "data/test_inputs/t2mgpt/t2mgpt_raw_joints"
OUT = "outputs/perceptual/clips"
IDS_FILE = "data/test_inputs/_expand/perceptual_ids.txt"
C_BONE, C_FOOT = '#3a3a3a', '#4a4a4a'
os.makedirs(OUT, exist_ok=True)


def deskate_only(m, sigma=0.0):
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    if sigma > 0:
        tgt = gaussian_filter1d(tgt, sigma=sigma, axis=0, mode='nearest')
    out = flat.copy(); out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def render_clip(motion, path, lo, hi):
    T = motion.shape[0]
    pad = 0.2
    fig = plt.figure(figsize=(4.2, 4.6), facecolor='#ffffff')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[2] - pad, hi[2] + pad)
    ax.set_zlim(0, hi[1] + pad)
    ax.set_box_aspect((1, 1, 1.15))
    ax.view_init(elev=10, azim=-70)
    ax.set_facecolor('#ffffff'); ax.grid(False)
    ax.set_axis_off()
    ls = [ax.plot([], [], [], color=C_BONE, linewidth=2.0)[0] for _ in BONES]
    p = ax.plot([], [], [], 'o', color=C_FOOT, markersize=4)[0]

    def upd(t):
        j = motion[min(t, T - 1)]
        for ln, (a, b) in zip(ls, BONES):
            ln.set_data([j[a, 0], j[b, 0]], [j[a, 2], j[b, 2]])
            ln.set_3d_properties([j[a, 1], j[b, 1]])
        f = j[FEET]
        p.set_data(f[:, 0], f[:, 2]); p.set_3d_properties(f[:, 1])
        return ls + [p]

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ani = FuncAnimation(fig, upd, frames=T, interval=50, blit=False)
    ani.save(path, writer=FFMpegWriter(fps=20, bitrate=2000))
    plt.close(fig)


def render_motion(mid, model):
    hits = glob.glob(f"{SRC}/*{mid}*.npy")
    if not hits:
        print(f"  !! {mid} not found"); return
    mo = np.load(hits[0]).astype(np.float32)
    if mo.ndim == 4:
        mo = mo[0]
    conds = {
        'orig':    mo,
        'deskate': apply_ik(mo, deskate_only(mo, 0.0)),
        'gauss':   apply_ik(mo, deskate_only(mo, 1.5)),
        'v19':     apply_ik(mo, smooth_fix_v19(mo, model, DEVICE)),
    }
    allp = np.concatenate(list(conds.values()), 0).reshape(-1, 3)
    lo, hi = allp.min(0), allp.max(0)
    for cond, m in conds.items():
        path = f"{OUT}/{mid}_{cond}.mp4"
        render_clip(m, path, lo, hi)
    print(f"  {mid}: 4 clips  (T={mo.shape[0]})")


def main():
    ids = sys.argv[1:] or [l.strip() for l in open(IDS_FILE) if l.strip()]
    model = V19Smoother().to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/v19_088a10/best.pth",
                                     map_location=DEVICE)['model_state_dict'])
    model.eval()
    print(f"rendering {len(ids)} motions x 4 conditions -> {OUT}")
    for mid in ids:
        render_motion(mid, model)
    print("done")


if __name__ == "__main__":
    main()
