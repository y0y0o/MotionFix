"""
V18+IK — Side-by-side comparison videos: Original vs Learned+IK
================================================================
Renders 3D skeleton animations to visually confirm:
  - leg-tear FIXED (shin no longer stretches),
  - foot-flip FIXED (foot orientation preserved),
  - no twitch (smooth), feet planted (no skate).

Foot joints [7,8,10,11] highlighted; shin bones drawn thick.
"""
import numpy as np, os, sys, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

BONES = [
    (0,3),(3,6),(6,9),(9,12),(12,15),          # spine+head
    (0,1),(1,4),(4,7),(7,10),                   # left leg
    (0,2),(2,5),(5,8),(8,11),                   # right leg
    (9,13),(13,16),(16,18),(18,20),             # left arm
    (9,14),(14,17),(17,19),(19,21),             # right arm
]
SHIN_BONES = {(4,7),(5,8)}                       # knee->ankle (leg-tear indicator)
FOOT_JOINTS = {7,8,10,11}

ORIG_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
FIX_DIR  = "outputs/fixed/v18_ik"
OUT_DIR  = "outputs/videos/v18_ik"
os.makedirs(OUT_DIR, exist_ok=True)


def draw(ax, joints, title):
    ax.clear()
    ax.set_title(title, fontsize=11, fontweight='bold')
    for a, b in BONES:
        lw = 3.2 if (a, b) in SHIN_BONES else 1.6
        col = '#E53935' if (a, b) in SHIN_BONES else '#1565C0'
        ax.plot([joints[a,0], joints[b,0]],
                [joints[a,2], joints[b,2]],
                [joints[a,1], joints[b,1]], color=col, lw=lw)
    for j in range(22):
        c = '#FF9800' if j in FOOT_JOINTS else '#1565C0'
        s = 40 if j in FOOT_JOINTS else 12
        ax.scatter(joints[j,0], joints[j,2], joints[j,1], color=c, s=s)


def render(name):
    orig = np.load(f"{ORIG_DIR}/{name}.npy").astype(np.float32)
    if orig.ndim == 4: orig = orig[0]
    fix = np.load(f"{FIX_DIR}/{name}_v18ik.npy").astype(np.float32)
    T = min(orig.shape[0], fix.shape[0], 196)
    orig, fix = orig[:T], fix[:T]

    allp = np.concatenate([orig.reshape(-1,3), fix.reshape(-1,3)], 0)
    xmn,xmx = allp[:,0].min(),allp[:,0].max()
    ymn,ymx = allp[:,1].min(),allp[:,1].max()
    zmn,zmx = allp[:,2].min(),allp[:,2].max()
    rng = max(xmx-xmn, ymx-ymn, zmx-zmn)/2
    cx,cy,cz = (xmn+xmx)/2,(ymn+ymx)/2,(zmn+zmx)/2

    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    fig.suptitle(name[:52], fontsize=12)

    def setlims(ax):
        ax.set_xlim(cx-rng,cx+rng); ax.set_ylim(cz-rng,cz+rng); ax.set_zlim(cy-rng,cy+rng)
        ax.set_box_aspect([1,1,1]); ax.view_init(elev=12, azim=-70)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    def update(t):
        draw(ax1, orig[t], f'Original  (frame {t})')
        draw(ax2, fix[t],  f'Learned+IK  (frame {t})')
        setlims(ax1); setlims(ax2)
        return []

    ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False)
    out = f"{OUT_DIR}/{name}.mp4"
    ani.save(out, writer=FFMpegWriter(fps=20, bitrate=2400))
    plt.close()
    print(f"  ✓ {out}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else [
        os.path.basename(f).replace('_v18ik.npy','')
        for f in sorted(glob.glob(f"{FIX_DIR}/*_v18ik.npy"))
    ]
    print(f"Rendering {len(names)} comparison videos → {OUT_DIR}/", flush=True)
    for nm in names:
        render(nm)
    print("Done.", flush=True)
