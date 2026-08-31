"""
IK-in-the-loop ablation as a FRONTIER (reviewer priority #2).
=============================================================
Decides whether contribution #2 (IK-in-the-loop = technical novelty) buys anything
beyond moving along the FSR-jitter frontier. Trains the SAME V19 model/loss at a
grid of jit-shares under two regimes:
  - with-IK   : loss computed on the post-IK trajectory (the delivery method)
  - without-IK: loss computed on the pre-IK smoother output (--no-ik-in-loop)
Inference applies 2-bone IK in BOTH regimes; only the training objective differs.
Each checkpoint is evaluated on T2M-GPT and MoMask (n=200) -> (FSR, jitter).

If the with-IK frontier sits below/left of the without-IK one, the novelty holds.
If they overlap, IK-in-the-loop only changes the operating point, not the frontier,
and contribution #2 must be downgraded.

Output: analysis/v19/ikloop_frontier.json + analysis/v19/ikloop_frontier.png
"""
import os, sys, json, glob, subprocess, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from utils.metrics import compute_fsr, compute_jitter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PY = sys.executable
EPOCHS = 120
ANA = "analysis/v19"
GENS = {
    't2mgpt': "data/test_inputs/t2mgpt/t2mgpt_raw_joints",
    'momask': "data/test_inputs/momask_pool",
}
# (jit_share, anch_share, tag) — anch lowered at the high-jit end to reach low jitter
GRID = [(0.45, 0.20, '045'), (0.60, 0.20, '060'), (0.75, 0.15, '075'),
        (0.88, 0.10, '088'), (0.94, 0.05, '094')]
REGIMES = [('withik', []), ('noik', ['--no-ik-in-loop'])]


def load(fp):
    m = np.load(fp).astype(np.float32)
    return m[0] if m.ndim == 4 else m


def eval_ckpt(ckpt):
    m = V19Smoother().to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)['model_state_dict'])
    m.eval()
    out = {}
    for g, d in GENS.items():
        files = sorted(glob.glob(f"{d}/*.npy"))
        fsr, jit = [], []
        for fp in files:
            mo = load(fp)
            corr = apply_ik(mo, smooth_fix_v19(mo, m, DEVICE))
            fsr.append(compute_fsr(corr)[0]); jit.append(compute_jitter(corr))
        out[g] = {'FSR': float(np.mean(fsr)), 'Jitter': float(np.mean(jit)), 'n': len(files)}
    return out


def main():
    t0 = time.time()
    results = {r: [] for r, _ in REGIMES}
    for regime, extra in REGIMES:
        for jit, anch, tag in GRID:
            fulltag = f"ikabl_{regime}_{tag}"
            ckdir = f"checkpoints/v19_{fulltag}"
            ckpt = f"{ckdir}/best.pth"
            if not os.path.exists(ckpt):
                cmd = [PY, "training/v19.py", "--epochs", str(EPOCHS),
                       "--jit-share", str(jit), "--anch-share", str(anch),
                       "--tag", fulltag] + extra
                print(f"[train] {regime} jit={jit} anch={anch} ...", flush=True)
                subprocess.run(cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            pt = eval_ckpt(ckpt)
            row = {'jit_share': jit, 'anch_share': anch, 'tag': tag, **pt}
            results[regime].append(row)
            print(f"[eval ] {regime} jit={jit}: "
                  + "  ".join(f"{g} FSR {pt[g]['FSR']*100:.2f}% jit {pt[g]['Jitter']:.5f}"
                              for g in GENS), flush=True)

    with open(f"{ANA}/ikloop_frontier.json", 'w') as f:
        json.dump(results, f, indent=2)

    # plot: one panel per generator, two curves
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(GENS), figsize=(11, 4.2), facecolor='#fcfcfb')
    if len(GENS) == 1:
        axes = [axes]
    col = {'withik': '#2a78d6', 'noik': '#c2582b'}
    for ax, g in zip(axes, GENS):
        for regime, _ in REGIMES:
            pts = sorted(results[regime], key=lambda r: r[g]['Jitter'])
            xs = [r[g]['Jitter'] * 1000 for r in pts]
            ys = [r[g]['FSR'] * 100 for r in pts]
            ax.plot(xs, ys, '-o', color=col[regime], label=regime, alpha=0.9)
        ax.set_title(f"{g} (n=200)", fontsize=11)
        ax.set_xlabel('Jitter ×10³  (lower better →)')
        ax.set_ylabel('FSR %  (lower better →)')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(True, color='#ebeae5')
        ax.legend(frameon=False)
    fig.suptitle('IK-in-the-loop ablation: does the training objective move the FSR–jitter frontier?',
                 fontsize=12.5, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{ANA}/ikloop_frontier.png", dpi=160, facecolor='#fcfcfb')
    print(f"\n-> {ANA}/ikloop_frontier.json | {ANA}/ikloop_frontier.png  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
