"""
V19 — trace the FSR-Jitter frontier, learned vs analytical
==========================================================
The V18+IK claim was "the learned smoother traces the SAME frontier as a tuned
Gaussian". That claim rested on 6 sweep points from a training setup with the
IK outside the loop, a mean-based skate proxy, no validation split, and 40
training motions. This script re-runs the comparison after those were fixed.

For each generator it evaluates:
  * the learned frontier  — one V19 checkpoint per --jit-share operating point
  * the Gaussian frontier — de-skate + gaussian_filter1d(sigma) + IK
  * two anchors           — Original, and DeSkate+IK (no smoother at all)

All methods end with the SAME 2-bone IK, so the comparison isolates the smoother.

Outputs:
  analysis/v19/frontier.json
  analysis/v19/frontier.png
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from utils.metrics import compute_fsr, compute_jitter, compute_foot_error

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOMASK_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
HELD_JSON = "data/training/v15/test_names.json"
SPLITS = {
    'MoMask (held-out, n=10)': None,          # filled in below
    'MDM (n=50)': "data/test_inputs/mdm/mdm_raw_joints",
    'T2M-GPT (n=50)': "data/test_inputs/t2mgpt/t2mgpt_raw_joints",
}
SIGMAS = [0.5, 0.8, 1.1, 1.5, 2.0, 2.6, 3.4]
ANA = "analysis/v19"
os.makedirs(ANA, exist_ok=True)

# Categorical slots 1 and 6 from the reference palette (fixed order, not cycled).
C_LEARN, C_GAUSS = '#2a78d6', '#eb6834'
C_INK, C_MUTED = '#0b0b0b', '#52514e'


def load(fp):
    m = np.load(fp).astype(np.float32)
    return m[0] if m.ndim == 4 else m


def deskate_only(m, sigma=0.0):
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    if sigma > 0:
        tgt = gaussian_filter1d(tgt, sigma=sigma, axis=0, mode='nearest')
    out = flat.copy()
    out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def score(fixed_list, orig_list):
    return {
        'FSR': float(np.mean([compute_fsr(f)[0] for f in fixed_list])),
        'Jitter': float(np.mean([compute_jitter(f) for f in fixed_list])),
        'FootErr': float(np.mean([compute_foot_error(f, o)
                                  for f, o in zip(fixed_list, orig_list)])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-glob', default='checkpoints/v19_*/best.pth')
    args = ap.parse_args()

    held = set(json.load(open(HELD_JSON)))
    SPLITS['MoMask (held-out, n=10)'] = [
        f for f in sorted(glob.glob(f"{MOMASK_DIR}/*.npy"))
        if os.path.basename(f)[:-4] in held]

    ckpts = sorted(glob.glob(args.ckpt_glob))
    models = []
    for c in ckpts:
        ck = torch.load(c, map_location=DEVICE)
        m = V19Smoother().to(DEVICE); m.load_state_dict(ck['model_state_dict']); m.eval()
        share = ck.get('lambdas', {})
        models.append((os.path.basename(os.path.dirname(c)), m, ck))
    print(f"learned operating points: {[n for n, _, _ in models]}")

    results = {}
    for label, src in SPLITS.items():
        files = src if isinstance(src, list) else sorted(glob.glob(f"{src}/*.npy"))
        motions = [load(f) for f in files]
        r = {'n': len(motions)}

        r['original'] = score(motions, motions)
        r['deskate_ik'] = score([apply_ik(m, deskate_only(m, 0.0)) for m in motions], motions)
        r['gauss'] = [{'sigma': s,
                       **score([apply_ik(m, deskate_only(m, s)) for m in motions], motions)}
                      for s in SIGMAS]
        r['learn'] = [{'tag': n,
                       **score([apply_ik(m, smooth_fix_v19(m, mod, DEVICE)) for m in motions],
                               motions)}
                      for n, mod, _ in models]
        results[label] = r

        print(f"\n── {label} ──")
        print(f"  original    FSR {r['original']['FSR']*100:5.2f}%  Jit {r['original']['Jitter']:.5f}")
        print(f"  deskate_ik  FSR {r['deskate_ik']['FSR']*100:5.2f}%  Jit {r['deskate_ik']['Jitter']:.5f}")
        for g in r['gauss']:
            print(f"  gauss s={g['sigma']:<4} FSR {g['FSR']*100:5.2f}%  Jit {g['Jitter']:.5f}")
        for l in r['learn']:
            print(f"  {l['tag']:<12} FSR {l['FSR']*100:5.2f}%  Jit {l['Jitter']:.5f}")

    with open(f"{ANA}/frontier.json", 'w') as f:
        json.dump(results, f, indent=2)
    plot(results)
    print(f"\n-> {ANA}/frontier.json  |  {ANA}/frontier.png")


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), facecolor='#fcfcfb')
    if n == 1:
        axes = [axes]

    for ax, (label, r) in zip(axes, results.items()):
        ax.set_facecolor('#fcfcfb')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#d9d8d3')
        ax.grid(True, color='#ebeae5', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        gx = [g['Jitter'] for g in r['gauss']]
        gy = [g['FSR'] * 100 for g in r['gauss']]
        lx = [l['Jitter'] for l in r['learn']]
        ly = [l['FSR'] * 100 for l in r['learn']]
        order = np.argsort(lx)
        lx = list(np.array(lx)[order]); ly = list(np.array(ly)[order])

        ax.plot(gx, gy, '-o', color=C_GAUSS, linewidth=2, markersize=8,
                markeredgecolor='#fcfcfb', markeredgewidth=2,
                label='Gaussian σ sweep', zorder=3)
        ax.plot(lx, ly, '-o', color=C_LEARN, linewidth=2, markersize=8,
                markeredgecolor='#fcfcfb', markeredgewidth=2,
                label='Learned (V19) sweep', zorder=4)

        for key, mark, name in (('original', 's', 'Original'),
                                ('deskate_ik', 'D', 'De-skate + IK')):
            p = r[key]
            ax.plot(p['Jitter'], p['FSR'] * 100, mark, color=C_INK, markersize=9,
                    markeredgecolor='#fcfcfb', markeredgewidth=2, zorder=5,
                    label=name)
            ax.annotate(name, (p['Jitter'], p['FSR'] * 100),
                        textcoords='offset points', xytext=(9, 4),
                        fontsize=9, color=C_MUTED)

        ax.set_title(label, fontsize=11, color=C_INK, pad=10, loc='left')
        ax.set_xlabel('Jitter  (foot accel RMS)  →  worse', fontsize=9.5, color=C_MUTED)
        ax.set_ylabel('FSR %  →  worse', fontsize=9.5, color=C_MUTED)
        ax.tick_params(colors=C_MUTED, labelsize=9)

    # figure-level legend above the panels: identity is never color-alone, and it
    # cannot collide with any series inside a panel
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9.5, labelcolor=C_MUTED,
               ncol=4, loc='upper left', bbox_to_anchor=(0.006, 0.945))
    fig.suptitle('Lower-left is better — does the learned smoother reach a frontier '
                 'the tuned Gaussian cannot?',
                 fontsize=12.5, color=C_INK, x=0.006, ha='left', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f"{ANA}/frontier.png", dpi=160, facecolor='#fcfcfb')


if __name__ == "__main__":
    main()
