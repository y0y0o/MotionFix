"""
Paired-bootstrap confidence intervals for the physics metrics (reviewer #15).
=============================================================================
Reproduces the Table 1/2 means (FSR, Jitter) for the 5-way ablation and adds:
  - per-config 95% CI on the mean (bootstrap over motions)
  - paired-difference 95% CI + bootstrap p-value for the comparisons that matter:
      v19 - gauss        (is Gaussian's apparent edge significant?  reviewer #1)
      v19 - original     (does the delivery point beat original?)
      deskate - original (does physics do the work?)
      gauss - deskate    (does smoothing repair jitter at an FSR cost?)

Configs & metric defs match testing/v19_eval.py exactly (compute_fsr/compute_jitter,
same IK, same Gaussian sigma=1.5). Pairing: all configs share the same motions, so
the difference is bootstrapped by resampling motion indices jointly.

Output: analysis/v19/bootstrap_ci.json
"""
import os, sys, json, glob
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import FootRefiner, smooth_fix, deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from utils.metrics import compute_fsr, compute_jitter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
V18_CKPT = "checkpoints/v18_ik/best.pth"
V19_CKPT = "checkpoints/v19_088a10/best.pth"
GAUSS_SIGMA = 1.5
B = 10000
SEED = 0
ANA = "analysis/v19"
GENS = {
    't2mgpt': "data/test_inputs/t2mgpt/t2mgpt_raw_joints",
    'momask': "data/test_inputs/momask_pool",
    'mdm':    "data/test_inputs/mdm/mdm_raw_joints",
}
CONFIGS = ['original', 'deskate_ik', 'gauss_ik', 'learn_ik', 'v19_ik']
PAIRS = [('v19_ik', 'gauss_ik'), ('v19_ik', 'original'),
         ('deskate_ik', 'original'), ('gauss_ik', 'deskate_ik')]


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


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    m18 = FootRefiner().to(DEVICE)
    m18.load_state_dict(torch.load(V18_CKPT, map_location=DEVICE)['model_state_dict'])
    m18.eval()
    m19 = V19Smoother().to(DEVICE)
    m19.load_state_dict(torch.load(V19_CKPT, map_location=DEVICE)['model_state_dict'])
    m19.eval()

    out = {'B': B, 'seed': SEED, 'sigma': GAUSS_SIGMA}
    for gen, d in GENS.items():
        files = sorted(glob.glob(f"{d}/*.npy"))
        n = len(files)
        # per-motion metric arrays
        FSR = {c: np.zeros(n) for c in CONFIGS}
        JIT = {c: np.zeros(n) for c in CONFIGS}
        for k, fp in enumerate(files):
            mo = load(fp)
            built = {
                'original': mo,
                'deskate_ik': apply_ik(mo, deskate_only(mo, 0.0)),
                'gauss_ik': apply_ik(mo, deskate_only(mo, GAUSS_SIGMA)),
                'learn_ik': apply_ik(mo, smooth_fix(mo, m18, DEVICE)),
                'v19_ik': apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE)),
            }
            for c in CONFIGS:
                FSR[c][k] = compute_fsr(built[c])[0]
                JIT[c][k] = compute_jitter(built[c])

        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(B, n))    # shared resample -> paired

        res = {'n': n, 'FSR': {}, 'Jitter': {}, 'pairs_FSR': {}, 'pairs_Jitter': {}}
        for metric, M in (('FSR', FSR), ('Jitter', JIT)):
            for c in CONFIGS:
                boot = M[c][idx].mean(axis=1)
                lo, hi = ci(boot)
                res[metric][c] = {'mean': float(M[c].mean()), 'ci95': [lo, hi]}
            pk = 'pairs_' + metric
            for a, b in PAIRS:
                dboot = (M[a][idx] - M[b][idx]).mean(axis=1)
                lo, hi = ci(dboot)
                # two-sided bootstrap p-value
                p = 2.0 * min((dboot <= 0).mean(), (dboot >= 0).mean())
                res[pk][f'{a}-{b}'] = {
                    'diff': float(M[a].mean() - M[b].mean()),
                    'ci95': [lo, hi], 'p': float(min(p, 1.0)),
                    'significant': bool(lo > 0 or hi < 0),
                }
        out[gen] = res

        print(f"\n=== {gen} (n={n}) — FSR% mean [95% CI] ===")
        for c in CONFIGS:
            m = res['FSR'][c]
            print(f"  {c:12s} {m['mean']*100:5.2f}  [{m['ci95'][0]*100:5.2f}, {m['ci95'][1]*100:5.2f}]")
        print(f"  paired FSR diffs (pp):")
        for a, b in PAIRS:
            r = res['pairs_FSR'][f'{a}-{b}']
            sig = 'SIG' if r['significant'] else 'ns '
            print(f"    {a}-{b:12s} {r['diff']*100:+5.2f}  "
                  f"[{r['ci95'][0]*100:+5.2f},{r['ci95'][1]*100:+5.2f}]  p={r['p']:.3f} {sig}")

    with open(f"{ANA}/bootstrap_ci.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {ANA}/bootstrap_ci.json")


if __name__ == "__main__":
    main()
