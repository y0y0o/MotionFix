"""
Cross-generator fixed-parameter transfer.
=========================================
Claim under test: the learned smoother is a drop-in (ONE fixed model, no
per-generator tuning), whereas a Gaussian needs its sigma re-tuned per generator.

Tuning target (generator-agnostic): de-skate raises jitter; a smoother should
bring jitter back to the generator's ORIGINAL level. So the "right" sigma for a
generator is the one whose post-smooth jitter == that generator's original jitter.

Protocol:
  - Sweep Gaussian sigma on a grid for all 3 generators -> per-sigma (FSR, jitter).
  - SOURCE = T2M-GPT. sigma* = sigma restoring T2M-GPT's original jitter (interp).
  - FROZEN transfer: apply that exact sigma* to MoMask / MDM (interp their frontier).
  - ORACLE: each generator's OWN sigma restoring its OWN original jitter (interp).
  - LEARNED: the single fixed v19_088a10 model, applied unchanged to all three
    (trained on HumanML3D GT — tuned on NONE of the three generators).

Positive result = the frozen Gaussian degrades on the target generators (FSR worse
than that generator's oracle, and/or jitter misses the target), while the
zero-tuning learned model degrades less.

Metric defs match testing/v19_eval.py (compute_fsr / compute_jitter).
Output: analysis/v19/transfer.json
"""
import os, sys, json, glob
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from utils.metrics import compute_fsr, compute_jitter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
V19_CKPT = "checkpoints/v19_088a10/best.pth"
SOURCE = 't2mgpt'
GENS = {
    't2mgpt': "data/test_inputs/t2mgpt/t2mgpt_raw_joints",
    'momask': "data/test_inputs/momask_pool",
    'mdm':    "data/test_inputs/mdm/mdm_raw_joints",
}
SIGMAS = [round(x, 2) for x in np.arange(0.5, 3.01, 0.25)]
ANA = "analysis/v19"


def load(fp):
    m = np.load(fp).astype(np.float32)
    return m[0] if m.ndim == 4 else m


def deskate_gauss(m, sigma):
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    if sigma > 0:
        tgt = gaussian_filter1d(tgt, sigma=sigma, axis=0, mode='nearest')
    out = flat.copy()
    out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def mean_fsr_jit(motions):
    fsr = np.mean([compute_fsr(m)[0] for m in motions])
    jit = np.mean([compute_jitter(m) for m in motions])
    return float(fsr), float(jit)


def interp_at(xs, ys, x0):
    """Linear interp ys at x0 (xs need not be monotonic in value; sort by xs)."""
    xs = np.asarray(xs); ys = np.asarray(ys)
    order = np.argsort(xs)
    return float(np.interp(x0, xs[order], ys[order]))


def sigma_for_target_jit(sigmas, jits, target):
    """The sigma whose jitter == target (interp; jitter decreasing in sigma)."""
    s = np.asarray(sigmas); j = np.asarray(jits)
    order = np.argsort(j)          # ascending jitter
    return float(np.interp(target, j[order], s[order]))


def main():
    m19 = V19Smoother().to(DEVICE)
    m19.load_state_dict(torch.load(V19_CKPT, map_location=DEVICE)['model_state_dict'])
    m19.eval()

    data = {}
    for gen, d in GENS.items():
        files = sorted(glob.glob(f"{d}/*.npy"))
        raws = [load(fp) for fp in files]
        o_fsr, o_jit = mean_fsr_jit(raws)                       # original
        # learned (fixed model), through IK
        learned = [apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE)) for mo in raws]
        l_fsr, l_jit = mean_fsr_jit(learned)
        # gaussian frontier
        front = {}
        for sig in SIGMAS:
            g = [apply_ik(mo, deskate_gauss(mo, sig)) for mo in raws]
            front[sig] = mean_fsr_jit(g)
        data[gen] = {
            'n': len(files),
            'original': {'FSR': o_fsr, 'Jitter': o_jit},
            'learned':  {'FSR': l_fsr, 'Jitter': l_jit},
            'gauss_frontier': {str(s): {'FSR': front[s][0], 'Jitter': front[s][1]} for s in SIGMAS},
        }
        print(f"[{gen}] n={len(files)}  orig FSR {o_fsr*100:.2f}% jit {o_jit:.5f} | "
              f"learned FSR {l_fsr*100:.2f}% jit {l_jit:.5f}")

    # sigma* on SOURCE: restore source's original jitter
    src = data[SOURCE]
    s_sig = list(src['gauss_frontier'].keys())
    s_sigf = [float(x) for x in s_sig]
    s_jit = [src['gauss_frontier'][s]['Jitter'] for s in s_sig]
    s_fsr = [src['gauss_frontier'][s]['FSR'] for s in s_sig]
    sigma_star = sigma_for_target_jit(s_sigf, s_jit, src['original']['Jitter'])
    sigma_star = float(np.clip(sigma_star, SIGMAS[0], SIGMAS[-1]))

    print(f"\nsigma* (restores {SOURCE} original jitter) = {sigma_star:.3f}\n")

    rows = {'sigma_star_source': sigma_star, 'source': SOURCE, 'per_generator': {}}
    hdr = f"{'gen':7s} | {'FROZEN g@s* ':>20s} | {'ORACLE g@ownJ0':>20s} | {'LEARNED(no tune)':>20s}"
    print(hdr); print('-' * len(hdr))
    for gen in GENS:
        g = data[gen]
        sig = [float(x) for x in g['gauss_frontier'].keys()]
        jit = [g['gauss_frontier'][str(s)]['Jitter'] for s in SIGMAS]
        fsr = [g['gauss_frontier'][str(s)]['FSR'] for s in SIGMAS]
        J0 = g['original']['Jitter']
        # frozen sigma* applied here
        f_fsr = interp_at(sig, fsr, sigma_star)
        f_jit = interp_at(sig, jit, sigma_star)
        # oracle: sigma restoring THIS gen's J0 -> FSR at jitter=J0
        o_fsr_iso = interp_at(jit, fsr, J0)   # FSR at jitter==J0 on this frontier
        sig_oracle = sigma_for_target_jit(sig, jit, J0)
        # learned
        lf, lj = g['learned']['FSR'], g['learned']['Jitter']
        rows['per_generator'][gen] = {
            'original_jitter': J0,
            'frozen_sigma_star': {'sigma': sigma_star, 'FSR': f_fsr, 'Jitter': f_jit},
            'oracle': {'sigma': sig_oracle, 'FSR_at_J0': o_fsr_iso, 'target_Jitter': J0},
            'learned': {'FSR': lf, 'Jitter': lj},
            'transfer_penalty_FSR_pp': (f_fsr - o_fsr_iso) * 100,   # frozen vs oracle
            'learned_vs_frozen_FSR_pp': (lf - f_fsr) * 100,
        }
        print(f"{gen:7s} | FSR {f_fsr*100:5.2f}% jit {f_jit:.4f} | "
              f"FSR {o_fsr_iso*100:5.2f}% @J0     | FSR {lf*100:5.2f}% jit {lj:.4f}")

    with open(f"{ANA}/transfer.json", 'w') as f:
        json.dump({'grid': SIGMAS, 'data': data, 'summary': rows}, f, indent=2)

    print("\n-- transfer penalty (frozen Gaussian sigma* vs each gen's oracle, pp of FSR) --")
    for gen in GENS:
        r = rows['per_generator'][gen]
        tag = '' if gen != SOURCE else '  (source: no penalty by construction)'
        print(f"  {gen:7s}  frozen-oracle = {r['transfer_penalty_FSR_pp']:+.2f}pp   "
              f"learned-frozen = {r['learned_vs_frozen_FSR_pp']:+.2f}pp{tag}")
    print(f"\n-> {ANA}/transfer.json")


if __name__ == "__main__":
    main()
