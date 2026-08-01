"""
V19 — Corrected evaluation harness (supersedes testing/v18ik_scale.py)
======================================================================
Fixes two evaluation defects found on 2026-07-20:

  B1  LEAKAGE — v18ik_scale.py evaluated MoMask on ALL 50 motions, but the
      smoother was TRAINED on 40 of them. The "same-distribution" MoMask
      column was 80% training data. Here MoMask is split explicitly:
        momask_heldout (n=10)  ← the only honest same-distribution number
        momask_train   (n=40)  ← reported separately, to SHOW the gap
      MDM / T2M-GPT (n=50 each) were always clean → unchanged.

  B2  MISSING ABLATION — deskate_ik (physics only, no smoother) has the best
      FSR of any configuration (8.2% on held-out) but was absent from the
      cross-generator table, which hid the fact that the smoothing stage
      REGRESSES FSR. It is now a first-class column.

Methods compared (all end with the same 2-bone IK):
    original      raw generator output
    deskate_ik    de-skate → IK                    (physics only, no learning)
    gauss_ik      de-skate → Gaussian σ → IK       (analytical smoother)
    learn_ik      de-skate → V18 smoother → IK     (the previous delivery)
    v19_ik        de-skate → V19 smoother → IK     (optional, --v19 <ckpt>)

Usage:
    python testing/v19_eval.py                       # baseline: 4 methods
    python testing/v19_eval.py --v19 checkpoints/v19/best.pth
    python testing/v19_eval.py --tag baseline

Outputs:
    analysis/v19/<tag>_results.json
    logs/v19_eval.log
"""
import torch, numpy as np, os, sys, json, glob, argparse
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import FootRefiner, smooth_fix, deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from utils.metrics import (compute_fsr, compute_jitter, compute_floating,
    compute_foot_error, compute_bone_length_consistency,
    compute_ground_penetration, compute_contact_accuracy)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOMASK_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
HELD_JSON = "data/training/v15/test_names.json"
GENERATORS = {
    'mdm':    "data/test_inputs/mdm/mdm_raw_joints",
    't2mgpt': "data/test_inputs/t2mgpt/t2mgpt_raw_joints",
}
V18_CKPT = "checkpoints/v18_ik/best.pth"
ANA_DIR = "analysis/v19"
LOG_PATH = "logs/v19_eval.log"
GAUSS_SIGMA = 1.5
os.makedirs(ANA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def M(motion, original):
    """All 7 metrics."""
    return {
        'FSR': compute_fsr(motion)[0],
        'Jitter': compute_jitter(motion),
        'Floating': compute_floating(motion)[0],
        'FootErr': compute_foot_error(motion, original),
        'ContactAcc': compute_contact_accuracy(motion, original),
        'BoneCV': compute_bone_length_consistency(motion),
        'PenMean': compute_ground_penetration(motion)[0],
    }


def deskate_only(m, sigma=0.0):
    """De-skate, optionally Gaussian-smoothed. Returns pre-IK motion."""
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    if sigma > 0:
        tgt = gaussian_filter1d(tgt, sigma=sigma, axis=0, mode='nearest')
    out = flat.copy()
    out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def avg(L):
    return {k: float(np.mean([x[k] for x in L])) for k in L[0]}


def load(fp):
    m = np.load(fp).astype(np.float32)
    return m[0] if m.ndim == 4 else m


def build_splits():
    """MoMask split by the ACTUAL training held-out list; MDM/T2M-GPT whole."""
    held = set(json.load(open(HELD_JSON)))
    momask = sorted(glob.glob(f"{MOMASK_DIR}/*.npy"))
    ho = [f for f in momask if os.path.basename(f)[:-4] in held]
    tr = [f for f in momask if os.path.basename(f)[:-4] not in held]
    splits = {'momask_heldout': ho, 'momask_train': tr}
    for g, d in GENERATORS.items():
        splits[g] = sorted(glob.glob(f"{d}/*.npy"))
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v19', default=None, help="path to a V19 smoother checkpoint")
    ap.add_argument('--tag', default='baseline')
    args = ap.parse_args()

    log("=" * 84)
    log(f"  V19 EVAL — leak-free splits + deskate_ik ablation   [tag={args.tag}]")
    log("=" * 84)

    m18 = FootRefiner().to(DEVICE)
    m18.load_state_dict(torch.load(V18_CKPT, map_location=DEVICE)['model_state_dict'])
    m18.eval()
    log(f"  V18 smoother: {m18.n_params():,} params  ({V18_CKPT})")

    m19 = None
    if args.v19:
        ck19 = torch.load(args.v19, map_location=DEVICE)
        m19 = V19Smoother().to(DEVICE)
        m19.load_state_dict(ck19['model_state_dict'])
        m19.eval()
        log(f"  V19 smoother: {m19.n_params():,} params  ({args.v19})  "
            f"epoch {ck19.get('epoch','?')}  val_loss {ck19.get('val_loss', float('nan')):.5f}")

    methods = ['original', 'deskate_ik', 'gauss_ik', 'learn_ik'] + (['v19_ik'] if m19 else [])
    splits = build_splits()
    results = {}

    for split, files in splits.items():
        log("")
        log(f"── {split.upper()}  (n={len(files)}) ──")
        acc = {k: [] for k in methods}
        for fp in files:
            mo = load(fp)
            acc['original'].append(M(mo, mo))
            acc['deskate_ik'].append(M(apply_ik(mo, deskate_only(mo, 0.0)), mo))
            acc['gauss_ik'].append(M(apply_ik(mo, deskate_only(mo, GAUSS_SIGMA)), mo))
            acc['learn_ik'].append(M(apply_ik(mo, smooth_fix(mo, m18, DEVICE)), mo))
            if m19:
                acc['v19_ik'].append(M(apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE)), mo))
        results[split] = {k: avg(v) for k, v in acc.items()}
        results[split]['n'] = len(files)

        log(f"  {'method':<12} {'FSR':>7} {'Jitter':>9} {'FootErr':>8} {'BoneCV':>8} {'ContAcc':>8}")
        for k in methods:
            r = results[split][k]
            log(f"  {k:<12} {r['FSR']*100:6.2f}% {r['Jitter']:9.5f} "
                f"{r['FootErr']:8.4f} {r['BoneCV']:8.5f} {r['ContactAcc']*100:7.1f}%")

    out = f"{ANA_DIR}/{args.tag}_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)

    # ── the two headline diagnostics ──
    log("")
    log("=" * 84)
    log("  DIAGNOSTIC 1 — leakage gap (learn_ik: trained-on vs held-out)")
    tr, ho = results['momask_train']['learn_ik'], results['momask_heldout']['learn_ik']
    log(f"    FSR    train {tr['FSR']*100:.2f}%  vs  held-out {ho['FSR']*100:.2f}%   "
        f"(gap {(ho['FSR']-tr['FSR'])*100:+.2f}pp)")
    log(f"    Jitter train {tr['Jitter']:.5f}  vs  held-out {ho['Jitter']:.5f}")

    log("")
    log("  DIAGNOSTIC 2 — does the smoothing stage REGRESS FSR? (vs deskate_ik)")
    for split in results:
        d = results[split]['deskate_ik']['FSR'] * 100
        g = results[split]['gauss_ik']['FSR'] * 100
        l = results[split]['learn_ik']['FSR'] * 100
        line = f"    {split:<16} deskate {d:5.2f}%  gauss {g:5.2f}% ({g-d:+.2f})  learn {l:5.2f}% ({l-d:+.2f})"
        if m19:
            v = results[split]['v19_ik']['FSR'] * 100
            line += f"  v19 {v:5.2f}% ({v-d:+.2f})"
        log(line)
    log("=" * 84)
    log(f"  → {out}")


if __name__ == "__main__":
    main()
