"""
Per-category FSR, reconstructed for the 088a10 delivery point.
==============================================================
The original `analysis/v19/by_category.json` (dated 2026-07-21) had no surviving
generating script and an unverified checkpoint. This script reconstructs it with
the SAME metric and config-building path as `testing/v19_eval.py`, pinned to the
delivery point `checkpoints/v19_088a10`.

Binning: the 50 base motion ids carry a category token in the momask_pool
filename `p<ID>_<category>_<text>.npy`. Each id is run through all three
generators (momask, t2mgpt, mdm) → 150 motion-generator pairs, aggregated by
category (matches the original n's: rotation 18, walking 21, backward 9,
turning 21, complex 30, dance 30, jumping 21).

FSR mean per category, for original / deskate_ik / v19_ik.
Output: analysis/v19/by_category_088a10.json
"""
import os, sys, re, json, glob
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from utils.metrics import compute_fsr

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
V19_CKPT = "checkpoints/v19_088a10/best.pth"
MOMASK_POOL = "data/test_inputs/momask_pool"
T2MGPT_DIR = "data/test_inputs/t2mgpt/t2mgpt_raw_joints"
MDM_DIR = "data/test_inputs/mdm/mdm_raw_joints"
ANA = "analysis/v19"
CAT_RE = re.compile(r"^p(\d{6})_([a-z]+)_")


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


def fsr(motion):
    return compute_fsr(motion)[0]


def gen_file(gen, mid):
    """Locate the raw-joints file for a given id in a generator dir."""
    if gen == 'momask':
        hit = glob.glob(f"{MOMASK_POOL}/p{mid}_*.npy")
    elif gen == 't2mgpt':
        hit = glob.glob(f"{T2MGPT_DIR}/t2mgpt_{mid}_joints.npy")
    else:
        hit = glob.glob(f"{MDM_DIR}/mdm_{mid}_joints.npy")
    return hit[0] if hit else None


def main():
    m19 = V19Smoother().to(DEVICE)
    ck = torch.load(V19_CKPT, map_location=DEVICE)
    m19.load_state_dict(ck['model_state_dict'])
    m19.eval()

    # category map from the 50 labelled momask_pool filenames
    cat_of = {}
    for fp in sorted(glob.glob(f"{MOMASK_POOL}/*.npy")):
        m = CAT_RE.match(os.path.basename(fp))
        if m:
            cat_of[m.group(1)] = m.group(2)
    print(f"labelled ids: {len(cat_of)}  categories: {sorted(set(cat_of.values()))}")

    agg = {}   # cat -> {config: [fsr,...]}
    missing = []
    for mid, cat in sorted(cat_of.items()):
        for gen in ('momask', 't2mgpt', 'mdm'):
            fp = gen_file(gen, mid)
            if not fp:
                missing.append((gen, mid))
                continue
            mo = load(fp)
            row = agg.setdefault(cat, {'original': [], 'deskate_ik': [], 'v19_ik': []})
            row['original'].append(fsr(mo))
            row['deskate_ik'].append(fsr(apply_ik(mo, deskate_only(mo, 0.0))))
            row['v19_ik'].append(fsr(apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE))))

    out = {}
    for cat in sorted(agg):
        r = agg[cat]
        out[cat] = {
            'n': len(r['original']),
            'original': float(np.mean(r['original'])),
            'deskate_ik': float(np.mean(r['deskate_ik'])),
            'v19_ik': float(np.mean(r['v19_ik'])),
        }
    out['_checkpoint'] = V19_CKPT
    out['_note'] = "150 motion-generator pairs (50 labelled ids x 3 generators); FSR mean per category"
    if missing:
        out['_missing'] = missing

    with open(f"{ANA}/by_category_088a10.json", 'w') as f:
        json.dump(out, f, indent=2)

    # console table
    print(f"\n{'category':10s} {'n':>3s} {'orig':>7s} {'deskate':>7s} {'v19':>7s}  flag")
    reg = 0
    for cat in sorted(out):
        if cat.startswith('_'):
            continue
        c = out[cat]
        flag = ''
        if c['v19_ik'] > c['original']:
            flag = 'v19 > orig (REGRESS)'
            reg += 1
        print(f"{cat:10s} {c['n']:3d} {c['original']:7.4f} {c['deskate_ik']:7.4f} "
              f"{c['v19_ik']:7.4f}  {flag}")
    cats = [k for k in out if not k.startswith('_')]
    worst = max(cats, key=lambda k: out[k]['v19_ik'])
    print(f"\nlargest v19 residual: {worst} ({out[worst]['v19_ik']:.4f})")
    print(f"categories where v19 regresses vs original: {reg}")
    if missing:
        print(f"missing files: {missing}")
    print(f"\n-> {ANA}/by_category_088a10.json")


if __name__ == "__main__":
    main()
