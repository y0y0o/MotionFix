"""
Jitter-SPIKE ratios (v19 vs original) for the 088a10 delivery point,
across ALL THREE generators — to check the paper's "on all three generators"
claim for the p99 / max ankle-acceleration reduction.

Same ankle-acceleration trace as analysis/v19_jitter_trace.py, same config
build (deskate/gauss/v19 all through 2-bone IK), pinned to v19_088a10.
Output: analysis/v19/spike_ratios_088a10.json
"""
import os, sys, json, glob
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
V19_CKPT = "checkpoints/v19_088a10/best.pth"
ANKLES = [7, 8]
GENERATORS = {
    't2mgpt': "data/test_inputs/t2mgpt/t2mgpt_raw_joints",
    'momask': "data/test_inputs/momask_pool",
    'mdm':    "data/test_inputs/mdm/mdm_raw_joints",
}
ANA = "analysis/v19"


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


def build(mo, m19):
    return {
        'original': mo,
        'deskate_ik': apply_ik(mo, deskate_only(mo, 0.0)),
        'gauss_ik': apply_ik(mo, deskate_only(mo, 1.5)),
        'v19_ik': apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE)),
    }


def acc_mag(m):
    a = m[:, ANKLES, :]
    v = a[1:] - a[:-1]
    return np.linalg.norm(v[1:] - v[:-1], axis=-1)   # (T-2, 2)


def main():
    m19 = V19Smoother().to(DEVICE)
    m19.load_state_dict(torch.load(V19_CKPT, map_location=DEVICE)['model_state_dict'])
    m19.eval()

    out = {'_checkpoint': V19_CKPT}
    print(f"{'generator':8s} {'n':>3s} | v19/orig  p99     max     rms")
    for gen, d in GENERATORS.items():
        files = sorted(glob.glob(f"{d}/*.npy"))
        stat = {k: {'p99': [], 'max': [], 'rms': []} for k in
                ('original', 'deskate_ik', 'gauss_ik', 'v19_ik')}
        for fp in files:
            mo = load(fp)
            for k, mm in build(mo, m19).items():
                f = acc_mag(mm).ravel()
                stat[k]['p99'].append(float(np.percentile(f, 99)))
                stat[k]['max'].append(float(f.max()))
                stat[k]['rms'].append(float(np.sqrt((f ** 2).mean())))
        summ = {k: {m: float(np.mean(v)) for m, v in s.items()} for k, s in stat.items()}
        o, v = summ['original'], summ['v19_ik']
        ratios = {m: v[m] / o[m] for m in ('p99', 'max', 'rms')}
        out[gen] = {'n': len(files), 'summary': summ, 'v19_over_orig': ratios}
        print(f"{gen:8s} {len(files):3d} |          "
              f"{ratios['p99']:.3f}x  {ratios['max']:.3f}x  {ratios['rms']:.3f}x")

    with open(f"{ANA}/spike_ratios_088a10.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {ANA}/spike_ratios_088a10.json")


if __name__ == "__main__":
    main()
