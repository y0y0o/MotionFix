"""
De-skate FSR decomposition — answers the reviewer's "why isn't FSR ~0 after
de-skate, since every planted frame is pinned to the same XZ?"

Chain: original -> de-skate (pins ankle at contact-window mean XZ, pre-IK) ->
2-bone IK (clamps the pinned target back to the leg's reachable sphere).

Reports, on T2M-GPT n=200 (same FSR metric as testing/v19_eval.py):
  FSR original
  FSR de-skate ONLY (pre-IK)         <- should be ~0 by construction
  FSR de-skate + IK                  <- the 3.98% in Table 1
  clamp rate = fraction of contact frames where IK moved the ankle off the
               pinned target by > 1mm (i.e. reach-clamp fired)
Output: analysis/v19/deskate_decomp.json
"""
import os, sys, json, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik
from utils.metrics import compute_fsr, compute_contact_labels

SRC = "data/test_inputs/t2mgpt/t2mgpt_raw_joints"
ANA = "analysis/v19"
ANKLES = [7, 8]
CLAMP_EPS = 1e-3   # 1 mm


def load(fp):
    m = np.load(fp).astype(np.float32)
    return m[0] if m.ndim == 4 else m


def deskate_only(m):
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    out = flat.copy()
    out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def main():
    files = sorted(glob.glob(f"{SRC}/*.npy"))
    fsr_o, fsr_d, fsr_dik = [], [], []
    clamp_num, clamp_den = 0, 0

    for fp in files:
        mo = load(fp)
        md = deskate_only(mo)            # pre-IK
        mdik = apply_ik(mo, md)          # post-IK (Table-1 deskate_ik)

        fsr_o.append(compute_fsr(mo)[0])
        fsr_d.append(compute_fsr(md)[0])
        fsr_dik.append(compute_fsr(mdik)[0])

        # clamp rate: at contact frames, did IK move the ankle off the pinned XZ target?
        contact = compute_contact_labels(md, tuple(ANKLES))   # (T,2)
        for i, fj in enumerate(ANKLES):
            planted = contact[:, i] > 0.5
            disp = np.linalg.norm(md[:, fj, :][:, [0, 2]] - mdik[:, fj, :][:, [0, 2]], axis=-1)
            clamp_num += int(((disp > CLAMP_EPS) & planted).sum())
            clamp_den += int(planted.sum())

    out = {
        'n': len(files),
        'src': SRC,
        'clamp_eps_m': CLAMP_EPS,
        'FSR_original': float(np.mean(fsr_o)),
        'FSR_deskate_preIK': float(np.mean(fsr_d)),
        'FSR_deskate_postIK': float(np.mean(fsr_dik)),
        'clamp_rate_contact_frames': clamp_num / max(clamp_den, 1),
    }
    with open(f"{ANA}/deskate_decomp.json", 'w') as f:
        json.dump(out, f, indent=2)

    print(f"T2M-GPT n={out['n']}")
    print(f"  FSR original          {out['FSR_original']*100:6.2f}%")
    print(f"  FSR de-skate (pre-IK) {out['FSR_deskate_preIK']*100:6.2f}%   <- ~0 by construction?")
    print(f"  FSR de-skate + IK     {out['FSR_deskate_postIK']*100:6.2f}%   <- Table 1 = 3.98%")
    print(f"  reach-clamp fired on  {out['clamp_rate_contact_frames']*100:6.2f}% of contact frames")
    print(f"\n-> {ANA}/deskate_decomp.json")


if __name__ == "__main__":
    main()
