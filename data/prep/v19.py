"""
V19 data prep — de-skate training corpus from raw HumanML3D
===========================================================
Builds the tensor cache that training/v19.py consumes.

WHY THIS EXISTS (a bug found on 2026-07-20)
-------------------------------------------
`data/prep/v14.py::convert_to_joints` applies

    motion_denorm = motion_263 * std + mean

to files from `HumanML3D/new_joint_vecs`. Those files are the RAW feature
vectors — they are not normalised — so this de-normalisation corrupts them.
Measured effect on 146 motions:

                       foot-Y dynamic range   frames judged "in contact"
    v14 (denormalised)        0.090 m                97% of motions >0.8
    raw (correct)             0.269 m                31% in a healthy 0.3-0.8 band
    MoMask (reference)        0.439 m                52% in 0.3-0.8

With the feet vertically compressed into a 9 cm band, the contact heuristic
marks essentially every frame as contact, so `deskated_target` plants a whole
walking trajectory at a single mean point (0.65 m from the original vs 0.042 m
on MoMask). Any model trained on that learns nonsense.

V14/V15 both trained on this corpus and both degenerated to identity. That is
not proof the bug caused those failures — the identity collapse was separately
explained by the L1 signal-washout — but the training data was ALSO broken, so
those two negative results are confounded and should not be cited as clean
evidence that learning cannot work here.

WHAT THIS SCRIPT DOES
---------------------
  1. reads HumanML3D/new_joint_vecs  (no de-normalisation)
  2. recover_from_ric -> (T, 22, 3) world-coordinate joints
  3. keeps motions whose contact fraction is in [MIN_CF, MAX_CF] — outside that
     band the contact heuristic (ground = 5th percentile of foot Y) is not
     meaningful, so the de-skate reference would be garbage
  4. pre-computes everything the V19 loss needs that does not depend on the model
  5. saves data/training/v19_cache.pt

Usage:  python data/prep/v19.py [--n 4000]
"""
import os, sys, glob, argparse, time
from datetime import datetime

import numpy as np
import torch

# T2M-GPT ships its own `utils` package; motionfix has one too. Drop the repo
# dir from sys.path for the import, then put it back (same trick as v14.py).
_MF = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path = [p for p in sys.path if p not in ('', _MF)]
sys.path.insert(0, '/home3/nxkh91/projects/T2M-GPT')
from utils.motion_process import recover_from_ric          # noqa: E402
sys.path.insert(0, _MF)

from models.v18 import compute_contact_weight_np, deskated_target, FOOT_XZ_DIMS, FOOT_Y_DIMS  # noqa: E402
from models.v19 import LEGS, ANKLES, H_THRESH, TEMP        # noqa: E402

SRC = "/home3/nxkh91/projects/HumanML3D/HumanML3D/new_joint_vecs"
OUT = "data/training/v19_cache.pt"
MAX_LEN = 196
MIN_LEN = 40
MIN_CF, MAX_CF = 0.20, 0.90        # keep motions with a meaningful contact/air mix
MAX_DESKATE_DEV = 0.30             # m — reject motions where de-skate is degenerate

ANK_XZ = [ANKLES[0] * 3, ANKLES[0] * 3 + 2, ANKLES[1] * 3, ANKLES[1] * 3 + 2]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000, help="max motions to KEEP")
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{SRC}/*.npy"))
    log(f"source: {len(files)} motions in {SRC}")

    keys = ('des4', 'orig4', 'w4', 'hip', 'L1', 'L2', 'ankY', 'toeoff', 'mask')
    out = {k: [] for k in keys}
    dev_all, cf_all = [], []
    n_short = n_cf = n_dev = 0
    t0 = time.time()

    for fp in files:
        if len(out['des4']) >= args.n:
            break
        d = np.load(fp)
        if d.shape[0] < MIN_LEN:
            n_short += 1
            continue

        # NO de-normalisation — new_joint_vecs are already raw features
        j = recover_from_ric(torch.from_numpy(d).float(), 22).numpy()
        T = min(j.shape[0], MAX_LEN)
        j = j[:T].astype(np.float32)
        flat = j.reshape(T, -1)

        w8 = np.repeat(compute_contact_weight_np(flat[:, FOOT_Y_DIMS], H_THRESH, TEMP), 2, axis=1)
        cf = float((w8[:, 0:4] > 0.5).mean())
        cf_all.append(cf)
        if not (MIN_CF <= cf <= MAX_CF):
            n_cf += 1
            continue

        des8 = deskated_target(flat[:, FOOT_XZ_DIMS], w8)
        des4, orig4 = des8[:, 0:4], flat[:, ANK_XZ]
        dev = float(np.mean(np.sqrt(
            (des4[:, 0::2] - orig4[:, 0::2]) ** 2 + (des4[:, 1::2] - orig4[:, 1::2]) ** 2)))
        dev_all.append(dev)
        if dev > MAX_DESKATE_DEV:
            n_dev += 1
            continue

        def pad(a):
            if a.shape[0] >= MAX_LEN:
                return a[:MAX_LEN]
            pw = [(0, MAX_LEN - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
            return np.pad(a, pw, mode='edge')

        out['des4'].append(pad(des4))
        out['orig4'].append(pad(orig4))
        out['w4'].append(pad(w8[:, 0:4]))
        out['hip'].append(pad(np.stack([j[:, l["hip"]] for l in LEGS], 1)))
        out['L1'].append(pad(np.stack(
            [np.linalg.norm(j[:, l["knee"]] - j[:, l["hip"]], axis=-1) for l in LEGS], 1)))
        out['L2'].append(pad(np.stack(
            [np.linalg.norm(j[:, l["ankle"]] - j[:, l["knee"]], axis=-1) for l in LEGS], 1)))
        out['ankY'].append(pad(np.stack([j[:, l["ankle"], 1] for l in LEGS], 1)))
        out['toeoff'].append(pad(np.stack(
            [j[:, l["toe"]] - j[:, l["ankle"]] for l in LEGS], 1)))
        msk = np.zeros(MAX_LEN, np.float32); msk[:T] = 1.0
        out['mask'].append(msk)

        if len(out['des4']) % 500 == 0:
            log(f"  kept {len(out['des4'])}  ({time.time()-t0:.0f}s)")

    data = {k: torch.from_numpy(np.stack(v).astype(np.float32)) for k, v in out.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(data, args.out)

    dev_all = np.array(dev_all); cf_all = np.array(cf_all)
    log("")
    log(f"  kept {data['des4'].shape[0]} motions -> {args.out}")
    log(f"  rejected: too short {n_short} | contact-fraction out of "
        f"[{MIN_CF},{MAX_CF}] {n_cf} | de-skate deviation > {MAX_DESKATE_DEV}m {n_dev}")
    log(f"  contact fraction (all seen): mean {cf_all.mean():.3f}")
    log(f"  de-skate deviation (kept):   mean {dev_all[dev_all <= MAX_DESKATE_DEV].mean():.4f} m "
        f"(MoMask reference ~0.042 m)")
    log(f"  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
