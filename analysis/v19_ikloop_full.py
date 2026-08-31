"""
Extended IK-in-loop frontier compute: denser with-IK grid + all 3 generators.
Trains missing with-IK points, then evaluates every ikabl checkpoint on
T2M-GPT / MoMask / MDM. Gaussian frontier is read from transfer.json at plot time.
Output: analysis/v19/ikloop_frontier_full.json
"""
import os, sys, json, glob, subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from utils.metrics import compute_fsr, compute_jitter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PY = sys.executable
ANA = "analysis/v19"
GENS = {
    't2mgpt': "data/test_inputs/t2mgpt/t2mgpt_raw_joints",
    'momask': "data/test_inputs/momask_pool",
    'mdm':    "data/test_inputs/mdm/mdm_raw_joints",
}
# denser with-IK grid (adds 055/068/082 to the existing 045/060/075/088/094)
WITHIK = [(0.45, 0.20, '045'), (0.55, 0.20, '055'), (0.60, 0.20, '060'),
          (0.68, 0.18, '068'), (0.75, 0.15, '075'), (0.82, 0.12, '082'),
          (0.88, 0.10, '088'), (0.94, 0.05, '094')]
NOIK = [(0.45, 0.20, '045'), (0.60, 0.20, '060'), (0.75, 0.15, '075'),
        (0.88, 0.10, '088'), (0.94, 0.05, '094')]


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


def ensure(regime, jit, anch, tag):
    fulltag = f"ikabl_{regime}_{tag}"
    ckpt = f"checkpoints/v19_{fulltag}/best.pth"
    if not os.path.exists(ckpt):
        extra = ['--no-ik-in-loop'] if regime == 'noik' else []
        cmd = [PY, "training/v19.py", "--epochs", "120", "--jit-share", str(jit),
               "--anch-share", str(anch), "--tag", fulltag] + extra
        print(f"[train] {fulltag} ...", flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return ckpt


def main():
    res = {'withik': [], 'noik': []}
    for regime, grid in (('withik', WITHIK), ('noik', NOIK)):
        for jit, anch, tag in grid:
            ckpt = ensure(regime, jit, anch, tag)
            pt = eval_ckpt(ckpt)
            res[regime].append({'jit_share': jit, 'anch_share': anch, 'tag': tag, **pt})
            print(f"[eval ] {regime} {tag}: "
                  + "  ".join(f"{g} {pt[g]['FSR']*100:.2f}%/{pt[g]['Jitter']*1000:.2f}" for g in GENS),
                  flush=True)
    with open(f"{ANA}/ikloop_frontier_full.json", 'w') as f:
        json.dump(res, f, indent=2)
    print("done ->", f"{ANA}/ikloop_frontier_full.json")


if __name__ == "__main__":
    main()
