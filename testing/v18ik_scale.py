"""
V18+IK — Scaled cross-generator evaluation
===========================================
Run the FINAL pipeline (de-skate → learned smoother → 2-bone IK) on ALL 50
motions of THREE generators (MoMask / MDM / T2M-GPT) with the SAME learned
smoother (trained on MoMask held-out) — a genuine generalization test:
the correction is post-hoc and generator-agnostic, so it should transfer.

For each generator we report Original vs Learned+IK on 7 metrics, plus the
tuned-Gaussian+IK reference (does the learned smoother hold up out-of-domain?).

Outputs:
  analysis/v18_ik_scale/{momask,mdm,t2mgpt}_summary.json
  analysis/v18_ik_scale/cross_model.json     (all means, for the results chart)
  outputs/fixed/v18_ik_scale/<gen>/<name>.npy
  logs/v18ik_scale.log
"""
import torch, numpy as np, os, sys, json, glob
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import FootRefiner, smooth_fix, deskate_xz, FOOT_XZ_DIMS
from models.v18_ik import apply_ik, LEGS
from utils.metrics import (compute_fsr, compute_jitter, compute_floating,
    compute_foot_error, compute_bone_length_consistency,
    compute_ground_penetration, compute_contact_accuracy)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENERATORS = {
    'momask': "data/test_inputs/momask_50/momask_50_results/no_ik",
    'mdm':    "data/test_inputs/mdm/mdm_raw_joints",
    't2mgpt': "data/test_inputs/t2mgpt/t2mgpt_raw_joints",
}
CKPT = "checkpoints/v18_ik/best.pth"
OUT_DIR = "outputs/fixed/v18_ik_scale"
ANA_DIR = "analysis/v18_ik_scale"
LOG_PATH = "logs/v18ik_scale.log"
GAUSS_SIGMA = 1.5
for d in (OUT_DIR, ANA_DIR, os.path.dirname(LOG_PATH)):
    os.makedirs(d, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def M(motion, original):
    return {
        'FSR': compute_fsr(motion)[0], 'Jitter': compute_jitter(motion),
        'Floating': compute_floating(motion)[0], 'FootErr': compute_foot_error(motion, original),
        'ContactAcc': compute_contact_accuracy(motion, original),
        'BoneCV': compute_bone_length_consistency(motion),
        'PenMean': compute_ground_penetration(motion)[0],
    }


def deskate_only(m, sigma=0.0):
    T = m.shape[0]; flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    if sigma > 0:
        tgt = gaussian_filter1d(tgt, sigma=sigma, axis=0, mode='nearest')
    out = flat.copy(); out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def avg(L):
    return {k: float(np.mean([x[k] for x in L])) for k in L[0]}


def main():
    log("=" * 78)
    log("  V18+IK SCALED — cross-generator (MoMask / MDM / T2M-GPT), n=50 each")
    log("=" * 78)
    ck = torch.load(CKPT, map_location=DEVICE)
    model = FootRefiner().to(DEVICE); model.load_state_dict(ck['model_state_dict']); model.eval()
    log(f"  Learned smoother: {model.n_params():,} params (trained on MoMask held-out)")

    cross = {}
    for gen, gdir in GENERATORS.items():
        files = sorted(glob.glob(f"{gdir}/*.npy"))
        log("")
        log(f"── {gen.upper()}  ({len(files)} motions) ──")
        os.makedirs(f"{OUT_DIR}/{gen}", exist_ok=True)
        Ro, Rg, Rl = [], [], []
        for fp in files:
            nm = os.path.basename(fp).replace('.npy', '')
            m = np.load(fp).astype(np.float32)
            if m.ndim == 4: m = m[0]
            gauss = apply_ik(m, deskate_only(m, GAUSS_SIGMA))
            learn = apply_ik(m, smooth_fix(m, model, DEVICE))
            Ro.append(M(m, m)); Rg.append(M(gauss, m)); Rl.append(M(learn, m))
            np.save(f"{OUT_DIR}/{gen}/{nm}_v18ik.npy", learn)
        ao, ag, al = avg(Ro), avg(Rg), avg(Rl)
        cross[gen] = {'original': ao, 'gauss_ik': ag, 'learn_ik': al, 'n': len(files)}
        with open(f"{ANA_DIR}/{gen}_summary.json", 'w') as f:
            json.dump(cross[gen], f, indent=2)
        log(f"  FSR    {ao['FSR']:.1%} → {al['FSR']:.1%}   (gauss {ag['FSR']:.1%})")
        log(f"  Jitter {ao['Jitter']:.4f} → {al['Jitter']:.4f}   (gauss {ag['Jitter']:.4f})")
        log(f"  BoneCV {ao['BoneCV']:.4f} → {al['BoneCV']:.4f}   [leg-tear {'OK' if al['BoneCV']<=ao['BoneCV']+1e-3 else 'BROKEN'}]")
        log(f"  FootErr {al['FootErr']:.3f}m  ContactAcc {al['ContactAcc']:.1%}  Pen {al['PenMean']:.4f}")

    with open(f"{ANA_DIR}/cross_model.json", 'w') as f:
        json.dump(cross, f, indent=2)

    # cross-model table
    log("")
    log("=" * 78)
    log("  CROSS-MODEL SUMMARY (Original → Learned+IK)")
    log("=" * 78)
    log(f"  {'Gen':<8} {'FSR_o':>6} {'FSR_L':>6} | {'Jit_o':>7} {'Jit_L':>7} | {'Bone_o':>6} {'Bone_L':>6} | {'FootErr':>7}")
    log("  " + "-" * 70)
    for gen, d in cross.items():
        o, l = d['original'], d['learn_ik']
        log(f"  {gen:<8} {o['FSR']:>5.1%} {l['FSR']:>5.1%} | {o['Jitter']:>7.4f} {l['Jitter']:>7.4f} | "
            f"{o['BoneCV']:>6.4f} {l['BoneCV']:>6.4f} | {l['FootErr']:>6.3f}m")
    log("=" * 78)
    log(f"  💾 {ANA_DIR}/cross_model.json  |  {OUT_DIR}/")
    log("  ✅ Done.")


if __name__ == "__main__":
    main()
