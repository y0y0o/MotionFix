"""
V19 — train the IK-in-the-loop, contact-partitioned adaptive foot smoother
=========================================================================
Fixes, relative to training/v18_ik.py:

  A1  the 2-bone IK ankle clamp is INSIDE the forward pass, so the loss is
      computed on the trajectory that is actually evaluated.
  A3  the anti-skate term is a threshold-aligned soft COUNT at contact, and
      smoothing is partitioned by contact phase (hard in the air, frozen at
      contact) — the one behaviour a global Gaussian cannot reproduce.
  B3  a real validation split; checkpoints are selected on VALIDATION loss.
      (v18_ik.py selected on TRAINING loss with no val set at all.)
  B5  lambdas are auto-calibrated by per-term gradient norm, so the weights
      mean what they look like. v18_ik.py used 250:12:1 across terms whose
      raw magnitudes differ by ~3 orders (acc^2 ~1e-6 vs |v| ~1e-3).
  B6  every term is mask-normalised by true sequence length.
  B7  trained on ~3.4k HumanML3D motions instead of 40 MoMask motions, so
      MoMask/MDM/T2M-GPT are ALL out-of-distribution at eval — leakage is
      structurally impossible.

Usage:
    python training/v19.py                 # build cache (once) + train
    python training/v19.py --rebuild       # force cache rebuild
"""
import os, sys, glob, json, time, argparse
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import compute_contact_weight_np, deskated_target, FOOT_XZ_DIMS, FOOT_Y_DIMS
from models.v19 import V19Smoother, V19Loss, torch_ankle_ik, LEGS, ANKLES, H_THRESH, TEMP

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Built by data/prep/v19.py from RAW HumanML3D new_joint_vecs. Do NOT point this
# at data/training/v14/ — that corpus was de-normalised by a bug in prep/v14.py
# and its feet are vertically compressed into a 9cm band (see data/prep/v19.py).
CACHE = "data/training/v19_cache.pt"
SAVE_DIR = "checkpoints/v19"
LOG_PATH = "logs/v19_train.log"

MAX_LEN = 196
VAL_FRAC = 0.12
NUM_EPOCHS = 300
LR = 2e-4        # 2e-3 collapses the model back to residual=0: Adam's first
                 # step overshoots and the gradient dies. See docs/v19_devlog.md §6.
BATCH = 16
SEED = 0

# Target share of total gradient norm per term (B5). Calibrated at step 0.
# `anch` is deliberately heavy: it is the tripwire against the V16 failure mode
# (model games the differentiable skate proxy by walking the foot off-body).
GRAD_TARGET = {'jit_air': 0.30, 'jit_all': 0.15, 'skate': 0.35, 'anch': 0.20}

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg, p=True):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if p:
        print(line, flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Forward: smoother -> differentiable IK -> rigid toe
# ═══════════════════════════════════════════════════════════════════════

def forward_pipeline(model, b, use_ik=True):
    """
    Returns the POST-IK ankle XZ and the rigidly derived toe XZ — i.e. exactly
    the quantities utils.metrics will later measure.

    use_ik=False is the ABLATION for the IK-in-the-loop contribution: the loss is
    computed on the PRE-IK smoother output instead of the post-IK trajectory
    (inference still applies IK either way). This isolates the effect of training
    with the IK clamp inside the forward pass.
    """
    res = model(b['des4'], b['orig4'], b['w4'])
    ank_xz = b['des4'] + res                                  # (B,T,4) pre-IK

    if not use_ik:
        toe = []
        for li in range(2):
            toe.append(ank_xz[..., 2 * li]     + b['toeoff'][:, :, li, 0])
            toe.append(ank_xz[..., 2 * li + 1] + b['toeoff'][:, :, li, 2])
        return ank_xz, torch.stack(toe, -1)

    ank_ik, toe_xz = [], []
    for li in range(2):
        tgt = torch.stack([ank_xz[..., 2 * li],
                           b['ankY'][..., li],
                           ank_xz[..., 2 * li + 1]], dim=-1)   # (B,T,3)
        a = torch_ankle_ik(b['hip'][:, :, li], tgt, b['L1'][..., li], b['L2'][..., li])
        ank_ik.append(a[..., 0]); ank_ik.append(a[..., 2])
        t = a + b['toeoff'][:, :, li]                          # rigid toe follow
        toe_xz.append(t[..., 0]); toe_xz.append(t[..., 2])

    return torch.stack(ank_ik, -1), torch.stack(toe_xz, -1)


def foot_err(ank_ik, b):
    """Tripwire for proxy gaming (the V16 failure mode): ankle drift vs original."""
    d = ank_ik - b['orig4']
    per_leg = torch.sqrt((d[..., 0::2] ** 2 + d[..., 1::2] ** 2).clamp(min=1e-12))
    m = b['mask'].unsqueeze(-1)
    return float((per_leg * m).sum() / m.sum().clamp(min=1.0) / 1.0)


def calibrate_lambdas(model, crit, b, targets=GRAD_TARGET, use_ik=True):
    """
    B5 — set each lambda so the term's share of total gradient norm matches
    `targets`. Removes the units problem (acc^2 ~1e-6 vs a soft count ~1e-1).
    """
    ank_ik, toe_xz = forward_pipeline(model, b, use_ik)
    terms = crit.terms(ank_ik, toe_xz, b['des4'], b['w4'], b['mask'])

    norms = {}
    for k, v in terms.items():
        model.zero_grad(set_to_none=True)
        if not v.requires_grad:
            norms[k] = 0.0
            continue
        v.backward(retain_graph=True)
        n = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                           if p.grad is not None))
        norms[k] = float(n)
    model.zero_grad(set_to_none=True)

    lam = {}
    for k, tgt in targets.items():
        lam[k] = tgt / norms[k] if norms[k] > 1e-12 else 0.0
    # normalise so the total is O(1) — keeps LR meaningful
    s = sum(lam.values()) or 1.0
    lam = {k: v / s for k, v in lam.items()}
    return lam, norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    ap.add_argument('--jit-share', type=float, default=0.45,
                    help="total gradient share given to the two jitter terms; "
                         "the rest goes to skate (anchor is fixed at 0.20). "
                         "This is the KNOB that selects the operating point on "
                         "the FSR-Jitter frontier — sweep it, do not tune it once.")
    ap.add_argument('--anch-share', type=float, default=0.20,
                    help="gradient share for the anti-drift anchor. Held at 0.20 for the"
                         " main sweep; lower it to reach the low-jitter end of the frontier.")
    ap.add_argument('--tag', default='default')
    ap.add_argument('--no-ik-in-loop', action='store_true',
                    help="ABLATION: compute the loss on the pre-IK smoother output "
                         "(inference still applies IK). Isolates IK-in-the-loop.")
    args = ap.parse_args()
    use_ik = not args.no_ik_in_loop

    # jit_share splits 2:1 between air-weighted and unweighted jitter
    js = args.jit_share
    global GRAD_TARGET, SAVE_DIR
    an = args.anch_share
    GRAD_TARGET = {'jit_air': js * 2 / 3, 'jit_all': js / 3,
                   'skate': max(1e-3, 1.0 - an - js), 'anch': an}
    SAVE_DIR = f"checkpoints/v19_{args.tag}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    torch.manual_seed(SEED); np.random.seed(SEED)

    log("=" * 78)
    log("  V19 — IK-in-the-loop, contact-partitioned smoother")
    log("=" * 78)
    log(f"  device={DEVICE}  epochs={args.epochs}  batch={BATCH}  lr={LR}")
    log(f"  tag={args.tag}  jit_share={args.jit_share}  targets={GRAD_TARGET}")

    if not os.path.exists(CACHE):
        log(f"  ERROR: {CACHE} not found — build it first:")
        log(f"         python data/prep/v19.py --n 4000")
        sys.exit(1)
    data = torch.load(CACHE)
    log(f"  loaded cache {CACHE}")

    n = data['des4'].shape[0]
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * VAL_FRAC)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    log(f"  motions: {n}  ->  train {len(tr_idx)} / val {len(val_idx)}")

    data = {k: v.to(DEVICE) for k, v in data.items()}
    take = lambda idx: {k: v[idx] for k, v in data.items()}

    model = V19Smoother().to(DEVICE)
    crit = V19Loss()
    log(f"  V19Smoother: {model.n_params():,} params (V18 was 48,840 over 8 dims)")

    # ── B5: calibrate lambdas on one batch ──
    lam, norms = calibrate_lambdas(model, crit, take(tr_idx[:BATCH]), GRAD_TARGET, use_ik)
    log("")
    log("  lambda calibration (grad-norm share targets):")
    for k in GRAD_TARGET:
        log(f"    {k:<8} raw_grad_norm {norms[k]:.4e}   ->  lambda {lam[k]:.4e}")
    crit.lam_air, crit.lam_all = lam['jit_air'], lam['jit_all']
    crit.lam_skate, crit.lam_anch = lam['skate'], lam['anch']

    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log("")
    log(f"  {'ep':>4} | {'train':>9} | {'val':>9} | {'jit_air':>9} | {'skate':>8} "
        f"| {'anch':>7} | {'FootErr':>7} | {'lr':>8}")
    log("  " + "-" * 84)

    best = float('inf'); best_ep = -1; t0 = time.time()
    hist = []
    for ep in range(args.epochs):
        model.train()
        pm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        tl, nb = 0.0, 0
        for i in range(0, len(pm), BATCH):
            b = take(pm[i:i + BATCH])
            ank_ik, toe_xz = forward_pipeline(model, b, use_ik)
            loss, _ = crit(ank_ik, toe_xz, b['des4'], b['w4'], b['mask'])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss); nb += 1
        sched.step()
        tl /= max(nb, 1)

        # ── validation (B3) ──
        model.eval()
        with torch.no_grad():
            vb = take(val_idx)
            ank_ik, toe_xz = forward_pipeline(model, vb, use_ik)
            vloss, comps = crit(ank_ik, toe_xz, vb['des4'], vb['w4'], vb['mask'])
            vloss = float(vloss)
            fe = foot_err(ank_ik, vb)

        hist.append({'epoch': ep, 'train': tl, 'val': vloss, 'footerr': fe, **comps})
        if vloss < best:
            best, best_ep = vloss, ep
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'val_loss': vloss, 'lambdas': lam, 'foot_err': fe},
                       f"{SAVE_DIR}/best.pth")
        torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                    'val_loss': vloss, 'lambdas': lam}, f"{SAVE_DIR}/latest.pth")

        if (ep + 1) % 20 == 0 or ep < 3:
            mark = " *" if ep == best_ep else ""
            log(f"  {ep+1:>4} | {tl:9.5f} | {vloss:9.5f} | {comps['jit_air']:9.2e} "
                f"| {comps['skate']:8.4f} | {comps['anch']:7.4f} | {fe:7.4f} "
                f"| {opt.param_groups[0]['lr']:8.2e}{mark}")

    with open(f"{SAVE_DIR}/history.json", 'w') as f:
        json.dump({'history': hist, 'lambdas': lam, 'grad_norms': norms}, f, indent=2)

    log("")
    log("=" * 78)
    log(f"  done — best VAL loss {best:.5f} @ epoch {best_ep+1}  ({time.time()-t0:.0f}s)")
    log(f"  checkpoint: {SAVE_DIR}/best.pth")
    log("=" * 78)


if __name__ == "__main__":
    main()
