"""
V18+IK — Train the adaptive foot smoother (position-space, drift-free)
=====================================================================
The smoother operates on the DE-SKATED target (plant-at-segment-mean), NOT on
integrated velocity — so it CANNOT drift (anchored to the no-skate reference).
It optimises the REAL objective directly (jitter + anti-skate), so it can beat
any fixed Gaussian σ (which only matches one heuristic).

  out = deskated + FootRefiner(deskated, w)
  L   = λ_jit·mean(acc²) + λ_skate·mean(|v|·w³) + λ_anch·‖out-deskated‖

This is the LEARNED component of the physics+learning hybrid:
  physics  = de-skate (FSR) + IK (bones)
  learning = adaptive smoother (jitter), contact-aware, beats global Gaussian σ.
"""
import torch, torch.nn as nn, torch.optim as optim, numpy as np, os, sys, time, glob, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import (FootRefiner, compute_contact_weight_np,
                        deskated_target, FOOT_XZ_DIMS, FOOT_Y_DIMS)


class SmoothLoss(nn.Module):
    """
    Direct-objective adaptive smoother loss.

      L = λ_jit   · mean( acc(out)² )            (jitter: 2nd-derivative energy)
        + λ_skate · mean( |v(out)| · w³ )        (FSR: foot velocity≈0 at solid contact)
        + λ_anch  · mean( |out - deskated| )     (anti-drift: stay near no-skate ref)

    The anti-skate term penalises foot VELOCITY (not position) and only at solid
    contact (w³, sharp), so the model rounds boundaries on the AIR side (low jitter)
    while stopping the foot DURING contact (low FSR) — adaptive, contact-aware
    smoothing a single global Gaussian σ cannot reproduce.
    """
    def __init__(self, lambda_jit=300.0, lambda_skate=3.0, lambda_anch=1.0):
        super().__init__()
        self.lj, self.ls, self.la = lambda_jit, lambda_skate, lambda_anch

    def forward(self, out, deskated, w):
        acc = out[:, 2:] - 2 * out[:, 1:-1] + out[:, :-2]
        jit = (acc ** 2).mean()
        v = (out[:, 1:] - out[:, :-1]).abs()
        skate = (v * (w[:, 1:] ** 3)).mean()
        anch = (out - deskated).abs().mean()
        total = self.lj * jit + self.ls * skate + self.la * anch
        return total, {'jit': jit.item(), 'skate': skate.item(), 'anch': anch.item()}


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
TEST_NAMES = "data/training/v15/test_names.json"
SAVE_DIR = "checkpoints/v18_ik"
LOG_PATH = "logs/v18_ik_train.log"

NUM_EPOCHS = 800
LR = 0.002
MAX_LEN = 196
L_JIT = 250.0        # jitter (acc²) — primary objective
L_SKATE = 12.0       # foot velocity≈0 at contact (keeps FSR low)
L_ANCH = 1.0         # stay near de-skated reference (anti-drift)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg, p=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if p:
        print(line)


def build():
    """Cache de-skated foot XZ targets + contact weights for the training motions."""
    with open(TEST_NAMES) as f:
        held = set(json.load(f))
    files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))
    tgt_list, w_list, names = [], [], []
    for fp in files:
        name = os.path.basename(fp).replace('.npy', '')
        if name in held:
            continue
        m = np.load(fp).astype(np.float32)
        if m.ndim == 4:
            m = m[0]
        flat = m.reshape(m.shape[0], -1)
        w = np.repeat(compute_contact_weight_np(flat[:, FOOT_Y_DIMS]), 2, axis=1)
        tgt = deskated_target(flat[:, FOOT_XZ_DIMS], w)

        def fit(a):
            if a.shape[0] > MAX_LEN: return a[:MAX_LEN]
            if a.shape[0] < MAX_LEN: return np.pad(a, ((0, MAX_LEN-a.shape[0]), (0,0)), mode='edge')
            return a
        tgt_list.append(torch.from_numpy(fit(tgt).astype(np.float32)))
        w_list.append(torch.from_numpy(fit(w).astype(np.float32)))
        names.append(name)
    return torch.stack(tgt_list), torch.stack(w_list), names


def main():
    log("=" * 70)
    log("  V18+IK — Adaptive Foot Smoother (direct objective, drift-free)")
    log("=" * 70)
    log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  Device: {DEVICE}")
    log(f"  Epochs={NUM_EPOCHS}, LR={LR}, λ_jit={L_JIT}, λ_skate={L_SKATE}, λ_anch={L_ANCH}")
    log("")

    tgt, w, names = build()
    tgt, w = tgt.to(DEVICE), w.to(DEVICE)
    log(f"  Train motions: {len(names)} (de-skated XZ targets + contact weights)")

    model = FootRefiner().to(DEVICE)
    log(f"  FootRefiner: {model.n_params():,} params (predicts position residual on de-skated)")
    crit = SmoothLoss(lambda_jit=L_JIT, lambda_skate=L_SKATE, lambda_anch=L_ANCH)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)

    log("")
    log(f"  {'Epoch':>5} | {'Total':>8} | {'jit':>9} | {'skate':>9} | {'anch':>8} | {'LR':>8}")
    log(f"  {'─'*5} | {'─'*8} | {'─'*9} | {'─'*9} | {'─'*8} | {'─'*8}")

    BATCH = 8
    n = len(names)
    best = float('inf')
    t0 = time.time()
    for ep in range(NUM_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        S = {'total': 0, 'jit': 0, 'skate': 0, 'anch': 0}
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            tt, ww = tgt[idx], w[idx]
            res = model(tt, ww)
            out = tt + res                          # position-space, drift-free
            loss, comps = crit(out, tt, ww)
            opt.zero_grad(); loss.backward(); opt.step()
            S['total'] += loss.item()
            for k in ('jit', 'skate', 'anch'): S[k] += comps[k]
            nb += 1
        sched.step()
        for k in S: S[k] /= nb
        lr = opt.param_groups[0]['lr']

        if (ep+1) % 50 == 0 or ep < 3:
            log(f"  {ep+1:>5} | {S['total']:>8.5f} | {S['jit']:>9.6f} | "
                f"{S['skate']:>9.6f} | {S['anch']:>8.5f} | {lr:>8.5f}")

        torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(), 'loss': S['total']},
                   f"{SAVE_DIR}/latest.pth")
        if S['total'] < best:
            best = S['total']
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(), 'loss': best},
                       f"{SAVE_DIR}/best.pth")

    log("")
    log("=" * 70)
    log(f"  ✅ Done — Best loss {best:.5f} — {time.time()-t0:.1f}s")
    log(f"  Checkpoint: {SAVE_DIR}/best.pth")
    log("=" * 70)


if __name__ == "__main__":
    main()
