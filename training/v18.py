"""
V18 — Train the FootRefiner (drift control + smoothness)
=========================================================
Self-supervised on 40 MoMask motions (world coords).
The contact mask (structural) guarantees low FSR; training teaches the model
to (a) keep the trajectory smooth and (b) pull the foot back toward the body
(reduce the 38cm drift of the analytical baseline) via air-weighted fidelity.
"""
import torch, torch.optim as optim, numpy as np, os, sys, time, glob, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import (FootRefiner, V18Loss, refine, compute_contact_weight_np,
                        deskated_target, FOOT_XZ_DIMS, FOOT_Y_DIMS)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
TEST_NAMES = "data/training/v15/test_names.json"
SAVE_DIR = "checkpoints/v18"
LOG_PATH = "logs/v18_train.log"

NUM_EPOCHS = 400
LR = 0.001
MAX_LEN = 196
L_SMOOTH = 10.0
L_FIDELITY = 8.0    # strong: must pull the 38cm drift back toward the body

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
    """Cache foot XZ positions + contact weights + de-skated targets for 40 motions."""
    with open(TEST_NAMES) as f:
        held = set(json.load(f))
    files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))
    pos_list, w_list, tgt_list, names = [], [], [], []
    for fp in files:
        name = os.path.basename(fp).replace('.npy', '')
        if name in held:
            continue
        m = np.load(fp).astype(np.float32)
        if m.ndim == 4:
            m = m[0]
        flat = m.reshape(m.shape[0], -1)
        pos = flat[:, FOOT_XZ_DIMS]                     # (T, 8)
        w = np.repeat(compute_contact_weight_np(flat[:, FOOT_Y_DIMS]), 2, axis=1)  # (T, 8)
        tgt = deskated_target(pos, w)                   # (T, 8) de-skated reference

        def fit(a):
            if a.shape[0] > MAX_LEN: return a[:MAX_LEN]
            if a.shape[0] < MAX_LEN: return np.pad(a, ((0, MAX_LEN-a.shape[0]), (0,0)), mode='edge')
            return a
        pos_list.append(torch.from_numpy(fit(pos).astype(np.float32)))
        w_list.append(torch.from_numpy(fit(w).astype(np.float32)))
        tgt_list.append(torch.from_numpy(fit(tgt).astype(np.float32)))
        names.append(name)
    return torch.stack(pos_list), torch.stack(w_list), torch.stack(tgt_list), names


def main():
    log("=" * 66)
    log("  V18 — FootRefiner Training (drift control + smoothness)")
    log("=" * 66)
    log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  Device: {DEVICE}")
    log(f"  Epochs={NUM_EPOCHS}, LR={LR}, λ_smooth={L_SMOOTH}, λ_fidelity={L_FIDELITY}")
    log("")

    pos, w, tgt, names = build()
    pos, w, tgt = pos.to(DEVICE), w.to(DEVICE), tgt.to(DEVICE)
    log(f"  Train motions: {len(names)} (foot XZ + contact weights + de-skated targets)")

    model = FootRefiner().to(DEVICE)
    log(f"  FootRefiner: {model.n_params():,} params (1D-CNN)")
    crit = V18Loss(lambda_smooth=L_SMOOTH, lambda_fidelity=L_FIDELITY)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=120, gamma=0.5)

    start, best = 0, float('inf')
    resume = f"{SAVE_DIR}/latest.pth"
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['optimizer_state_dict'])
        start = ck['epoch'] + 1
        best = ck.get('loss', float('inf'))
        log(f"  📎 Resumed from epoch {start}")

    log("")
    log(f"  {'Epoch':>5} | {'Total':>8} | {'smooth':>9} | {'fidelity':>9} | {'LR':>8}")
    log(f"  {'─'*5} | {'─'*8} | {'─'*9} | {'─'*9} | {'─'*8}")

    BATCH = 8
    n = len(names)
    t0 = time.time()
    for ep in range(start, NUM_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        S = {'total': 0, 'smooth': 0, 'fidelity': 0}
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            p, ww, tt = pos[idx], w[idx], tgt[idx]
            res = model(p, ww)
            pos_new = refine(p, ww, res)
            loss, comps = crit(pos_new, tt)
            opt.zero_grad(); loss.backward(); opt.step()
            S['total'] += loss.item()
            for k in ('smooth', 'fidelity'): S[k] += comps[k]
            nb += 1
        sched.step()
        for k in S: S[k] /= nb
        lr = opt.param_groups[0]['lr']

        if (ep+1) % 25 == 0 or ep < 3:
            log(f"  {ep+1:>5} | {S['total']:>8.5f} | {S['smooth']:>9.6f} | "
                f"{S['fidelity']:>9.6f} | {lr:>8.5f}")

        torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(), 'loss': S['total']},
                   f"{SAVE_DIR}/latest.pth")
        if S['total'] < best:
            best = S['total']
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(), 'loss': best},
                       f"{SAVE_DIR}/best.pth")

    log("")
    log("=" * 66)
    log(f"  ✅ Done — Best loss {best:.5f} — {time.time()-t0:.1f}s")
    log(f"  Checkpoint: {SAVE_DIR}/best.pth")
    log("=" * 66)


if __name__ == "__main__":
    main()
