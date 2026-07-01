"""
V17 — Train the Foot Smoother (Stage 2 of the hybrid)
======================================================
Self-supervised on the PHYSICS output of the 40 training motions.
The smoother learns to reduce physics-induced jitter while staying
anchored to the physics correction (preserving the FSR gain).
"""
import torch, torch.optim as optim, numpy as np, os, sys, time, glob, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v17 import (FootSmoother, SmootherLoss, FOOT_XZ_DIMS, FOOT_JOINTS,
                        gaussian_smooth_traj)
from utils.physics_fix import physics_foot_fix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
TEST_NAMES = "data/training/v15/test_names.json"   # reuse same held-out split
SAVE_DIR = "checkpoints/v17"
LOG_PATH = "logs/v17_train.log"

NUM_EPOCHS = 300
LR = 0.001
DAMP = 0.0
MAX_LEN = 196
# Loss weights
L_MATCH = 1.0
L_SKATE = 8.0
GAUSS_SIGMA = 1.5   # temporal smoothing strength for the target
H_THRESH = 0.05
CONTACT_TEMP = 0.02

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Foot Y flat dims (for contact detection): joints 7,8,10,11 → *3+1
FOOT_Y_DIMS = [j*3+1 for j in FOOT_JOINTS]


def log(msg, p=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if p:
        print(line)


def build_dataset():
    """Physics-correct 40 train motions; cache foot XZ (input), Gaussian-smoothed
    XZ (target), and foot Y (contact detection)."""
    with open(TEST_NAMES) as f:
        held = set(json.load(f))
    files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))

    phys_xz_list, target_xz_list, foot_y_list, names = [], [], [], []
    for fp in files:
        name = os.path.basename(fp).replace('.npy', '')
        if name in held:
            continue
        m = np.load(fp).astype(np.float32)
        if m.ndim == 4:
            m = m[0]
        phys, _ = physics_foot_fix(m, damp_factor=DAMP, return_stats=True)
        phys_flat = phys.reshape(phys.shape[0], -1)

        foot_xz = phys_flat[:, FOOT_XZ_DIMS]                  # (T, 8) physics
        foot_y = phys_flat[:, FOOT_Y_DIMS]                   # (T, 4)

        # ── Contact-aware target ──
        # At contact (foot on ground): keep physics (frozen → low FSR).
        # In air / transition: Gaussian-smoothed (remove boundary jitter).
        smoothed = gaussian_smooth_traj(foot_xz, sigma=GAUSS_SIGMA)  # (T, 8)
        # per-joint contact weight (T, 4) → expand to XZ (T, 8)
        cw = np.zeros((foot_y.shape[0], 4), dtype=np.float32)
        for jj in range(4):
            g = np.percentile(foot_y[:, jj], 5)
            cw[:, jj] = 1.0 / (1.0 + np.exp((foot_y[:, jj] - (g + H_THRESH)) / CONTACT_TEMP))
        cw_xz = np.repeat(cw, 2, axis=1)                     # (T, 8) [j7,j7,j8,j8,...]
        # blend: contact → physics, air → smoothed
        target_xz = cw_xz * foot_xz + (1.0 - cw_xz) * smoothed

        T = foot_xz.shape[0]
        def fit(a):
            if a.shape[0] > MAX_LEN: return a[:MAX_LEN]
            if a.shape[0] < MAX_LEN: return np.pad(a, ((0, MAX_LEN-a.shape[0]), (0,0)), mode='edge')
            return a
        phys_xz_list.append(torch.from_numpy(fit(foot_xz).astype(np.float32)))
        target_xz_list.append(torch.from_numpy(fit(target_xz).astype(np.float32)))
        foot_y_list.append(torch.from_numpy(fit(foot_y).astype(np.float32)))
        names.append(name)

    return (torch.stack(phys_xz_list), torch.stack(target_xz_list),
            torch.stack(foot_y_list), names)


def main():
    log("=" * 68)
    log("  V17 — Foot Smoother Training (Stage 2 of hybrid)")
    log("=" * 68)
    log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  Device: {DEVICE}")
    log(f"  Physics damp={DAMP}, Epochs={NUM_EPOCHS}, LR={LR}")
    log(f"  Loss weights: match={L_MATCH}, skate={L_SKATE}, gauss_sigma={GAUSS_SIGMA}")
    log("")

    phys_xz, target_xz, foot_y, names = build_dataset()
    phys_xz, target_xz, foot_y = phys_xz.to(DEVICE), target_xz.to(DEVICE), foot_y.to(DEVICE)
    log(f"  Train motions: {len(names)} (physics-corrected; Gaussian target cached)")

    model = FootSmoother().to(DEVICE)
    log(f"  FootSmoother: {model.n_params():,} params (lightweight 1D-CNN)")
    criterion = SmootherLoss(lambda_match=L_MATCH, lambda_skate=L_SKATE)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    start_epoch, best = 0, float('inf')
    resume = f"{SAVE_DIR}/latest.pth"
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['optimizer_state_dict'])
        start_epoch = ck['epoch'] + 1
        best = ck.get('loss', float('inf'))
        log(f"  📎 Resumed from epoch {start_epoch}")

    log("")
    log(f"  {'Epoch':>5} | {'Total':>8} | {'match':>8} | {'skate':>8} | {'LR':>8}")
    log(f"  {'─'*5} | {'─'*8} | {'─'*8} | {'─'*8} | {'─'*8}")

    BATCH = 8
    n = len(names)
    t0 = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        S = {'total': 0, 'match': 0, 'skate': 0}
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            pxz = phys_xz[idx]                  # (b, T, 8) physics input
            txz = target_xz[idx]               # (b, T, 8) gaussian target
            fy = foot_y[idx]                    # (b, T, 4)

            residual = model(pxz)              # (b, T, 8)
            final_xz = pxz + residual

            loss, comps = criterion(final_xz, txz, fy)
            opt.zero_grad(); loss.backward(); opt.step()

            S['total'] += loss.item()
            for k in ('match', 'skate'):
                S[k] += comps[k]
            nb += 1
        sched.step()
        for k in S: S[k] /= nb
        lr = opt.param_groups[0]['lr']

        if (epoch+1) % 20 == 0 or epoch < 3:
            log(f"  {epoch+1:>5} | {S['total']:>8.5f} | {S['match']:>8.6f} | "
                f"{S['skate']:>8.6f} | {lr:>8.5f}")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(), 'loss': S['total']},
                   f"{SAVE_DIR}/latest.pth")
        if S['total'] < best:
            best = S['total']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(), 'loss': best},
                       f"{SAVE_DIR}/best.pth")

    log("")
    log("=" * 68)
    log(f"  ✅ Done — Best loss {best:.4f} — {time.time()-t0:.1f}s")
    log(f"  Checkpoint: {SAVE_DIR}/best.pth")
    log("=" * 68)


if __name__ == "__main__":
    main()
