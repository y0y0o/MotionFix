"""
V16 — Self-Supervised Training (Anti-Gaming Loss)
==================================================
Trains directly on raw MoMask output (world coords). NO target needed.
The loss optimizes the goal (reduce skating) while 3 guard terms
prevent metric-gaming / twitching.

Data:   40 MoMask motions (same train split as V15)
Model:  V14/V16 architecture (Transformer Encoder ×6, 19.1M)
Loss:   V16Loss = skate + smooth + anchor + preserve (self-supervised)
"""
import torch, torch.optim as optim, numpy as np, os, sys, time, glob, pickle, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v16 import MotionFixNetworkV16, V16Loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
TEST_NAMES = "data/training/v15/test_names.json"   # reuse V15 split (same held-out)
SAVE_DIR = "checkpoints/v16"
LOG_PATH = "logs/v16_train.log"

NUM_EPOCHS = 150
LEARNING_RATE = 0.0001
MAX_LEN = 196

# Loss weights (see models/v16.py for rationale)
LAMBDA_SKATE = 10.0
LAMBDA_SMOOTH = 5.0
LAMBDA_ANCHOR = 2.0
LAMBDA_PRESERVE = 20.0

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg, also_print=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line)


def fmt(s):
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def load_train_motions():
    """Load the 40 training MoMask motions (world coords), pad to MAX_LEN."""
    with open(TEST_NAMES) as f:
        test_names = set(json.load(f))

    files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))
    motions = []
    names = []
    for fp in files:
        name = os.path.basename(fp).replace('.npy', '')
        if name in test_names:
            continue  # held out
        m = np.load(fp).astype(np.float32)
        if m.ndim == 4:
            m = m[0]
        T = m.shape[0]
        flat = m.reshape(T, -1)
        if T > MAX_LEN:
            flat = flat[:MAX_LEN]
        elif T < MAX_LEN:
            flat = np.pad(flat, ((0, MAX_LEN - T), (0, 0)), mode='edge')  # edge pad
        motions.append(torch.from_numpy(flat))
        names.append(name)
    return torch.stack(motions), names


def main():
    log("=" * 70)
    log("  V16 — Self-Supervised Training (Anti-Gaming Loss)")
    log("=" * 70)
    log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Device: {DEVICE}")
    log(f"  Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    log(f"  Loss weights: skate={LAMBDA_SKATE}, smooth={LAMBDA_SMOOTH}, "
        f"anchor={LAMBDA_ANCHOR}, preserve={LAMBDA_PRESERVE}")
    log("")

    # Load data
    motions, names = load_train_motions()
    motions = motions.to(DEVICE)
    log(f"  Training motions: {len(names)} (world coords, padded to {MAX_LEN})")
    log(f"  Held-out: from {TEST_NAMES}")
    log("")

    # Model
    model = MotionFixNetworkV16(blend_alpha=0.5).to(DEVICE)
    log(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    criterion = V16Loss(lambda_skate=LAMBDA_SKATE, lambda_smooth=LAMBDA_SMOOTH,
                        lambda_anchor=LAMBDA_ANCHOR, lambda_preserve=LAMBDA_PRESERVE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Resume
    start_epoch = 0
    best_loss = float('inf')
    resume = f"{SAVE_DIR}/latest.pth"
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_loss = ck.get('loss', float('inf'))
        log(f"  📎 Resumed from epoch {start_epoch}")
        log("")

    log(f"  {'Epoch':>5} | {'Total':>8} | {'skate':>7} | {'smooth':>7} | "
        f"{'anchor':>7} | {'presv':>7} | {'LR':>9} | {'Time':>6}")
    log(f"  {'─'*5} | {'─'*8} | {'─'*7} | {'─'*7} | {'─'*7} | {'─'*7} | {'─'*9} | {'─'*6}")

    BATCH = 8
    n = len(names)
    t0 = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        sums = {'total': 0.0, 'skate': 0.0, 'smooth': 0.0, 'anchor': 0.0, 'preserve': 0.0}
        n_batches = 0
        ep_t0 = time.time()

        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            batch = motions[idx]                                  # (b, T, 66) world

            # Self-supervised: model output vs the SAME input
            out = model(batch, foot_only=False, root_relative=True)
            loss, comps = criterion(out, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sums['total'] += loss.item()
            for k in ('skate', 'smooth', 'anchor', 'preserve'):
                sums[k] += comps[k]
            n_batches += 1

        scheduler.step()
        for k in sums: sums[k] /= n_batches
        lr = optimizer.param_groups[0]['lr']
        ep_time = time.time() - ep_t0

        if (epoch + 1) % 5 == 0 or epoch < 5:
            log(f"  {epoch+1:>5} | {sums['total']:>8.4f} | {sums['skate']:>7.4f} | "
                f"{sums['smooth']:>7.4f} | {sums['anchor']:>7.4f} | "
                f"{sums['preserve']:>7.4f} | {lr:>9.6f} | {fmt(ep_time):>6}")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': sums['total']}, f"{SAVE_DIR}/latest.pth")

        if sums['total'] < best_loss:
            best_loss = sums['total']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss}, f"{SAVE_DIR}/best.pth")

    log("")
    log("=" * 70)
    log(f"  ✅ Done — Best loss: {best_loss:.4f} — {fmt(time.time()-t0)}")
    log(f"  Checkpoint: {SAVE_DIR}/best.pth")
    log("=" * 70)


if __name__ == "__main__":
    main()
