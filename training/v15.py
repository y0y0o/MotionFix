"""
V15 — Physics-Teacher Training
================================
Trains Transformer model to replicate physics-based foot skating correction.

Data:     MoMask (skating) → Physics-fixed (corrected), root-relative
Model:    V14 architecture (Transformer Encoder ×6, 19.1M)
Loss:     V14Loss (foot-XZ focused, λ_foot=3.0)
Strategy: Model learns the physics constraint implicitly from data

Key difference from V14:
  V14: simulated 2cm sliding → clean HumanML3D (weak signal)
  V15: real MoMask skating → physics-corrected version (strong, real signal)
"""
import torch, torch.optim as optim, numpy as np, os, sys, time
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v14 import MotionFixNetworkV14, V14Loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = "data/training/v15"
SAVE_DIR = "checkpoints/v15"
LOG_PATH = "logs/v15_train.log"

BATCH_SIZE = 8       # Small: only ~200 training pairs
NUM_EPOCHS = 100     # More epochs to compensate for small dataset
LEARNING_RATE = 0.00005  # Lower LR for stability
LAMBDA_VEL = 0.5
LAMBDA_FOOT = 5.0     # Higher foot weight (physics target is foot-specific)
LAMBDA_FOOT_Y = 0.3   # Lower Y weight (physics fix doesn't change Y)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg, also_print=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line)


def load_data():
    """Load V15 training pairs, pad to uniform length, return TensorDataset."""
    import glob
    files = sorted(glob.glob(f"{DATA_DIR}/distorted_*.npy"))
    n = len(files)
    print(f"  Loading {n} pairs...")

    # Find max length
    max_len = 0
    lengths = []
    for df in files:
        d = np.load(df)
        lengths.append(d.shape[0])
        if d.shape[0] > max_len:
            max_len = d.shape[0]

    print(f"  Max length: {max_len}")

    all_d, all_t = [], []
    for i, df in enumerate(files):
        tf = df.replace('distorted_', 'target_')
        d = np.load(df).astype(np.float32)
        t = np.load(tf).astype(np.float32)
        T = d.shape[0]

        # Pad to max_len
        if T < max_len:
            d = np.pad(d, ((0, max_len - T), (0, 0)), mode='constant')
            t = np.pad(t, ((0, max_len - T), (0, 0)), mode='constant')

        all_d.append(torch.from_numpy(d))
        all_t.append(torch.from_numpy(t))

    return TensorDataset(torch.stack(all_d), torch.stack(all_t)), max_len, n


def format_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def main():
    log("=" * 70)
    log("  V15 — Physics-Teacher Training")
    log("=" * 70)
    log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Device: {DEVICE}")
    log(f"  Data: {DATA_DIR}")
    log(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    log(f"  λ_foot: {LAMBDA_FOOT}, λ_foot_y: {LAMBDA_FOOT_Y}")
    log("")

    # Load data
    dataset, max_len, n_pairs = load_data()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    log(f"  Training pairs: {n_pairs}, Batches/epoch: {len(loader)}")
    log("")

    # Model
    model = MotionFixNetworkV14(blend_alpha=0.5).to(DEVICE)
    log(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    criterion = V14Loss(lambda_vel=LAMBDA_VEL, lambda_foot=LAMBDA_FOOT,
                        lambda_foot_y=LAMBDA_FOOT_Y)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Resume
    start_epoch = 0
    best_loss = float('inf')
    resume = f"{SAVE_DIR}/latest.pth"
    if os.path.exists(resume):
        ckpt = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('loss', float('inf'))
        log(f"  📎 Resumed from epoch {start_epoch} (loss={best_loss:.4f})")
        log("")

    log(f"  {'Epoch':>5} | {'Loss':>8} | {'L1':>8} | {'FootXZ':>8} | {'LR':>10} | {'Time':>8}")
    log(f"  {'─'*5} | {'─'*8} | {'─'*8} | {'─'*8} | {'─'*10} | {'─'*8}")

    t0 = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        sum_loss, sum_l1, sum_foot = 0.0, 0.0, 0.0
        ep_t0 = time.time()

        for distorted, target in loader:
            distorted, target = distorted.to(DEVICE), target.to(DEVICE)
            pred = model(distorted, foot_only=False, root_relative=False)
            loss, l1, foot = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_l1 += l1.item()
            sum_foot += foot.item()

        scheduler.step()
        n = len(loader)
        avg_loss = sum_loss / n
        avg_l1 = sum_l1 / n
        avg_foot = sum_foot / n
        lr = optimizer.param_groups[0]['lr']
        ep_time = time.time() - ep_t0

        log(f"  {epoch+1:>5} | {avg_loss:>8.4f} | {avg_l1:>8.4f} | "
            f"{avg_foot:>8.4f} | {lr:>10.6f} | {format_time(ep_time):>8}")

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss,
        }, f"{SAVE_DIR}/latest.pth")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss,
            }, f"{SAVE_DIR}/best.pth")

    elapsed = time.time() - t0
    log(f"")
    log("=" * 70)
    log(f"  ✅ Training Complete — Best loss: {best_loss:.4f} — {format_time(elapsed)}")
    log(f"  Checkpoint: {SAVE_DIR}/best.pth")
    log("=" * 70)


if __name__ == "__main__":
    main()
