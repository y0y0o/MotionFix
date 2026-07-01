"""
MotionFix V14 — Training Script
================================
Foot-skating-targeted training with simulated horizontal sliding data.

Training paradigm:
  Input  — HumanML3D + simulated foot skating (horizontal drift at contact frames)
  Target — Clean HumanML3D (unchanged)
  Model  — Transformer Encoder ×6 (same as V8)
  Loss   — Contact-weighted, foot-XZ-focused V14Loss
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
import numpy as np
from datetime import datetime

# Ensure project root is on path (script is in training/ subdir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.v14 import MotionFixNetworkV14, V14Loss

# Use packed dataset for fast loading (no per-file I/O)
from torch.utils.data import TensorDataset

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.0001
DATA_DIR = "data/training/v14"
SAVE_DIR = "checkpoints/v14"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss weights (V14-specific: foot-XZ focused)
LAMBDA_VEL = 0.5       # velocity smoothness weight
LAMBDA_FOOT = 3.0      # foot joint weight (3× body)
LAMBDA_FOOT_Y = 0.5    # foot Y weight relative to foot XZ (Y is not distorted)

LOG_PATH = "logs/v14_train.log"


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def log(msg: str, also_print: bool = True):
    """Write timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line)


def format_time(seconds: float) -> str:
    """Format seconds into H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # ── Log header ──
    log("=" * 70)
    log("  MotionFix V14 — Training Start")
    log("=" * 70)
    log(f"  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Device:     {DEVICE}")
    if DEVICE == 'cuda':
        log(f"  GPU:        {torch.cuda.get_device_name(0)}")
        log(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log(f"")
    log(f"  ── Training Config ──")
    log(f"  Data dir:   {DATA_DIR}")
    log(f"  Save dir:   {SAVE_DIR}")
    log(f"  Batch size: {BATCH_SIZE}")
    log(f"  Epochs:     {NUM_EPOCHS}")
    log(f"  LR:         {LEARNING_RATE}")
    log(f"  λ_vel:      {LAMBDA_VEL}")
    log(f"  λ_foot:     {LAMBDA_FOOT}")
    log(f"  λ_foot_y:   {LAMBDA_FOOT_Y}")
    log(f"")

    # ── Dataset (packed single-file for fast I/O) ──
    PACKED_PATH = "data/training/v14_packed.pt"
    log(f"  Loading packed dataset from {PACKED_PATH} ...")
    packed = torch.load(PACKED_PATH, map_location='cpu')
    distorted_all = packed['distorted']  # (N, 196, 66)
    target_all = packed['target']        # (N, 196, 66)
    n_samples = distorted_all.shape[0]
    log(f"  Packed samples: {n_samples}")
    log(f"  Shape: {distorted_all.shape}")
    log(f"")

    # Sanity check
    d_sample = distorted_all[0].reshape(-1, 22, 3).numpy()
    t_sample = target_all[0].reshape(-1, 22, 3).numpy()
    diff = np.abs(d_sample - t_sample)
    foot_diff = diff[:, [7,8,10,11], :]
    nf_joints = [j for j in range(22) if j not in [7,8,10,11]]
    nf_diff = diff[:, nf_joints, :]
    log(f"  Data sanity:")
    log(f"    Foot mean diff:   {foot_diff.mean():.6f} m")
    log(f"    Foot max diff:    {foot_diff.max():.4f} m")
    log(f"    Non-foot max diff: {nf_diff.max():.10f} m (should be 0)")
    log(f"")

    dataset = TensorDataset(distorted_all, target_all)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE == 'cuda'))

    # ── Model ──
    model = MotionFixNetworkV14(blend_alpha=0.5).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: MotionFixNetworkV14")
    log(f"  Architecture: Transformer Encoder ×6, d_model=512, nhead=8")
    log(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    log(f"  Blend alpha: {model.blend_alpha}")

    # ── Loss & Optimizer ──
    criterion = V14Loss(lambda_vel=LAMBDA_VEL,
                        lambda_foot=LAMBDA_FOOT,
                        lambda_foot_y=LAMBDA_FOOT_Y)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    log(f"  Optimizer: Adam, lr={LEARNING_RATE}")
    log(f"  Scheduler: StepLR, step=20, gamma=0.5")
    log(f"")

    # ── Resume ──
    start_epoch = 0
    resume_path = f"{SAVE_DIR}/latest.pth"
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        log(f"  📎 Resumed from epoch {start_epoch} (loss={ckpt['loss']:.4f})")
        log(f"")

    log("─" * 70)
    log(f"  {'Epoch':>5} | {'Loss':>8} | {'L1':>8} | {'FootXZ':>8} | "
        f"{'LR':>10} | {'Time':>8} | {'Total':>10}")
    log("─" * 70)

    # ── Training loop ──
    best_loss = float('inf')
    t_train_start = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        t0 = time.time()

        sum_loss = 0.0
        sum_l1 = 0.0
        sum_foot = 0.0

        for distorted, target in loader:
            distorted = distorted.to(DEVICE)
            target = target.to(DEVICE)

            # Training: full output (all joints)
            pred = model(distorted, foot_only=False, root_relative=False)

            loss, l1, foot_xz = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_l1 += l1.item()
            sum_foot += foot_xz.item()

        scheduler.step()
        n_batches = len(loader)
        avg_loss = sum_loss / n_batches
        avg_l1 = sum_l1 / n_batches
        avg_foot = sum_foot / n_batches
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t0
        total_time = time.time() - t_train_start

        log(f"  {epoch+1:>5} | {avg_loss:>8.4f} | {avg_l1:>8.4f} | "
            f"{avg_foot:>8.4f} | {lr:>10.6f} | {format_time(epoch_time):>8} | "
            f"{format_time(total_time):>10}")

        # ── Save latest ──
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'loss_l1': avg_l1,
            'loss_foot_xz': avg_foot,
        }, f"{SAVE_DIR}/latest.pth")

        # ── Save best ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'loss_l1': avg_l1,
                'loss_foot_xz': avg_foot,
            }, f"{SAVE_DIR}/best.pth")

    # ── Done ──
    total_time = time.time() - t_train_start
    log("─" * 70)
    log(f"")
    log("=" * 70)
    log(f"  ✅ Training Complete")
    log(f"  Best loss:     {best_loss:.4f}")
    log(f"  Total time:    {format_time(total_time)}")
    log(f"  Checkpoint:    {SAVE_DIR}/best.pth")
    log(f"  Log:           {LOG_PATH}")
    log("=" * 70)


if __name__ == "__main__":
    train()
