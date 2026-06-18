"""
MotionFix V11 - Training Script

V11 改进（修复 V10 抖动问题）:
  - 重新平衡损失权重: λ_foot 2.0→0.8, λ_smooth 0.3→1.5
  - 新增 L_UpperVel: 约束上半身速度不变，防止抖动传播
  - 训练数据复用 V10 的 training_data_v10/
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from motionfix_model import MotionFixNetwork, MotionFixLoss
from dataset import MotionFixDataset

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
DATA_DIR = "../motionfix_v10/training_data_v10"
SAVE_DIR = "checkpoints_v11"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():
    print("=" * 60)
    print("MotionFix V11 - Training with rebalanced FRDM-style losses")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_DIR}")
    print(f"Checkpoints: {SAVE_DIR}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- 数据 ----
    dataset = MotionFixDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)
    print(f"Samples: {len(dataset)}, Batches/epoch: {len(loader)}")

    # ---- 模型 ----
    model = MotionFixNetwork().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 损失 & 优化器 ----
    criterion = MotionFixLoss(
        lambda_foot=0.8,
        lambda_smooth=1.5,
        lambda_vel_cons=0.1,
        lambda_upper_vel=0.5,
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # ---- 断点续训 ----
    start_epoch = 0
    resume_path = f"{SAVE_DIR}/latest.pth"
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    best_loss = float('inf')

    # ---- 训练循环 ----
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        t0 = time.time()

        total_loss = total_recon = total_foot = total_smooth = total_vel = total_upper = 0.0

        for distorted, target, contact in loader:
            distorted = distorted.to(DEVICE)
            target = target.to(DEVICE)
            contact = contact.to(DEVICE)

            # 统一 forward（无 foot_only 标志）
            pred = model(distorted)

            # V11 结构化损失（含上半身速度保持）
            loss, l_recon, l_foot, l_smooth, l_vel, l_upper = criterion(pred, target, contact)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += l_recon.item()
            total_foot += l_foot.item()
            total_smooth += l_smooth.item()
            total_vel += l_vel.item()
            total_upper += l_upper.item()

        scheduler.step()
        n = len(loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # 打印分量损失
        print(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
            f"Loss: {total_loss/n:.4f} = "
            f"R:{total_recon/n:.4f} + "
            f"F:{total_foot/n:.4f} + "
            f"S:{total_smooth/n:.4f} + "
            f"V:{total_vel/n:.4f} + "
            f"U:{total_upper/n:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        # 保存
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / n,
        }, f"{SAVE_DIR}/latest.pth")

        if total_loss / n < best_loss:
            best_loss = total_loss / n
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"{SAVE_DIR}/best.pth")
            print(f"  -> Best model saved (loss={best_loss:.4f})")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Model: {SAVE_DIR}/best.pth")


if __name__ == "__main__":
    train()
