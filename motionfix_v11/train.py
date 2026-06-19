"""
MotionFix V11 (Fixed) - Training Script

修复要点:
  - 训练: foot_only=False, 全量重建
  - 损失: V8 直接脚部监督 + FRDM 接触门控辅助
  - 损失权重: 修正力 >> 保守力
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
    print("MotionFix V11 (Fixed) - Training")
    print("  V8 selective replace + rebalanced FRDM losses")
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
    model = MotionFixNetwork(blend_alpha=0.5).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 损失 & 优化器 ----
    # 修正力: λ_foot=2.0, λ_foot_vel=1.0 (直接脚部监督)
    # 保守力: λ_vel=0.3 (温和速度匹配), λ_foot_ct=0.5 (接触门控辅助)
    criterion = MotionFixLoss(
        lambda_foot=2.0,
        lambda_foot_vel=1.0,
        lambda_vel=0.3,
        lambda_foot_ct=0.5,
        lambda_vel_cons=0.2,
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

        total_loss = total_recon = total_foot = total_foot_ct = total_vel_cons = 0.0

        for distorted, target, contact in loader:
            distorted = distorted.to(DEVICE)
            target = target.to(DEVICE)
            contact = contact.to(DEVICE)

            # 训练模式: 全量重建 (foot_only=False)
            pred = model(distorted, foot_only=False)

            # 混合损失: V8 直接监督 + FRDM 接触门控辅助
            loss, l_recon, l_foot, l_foot_ct, l_vel_cons = criterion(
                pred, target, contact
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += l_recon.item()
            total_foot += l_foot.item()
            total_foot_ct += l_foot_ct.item()
            total_vel_cons += l_vel_cons.item()

        scheduler.step()
        n = len(loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # 打印分量损失
        print(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
            f"Loss: {total_loss/n:.4f} = "
            f"R:{total_recon/n:.4f} + "
            f"Foot:{total_foot/n:.4f} + "
            f"Ct:{total_foot_ct/n:.4f} + "
            f"Vc:{total_vel_cons/n:.4f} | "
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
