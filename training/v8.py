import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from models.v8 import MotionFixNetwork, MotionFixLoss
from data.datasets.v8 import MotionFixDataset

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
DATA_DIR = "data/training/v2"
SAVE_DIR = "checkpoints/v8"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    print("=" * 60)
    print("MotionFix V8 (selective foot replacement)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = MotionFixDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Batches per epoch: {len(loader)}")

    # 训练时用全部重建（和V3一样）
    model = MotionFixNetwork(blend_alpha=0.5).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = MotionFixLoss(lambda_vel=0.5, lambda_foot=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    start_epoch = 0
    resume_path = f"{SAVE_DIR}/latest.pth"
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    best_loss = float('inf')

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        t0 = time.time()
        sum_loss = sum_l1 = sum_foot = 0

        for distorted, target in loader:
            distorted = distorted.to(DEVICE)
            target = target.to(DEVICE)

            # 训练模式：全部重建
            pred = model(distorted, foot_only=False)
            loss, l1, foot = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_l1 += l1.item()
            sum_foot += foot.item()

        scheduler.step()
        n = len(loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Loss: {sum_loss/n:.4f} (L1: {sum_l1/n:.4f}, Foot: {sum_foot/n:.4f}) | "
              f"LR: {lr:.6f} | Time: {elapsed:.1f}s")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': sum_loss/n,
        }, f"{SAVE_DIR}/latest.pth")

        if sum_loss/n < best_loss:
            best_loss = sum_loss/n
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"{SAVE_DIR}/best.pth")

    print(f"\nDone. Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()
