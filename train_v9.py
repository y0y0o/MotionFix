"""
MotionFix V9 训练脚本

改进点 vs V8:
  1. 软门控 + 时间平滑: 用连续 blend weight 替代硬二值切换
  2. 骨骼链感知: 训练时加 bone-length consistency loss
  3. 物理约束: 着地时脚的速度应为零 (contact velocity loss)

用法:
  python train_v9.py                # 从头训练
  python train_v9.py --resume       # 从 latest checkpoint 恢复
  python train_v9.py --from-v8      # 从 V8 checkpoint 初始化（迁移学习）
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
import argparse

from motionfix_model_v9 import MotionFixNetworkV9, MotionFixLossV9
from dataset import MotionFixDataset


# ============================================================
# 超参数
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
DATA_DIR = "training_data_v2"
SAVE_DIR = "checkpoints_v9"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# V9 特有超参数
TEMPERATURE = 0.01         # sigmoid 温度（越小越陡，更接近硬门控）
SMOOTH_KERNEL_SIZE = 5     # 时间平滑高斯核大小
BLEND_ALPHA = 0.5          # 最大混合权重
LAMBDA_VEL = 0.5           # 速度损失权重
LAMBDA_FOOT = 2.0          # 脚部损失权重
LAMBDA_BONE = 1.0          # 骨骼长度损失权重


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='从 latest checkpoint 恢复')
    parser.add_argument('--from-v8', type=str, default=None,
                        help='从 V8 checkpoint 初始化，例如: --from-v8 checkpoints_v8/best.pth')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR)
    return parser.parse_args()


def build_model(device):
    """构建 V9 模型"""
    model = MotionFixNetworkV9(
        blend_alpha=BLEND_ALPHA,
        temperature=TEMPERATURE,
        smooth_kernel_size=SMOOTH_KERNEL_SIZE,
    ).to(device)
    return model


def load_v8_weights(model, v8_path, device):
    """
    从 V8 checkpoint 迁移权重到 V9
    匹配的层直接加载，V9 新增的层（gate_network, temporal_smooth）随机初始化
    """
    print(f"Loading V8 weights from {v8_path}...")
    v8_ckpt = torch.load(v8_path, map_location=device)
    v8_state = v8_ckpt['model_state_dict']

    v9_state = model.state_dict()

    # 逐层匹配
    loaded = 0
    skipped = 0
    for name, param in v9_state.items():
        if name in v8_state and v8_state[name].shape == param.shape:
            v9_state[name] = v8_state[name]
            loaded += 1
        else:
            skipped += 1
            if name in v8_state:
                print(f"  Shape mismatch: {name} V8:{v8_state[name].shape} V9:{param.shape}")
            else:
                print(f"  New layer (random init): {name}")

    model.load_state_dict(v9_state)
    print(f"  Loaded: {loaded} layers, Skipped (random init): {skipped} layers")
    return model


def train():
    args = parse_args()

    print("=" * 60)
    print("MotionFix V9 Training")
    print("=" * 60)
    print(f"Device:      {DEVICE}")
    print(f"Data:        {args.data_dir}")
    print(f"Save:        {args.save_dir}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.lr}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Smooth kern: {SMOOTH_KERNEL_SIZE}")
    print("=" * 60)

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- 数据 ----
    dataset = MotionFixDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2, pin_memory=True)
    print(f"Training pairs: {len(dataset)}, batches/epoch: {len(loader)}")

    # ---- 模型 ----
    model = build_model(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 损失函数 ----
    criterion = MotionFixLossV9(
        lambda_vel=LAMBDA_VEL,
        lambda_foot=LAMBDA_FOOT,
        lambda_bone=LAMBDA_BONE,
    )

    # ---- 优化器 ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # ---- 断点续训 ----
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        resume_path = f"{args.save_dir}/latest.pth"
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt.get('best_loss', best_loss)
            print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
        else:
            print(f"No checkpoint found at {resume_path}, starting fresh.")

    # ---- V8 迁移 ----
    if args.from_v8 and start_epoch == 0:
        model = load_v8_weights(model, args.from_v8, DEVICE)
        # 迁移后重建优化器（新层需要自己的 optimizer state）
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- 训练循环 ----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()

        # 累积统计
        sum_loss = 0.0
        sum_l1 = 0.0
        sum_foot = 0.0
        sum_bone = 0.0
        sum_contact = 0.0

        for distorted, target in loader:
            distorted = distorted.to(DEVICE)
            target = target.to(DEVICE)

            # V9 训练: foot_only=False 也走软门控混合路径（训练/推理一致）
            pred = model(distorted, foot_only=False)

            # V9 损失: 需要传入 original (distorted) 用于接触检测
            loss, l1, foot, bone, contact = criterion(pred, target, original=distorted)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            sum_loss += loss.item()
            sum_l1 += l1.item()
            sum_foot += foot.item()
            sum_bone += bone.item()
            sum_contact += contact.item()

        scheduler.step()

        n = len(loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        avg_loss = sum_loss / n

        # 打印
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"L1: {sum_l1/n:.4f} | Foot: {sum_foot/n:.4f} | "
              f"Bone: {sum_bone/n:.4f} | Contact: {sum_contact/n:.4f} | "
              f"LR: {lr:.6f} | {elapsed:.0f}s")

        # 保存 latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'args': vars(args),
        }, f"{args.save_dir}/latest.pth")

        # 保存 best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args),
            }, f"{args.save_dir}/best.pth")
            print(f"  ✓ New best model saved (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {args.save_dir}/")


if __name__ == "__main__":
    train()
