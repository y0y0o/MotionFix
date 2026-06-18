# test_v5.py

import torch
import torch.nn as nn
import numpy as np
import glob
import os
import math


# V5的网络结构（和训练时一致）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :].unsqueeze(1)


class MotionFixV5(nn.Module):
    """V5的网络：直接预测脚部位置，非脚部不变"""
    def __init__(self, input_dim=66, d_model=512, nhead=8,
                 num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.foot_joints = [7, 8, 10, 11]
        self.foot_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j*3, j*3+1, j*3+2])

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # V5用的是foot_predictor，不是output_proj
        self.foot_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, len(self.foot_dims)),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = h.permute(1, 0, 2)
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)

        foot_pred = self.foot_predictor(h)

        output = x.clone()
        for i, dim in enumerate(self.foot_dims):
            output[:, :, dim] = foot_pred[:, :, i]

        return output


def detect_skating(motion):
    skating = 0
    contact = 0
    for foot_idx in [10, 11]:
        foot = motion[:, foot_idx, :]
        heights = foot[:, 1]
        ground = np.percentile(heights, 5)
        threshold = ground + 0.05
        for t in range(len(motion) - 1):
            if heights[t] < threshold:
                contact += 1
                vel = np.linalg.norm(foot[t+1, [0,2]] - foot[t, [0,2]])
                if vel > 0.03:
                    skating += 1
    return skating / contact if contact > 0 else 0


def fix_motion(model, motion, device='cuda'):
    T = motion.shape[0]
    motion_flat = motion.reshape(T, -1)
    motion_tensor = torch.FloatTensor(motion_flat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        fixed_tensor = model(motion_tensor)
    return fixed_tensor.squeeze(0).cpu().numpy().reshape(T, 22, 3)


def main():
    print("=" * 60)
    print("MotionFix V5 - Test (foot direct prediction)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 用V5的网络结构加载V5的checkpoint
    model = MotionFixV5().to(device)
    ckpt = torch.load("checkpoints_v5/best.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f}")

    momask_dir = os.path.expanduser("./momask_results")
    momask_files = sorted(glob.glob(f"{momask_dir}/momask_*_no_ik.npy"))

    output_dir = "fixed_outputs_v5"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nFound {len(momask_files)} MoMask files")
    print(f"{'Test':<50} | {'Before':>7} | {'After':>7} | {'Change':>8}")
    print("-" * 85)

    results = []
    for filepath in momask_files:
        filename = os.path.basename(filepath)
        name = filename.replace('momask_', '').replace('_no_ik.npy', '')

        data = np.load(filepath)
        motion = data[0] if len(data.shape) == 4 else data

        sr_before = detect_skating(motion)
        fixed = fix_motion(model, motion, device)
        sr_after = detect_skating(fixed)

        change = sr_after - sr_before
        print(f"{name[:50]:<50} | {sr_before:>6.1%} | {sr_after:>6.1%} | {change:>+7.1%}")

        np.save(f"{output_dir}/{filename}", motion)
        np.save(f"{output_dir}/{filename.replace('_no_ik.npy', '_fixed.npy')}", fixed)
        results.append({'name': name, 'before': sr_before, 'after': sr_after})

    if results:
        avg_b = np.mean([r['before'] for r in results])
        avg_a = np.mean([r['after'] for r in results])
        print("-" * 85)
        print(f"{'Average':<50} | {avg_b:>6.1%} | {avg_a:>6.1%} | {avg_a-avg_b:>+7.1%}")
        print("=" * 85)

    print(f"\n✓ Saved to: {output_dir}/")

if __name__ == "__main__":
    main()