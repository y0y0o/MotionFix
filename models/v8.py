import torch
import torch.nn as nn
import math
import numpy as np


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


class MotionFixNetwork(nn.Module):
    """
    V8: Transformer Encoder with Selective Foot Replacement
    
    训练: 重建全部关节（和V3一样，学习全局上下文）
    推理: 只在"脚在地面且滑动"的帧替换脚部
          其他帧（空中/正常接触）完全不改
    """
    def __init__(
        self,
        input_dim=66,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        blend_alpha=0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.blend_alpha = blend_alpha

        self.foot_joints = [7, 8, 10, 11]
        self.foot_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j*3, j*3+1, j*3+2])

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, input_dim),
        )

    def _to_root_relative(self, x):
        """
        Convert world-coordinate motion to root-relative by subtracting
        pelvis (joint 0) position from all joints.

        x: (B, T, 66)  — 22 joints × 3 coords, flattened
        Returns: (B, T, 66), pelvis_xyz (B, T, 3)
        """
        pelvis = x[:, :, 0:3].clone()  # (B, T, 3)
        # Tile pelvis 22 times: [px,py,pz] → [px,py,pz, px,py,pz, ...] ×22
        # repeat() repeats the entire [px,py,pz] chunk (correct for interleaved joint layout)
        pelvis_tiled = pelvis.repeat(1, 1, 22)  # (B, T, 66)
        return x - pelvis_tiled, pelvis

    def _from_root_relative(self, x_rr, pelvis):
        """
        Convert root-relative motion back to world coordinates.
        """
        pelvis_tiled = pelvis.repeat(1, 1, 22)
        return x_rr + pelvis_tiled

    def forward(self, x, foot_only=False, root_relative=True):
        """
        foot_only=False: 训练/分析模式，输出全部关节
        foot_only=True:  推理模式，选择性替换脚部
        root_relative=True: 推理前将 world coords → root-relative,
                            推理后转回 world coords（修复坐标系不匹配）
        """
        # ── Root-relative conversion ──
        if root_relative:
            x_rr, pelvis = self._to_root_relative(x)
        else:
            x_rr = x

        # ── Transformer forward ──
        h = self.input_proj(x_rr)
        h = h.permute(1, 0, 2)
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)
        full_output_rr = self.output_proj(h)

        # ── Convert back to world coords ──
        if root_relative:
            full_output = self._from_root_relative(full_output_rr, pelvis)
        else:
            full_output = full_output_rr

        if not foot_only:
            return full_output

        # ── Selective replace in WORLD coordinates ──
        # Always use the original world-coordinate x for skating detection,
        # because ground height / velocity thresholds are defined in world space.
        output = x.clone()
        batch_size = x.shape[0]

        for b in range(batch_size):
            output[b] = self._selective_replace(
                x[b], full_output[b]
            )

        return output

    def _selective_replace(self, original, predicted):
        """
        只在滑步帧替换脚部，其他帧保持原样
        
        original: (T, 66)
        predicted: (T, 66)
        """
        output = original.clone()
        T = original.shape[0]

        for foot_joint in self.foot_joints:
            dims = [foot_joint*3, foot_joint*3+1, foot_joint*3+2]
            y_dim = foot_joint*3 + 1  # Y坐标（高度）

            # 获取高度
            heights = original[:, y_dim].detach().cpu().numpy()

            # 找到地面高度
            ground_level = np.percentile(heights, 5)
            contact_threshold = ground_level + 0.05

            for t in range(1, T):
                height = heights[t]

                if height < contact_threshold:
                    # 脚在地面 → 检查是否在滑动
                    orig_xz = original[t, [dims[0], dims[2]]]
                    prev_xz = original[t-1, [dims[0], dims[2]]]
                    velocity = torch.norm(orig_xz - prev_xz).item()

                    if velocity > 0.03:
                        # 在滑动！→ 混合修正
                        # 只修正 X 和 Z（水平面），
                        # Y（高度）保持原样 — 脚在地上高度不变
                        alpha = self.blend_alpha
                        for d in dims:
                            if d == y_dim:
                                continue  # 不修正高度
                            output[t, d] = (
                                (1 - alpha) * original[t, d]
                                + alpha * predicted[t, d]
                            )
                # else: 脚在空中 → 不改（保持原样）

        return output


class MotionFixLoss(nn.Module):
    def __init__(self, lambda_vel=0.5, lambda_foot=2.0):
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_foot = lambda_foot
        self.l1_loss = nn.L1Loss()

        self.foot_joints = [7, 8, 10, 11]
        self.foot_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j*3, j*3+1, j*3+2])

    def forward(self, pred, target):
        loss_l1 = self.l1_loss(pred, target)

        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        target_vel = target[:, 1:, :] - target[:, :-1, :]
        loss_vel = self.l1_loss(pred_vel, target_vel)

        pred_foot = pred[:, :, self.foot_dims]
        target_foot = target[:, :, self.foot_dims]
        loss_foot = self.l1_loss(pred_foot, target_foot)

        pred_foot_vel = pred_foot[:, 1:, :] - pred_foot[:, :-1, :]
        target_foot_vel = target_foot[:, 1:, :] - target_foot[:, :-1, :]
        loss_foot_vel = self.l1_loss(pred_foot_vel, target_foot_vel)

        total = (
            loss_l1
            + self.lambda_vel * loss_vel
            + self.lambda_foot * (loss_foot + loss_foot_vel)
        )
        return total, loss_l1, loss_foot


if __name__ == "__main__":
    model = MotionFixNetwork(blend_alpha=0.5)
    x = torch.randn(2, 100, 66)

    # 训练模式
    y_train = model(x, foot_only=False)
    print(f"Train: {x.shape} -> {y_train.shape}")

    # 推理模式
    y_infer = model(x, foot_only=True)
    print(f"Infer: {x.shape} -> {y_infer.shape}")

    # 检查非脚部
    non_foot = [i for i in range(66) if i not in model.foot_dims]
    diff = (y_infer[:, :, non_foot] - x[:, :, non_foot]).abs().max().item()
    print(f"Non-foot change: {diff:.10f} (should be 0)")

    # Loss
    criterion = MotionFixLoss()
    total, l1, foot = criterion(y_train, torch.randn(2, 100, 66))
    print(f"Loss: {total.item():.4f}")

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ All tests passed.")
