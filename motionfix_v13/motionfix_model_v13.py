"""
MotionFix V13 — V8 Architecture with Amplified Training Noise

Same proven V8 architecture:
  - Single-head Transformer (no dual-head complexity)
  - Selective foot replacement at inference
  - All-joint training (full body context)

Difference from V8: trained on 3-5x larger noise → hopefully larger corrections
"""

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


class MotionFixNetworkV13(nn.Module):
    """
    V13: V8 architecture — single-head Transformer + selective foot replacement.

    Training: full body reconstruction
    Inference: only blend foot joints at detected skating frames
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

    def forward(self, x, foot_only=False):
        h = self.input_proj(x)
        h = h.permute(1, 0, 2)
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)
        full_output = self.output_proj(h)

        if not foot_only:
            return full_output

        # Inference: selective replace (same as V8)
        output = x.clone()
        batch_size = x.shape[0]

        for b in range(batch_size):
            output[b] = self._selective_replace(x[b], full_output[b])

        return output

    def _selective_replace(self, original, predicted):
        """
        Only blend foot joints at detected skating frames.
        Non-foot joints and non-skating frames: untouched.
        """
        output = original.clone()
        T = original.shape[0]

        for foot_joint in self.foot_joints:
            dims = [foot_joint*3, foot_joint*3+1, foot_joint*3+2]
            y_dim = foot_joint*3 + 1

            heights = original[:, y_dim].detach().cpu().numpy()
            ground_level = np.percentile(heights, 5)
            contact_threshold = ground_level + 0.05

            for t in range(1, T):
                height = heights[t]

                if height < contact_threshold:
                    orig_xz = original[t, [dims[0], dims[2]]]
                    prev_xz = original[t-1, [dims[0], dims[2]]]
                    velocity = torch.norm(orig_xz - prev_xz).item()

                    if velocity > 0.03:
                        alpha = self.blend_alpha
                        for d in dims:
                            output[t, d] = (
                                (1 - alpha) * original[t, d]
                                + alpha * predicted[t, d]
                            )

        return output


class MotionFixLossV13(nn.Module):
    """V8-proven loss: L1 + velocity + foot position + foot velocity"""
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
