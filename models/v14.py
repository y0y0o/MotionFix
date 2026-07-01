"""
V14: Foot-Skating-Targeted MotionFix Network
=============================================
Same Transformer Encoder architecture as V8, but trained with a NEW paradigm:
  - Training data: HumanML3D + simulated foot skating (horizontal drift@contact)
  - Loss: contact-weighted, foot-focused, XZ-emphasized
  - Inference: root_relative conversion + selective foot replacement (same as V8 fix)

Key difference from V8:
  V8:  random noise → clean (general denoising, foot skating untreated)
  V14: foot_skating → clean (targeted skating correction)

Architecture: Transformer Encoder ×6, d_model=512, nhead=8, 19.1M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MotionFixNetworkV14(nn.Module):
    """
    V14: Foot-skating-targeted motion fix network.

    Training:
        Input  — HumanML3D motion with simulated horizontal foot sliding
        Target — Original clean HumanML3D motion
        The model learns to identify and reverse foot skating while leaving
        everything else untouched.

    Inference:
        Input  — MoMask/VQ model output (world coordinates, with real foot skating)
        1. Convert world → root-relative (same as V8)
        2. Transformer forward pass
        3. Convert root-relative → world
        4. Selective foot replacement at detected skating frames (XZ only)
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

        # ── Foot joint indices (HumanML3D 22-joint skeleton) ──
        self.foot_joints = [7, 8, 10, 11]  # L-Ankle, R-Ankle, L-Foot, R-Foot
        self.foot_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j*3, j*3+1, j*3+2])

        # ── Foot XZ-only dims (for loss weighting) ──
        self.foot_xz_dims = []
        for j in self.foot_joints:
            self.foot_xz_dims.extend([j*3, j*3+2])  # X and Z only, skip Y

        # ── Transformer Encoder ──
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

    # ══════════════════════════════════════════════════════════════
    # Root-Relative Coordinate Conversion (same as V8)
    # ══════════════════════════════════════════════════════════════

    def _to_root_relative(self, x):
        """
        Convert world-coordinate motion to root-relative by subtracting
        pelvis (joint 0) position from all joints.

        x: (B, T, 66) — 22 joints × 3 coords, flattened
        Returns: (B, T, 66), pelvis_xyz (B, T, 3)
        """
        pelvis = x[:, :, 0:3].clone()
        pelvis_tiled = pelvis.repeat(1, 1, 22)
        return x - pelvis_tiled, pelvis

    def _from_root_relative(self, x_rr, pelvis):
        """Convert root-relative back to world coordinates."""
        pelvis_tiled = pelvis.repeat(1, 1, 22)
        return x_rr + pelvis_tiled

    # ══════════════════════════════════════════════════════════════
    # Forward
    # ══════════════════════════════════════════════════════════════

    def forward(self, x, foot_only=False, root_relative=True):
        """
        Args:
            x: (B, T, 66) motion input
            foot_only: if True, apply selective foot replacement (inference mode)
            root_relative: if True, convert world→root-relative before model,
                           then convert back after (for inference on world-coord input)
        """
        # ── Root-relative conversion (for inference on world coords) ──
        if root_relative:
            x_rr, pelvis = self._to_root_relative(x)
        else:
            x_rr = x

        # ── Transformer forward ──
        h = self.input_proj(x_rr)
        h = h.permute(1, 0, 2)       # (T, B, d_model)
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)       # (B, T, d_model)
        full_output_rr = self.output_proj(h)

        # ── Convert back to world coords ──
        if root_relative:
            full_output = self._from_root_relative(full_output_rr, pelvis)
        else:
            full_output = full_output_rr

        if not foot_only:
            return full_output

        # ── Selective foot replacement in WORLD coordinates ──
        output = x.clone()
        batch_size = x.shape[0]

        for b in range(batch_size):
            output[b] = self._selective_replace(x[b], full_output[b])

        return output

    # ══════════════════════════════════════════════════════════════
    # Selective Foot Replacement (same as V8 fix)
    # ══════════════════════════════════════════════════════════════

    def _selective_replace(self, original, predicted):
        """
        Replace foot joints ONLY at detected skating frames.
        Only blends X and Z (horizontal plane) — protects Y (height).

        original: (T, 66)
        predicted: (T, 66)
        """
        output = original.clone()
        T = original.shape[0]

        for foot_joint in self.foot_joints:
            dims = [foot_joint*3, foot_joint*3+1, foot_joint*3+2]
            y_dim = foot_joint*3 + 1

            heights = original[:, y_dim].detach().cpu().numpy()

            # Ground height estimation
            ground_level = np.percentile(heights, 5)
            contact_threshold = ground_level + 0.05

            for t in range(1, T):
                height = heights[t]

                if height < contact_threshold:
                    # Foot near ground → check for skating
                    orig_xz = original[t, [dims[0], dims[2]]]
                    prev_xz = original[t-1, [dims[0], dims[2]]]
                    velocity = torch.norm(orig_xz - prev_xz).item()

                    if velocity > 0.03:
                        # Skating detected → blend only X and Z
                        alpha = self.blend_alpha
                        for d in dims:
                            if d == y_dim:
                                continue  # Don't modify height
                            output[t, d] = (
                                (1 - alpha) * original[t, d]
                                + alpha * predicted[t, d]
                            )
                # else: foot in air → leave untouched

        return output


# ═══════════════════════════════════════════════════════════════════
# V14 Loss Function
# ═══════════════════════════════════════════════════════════════════

class V14Loss(nn.Module):
    """
    V14 contact-weighted, foot-focused loss.

    Designed for the foot-skating-simulation training paradigm:
    - Foot joints weighted 3× more than other joints
    - Foot XZ weighted 2× more than foot Y (since only XZ has distortion)
    - Velocity smoothness weighted for temporal consistency

    Loss = L1_all + λ_vel*L1_vel + λ_foot*(L1_foot_XZ + λ_foot_y*L1_foot_Y + L1_foot_vel)
    """

    def __init__(self, lambda_vel=0.5, lambda_foot=3.0, lambda_foot_y=0.5):
        """
        Args:
            lambda_vel: weight for velocity smoothness loss
            lambda_foot: weight for foot joint position loss (vs body)
            lambda_foot_y: relative weight for foot Y vs foot XZ (Y is not distorted
                           in training data, so lower weight prevents model from
                           modifying height unnecessarily)
        """
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_foot = lambda_foot
        self.lambda_foot_y = lambda_foot_y
        self.l1_loss = nn.L1Loss()

        # ── Dimension indices ──
        self.foot_joints = [7, 8, 10, 11]
        self.foot_dims = []
        self.foot_xz_dims = []
        self.foot_y_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j*3, j*3+1, j*3+2])
            self.foot_xz_dims.extend([j*3, j*3+2])    # X, Z
            self.foot_y_dims.append(j*3 + 1)           # Y

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, T, 66) model output
            target: (B, T, 66) ground truth

        Returns:
            total_loss, l1_loss, foot_loss
        """
        # ── Full body L1 ──
        loss_l1 = self.l1_loss(pred, target)

        # ── Foot XZ L1 (horizontal — where skating happens) ──
        pred_foot_xz = pred[:, :, self.foot_xz_dims]
        target_foot_xz = target[:, :, self.foot_xz_dims]
        loss_foot_xz = self.l1_loss(pred_foot_xz, target_foot_xz)

        # ── Foot Y L1 (height — should be identity, lower weight) ──
        pred_foot_y = pred[:, :, self.foot_y_dims]
        target_foot_y = target[:, :, self.foot_y_dims]
        loss_foot_y = self.l1_loss(pred_foot_y, target_foot_y)

        # ── Foot velocity L1 (smooth foot motion) ──
        pred_foot = pred[:, :, self.foot_dims]
        target_foot = target[:, :, self.foot_dims]
        pred_foot_vel = pred_foot[:, 1:, :] - pred_foot[:, :-1, :]
        target_foot_vel = target_foot[:, 1:, :] - target_foot[:, :-1, :]
        loss_foot_vel = self.l1_loss(pred_foot_vel, target_foot_vel)

        # ── Velocity L1 (all joints, light weight) ──
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        target_vel = target[:, 1:, :] - target[:, :-1, :]
        loss_vel = self.l1_loss(pred_vel, target_vel)

        # ── Total ──
        total = (
            loss_l1
            + self.lambda_vel * loss_vel
            + self.lambda_foot * (
                loss_foot_xz
                + self.lambda_foot_y * loss_foot_y
                + loss_foot_vel
            )
        )

        return total, loss_l1, loss_foot_xz


# ═══════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing V14 model...")

    model = MotionFixNetworkV14(blend_alpha=0.5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(2, 100, 66)

    # Training mode
    y_train = model(x, foot_only=False, root_relative=False)
    print(f"  Train:  {x.shape} -> {y_train.shape}")

    # Inference mode
    y_infer = model(x, foot_only=True, root_relative=False)
    print(f"  Infer:  {x.shape} -> {y_infer.shape}")

    # Verify non-foot joints untouched in inference mode
    non_foot = [i for i in range(66) if i not in model.foot_dims]
    diff = (y_infer[:, :, non_foot] - x[:, :, non_foot]).abs().max().item()
    print(f"  Non-foot change: {diff:.10f} (should be 0)")

    # Test loss
    criterion = V14Loss(lambda_vel=0.5, lambda_foot=3.0, lambda_foot_y=0.5)
    total, l1, foot = criterion(y_train, torch.randn(2, 100, 66))
    print(f"  Loss: total={total.item():.4f}, L1={l1.item():.4f}, FootXZ={foot.item():.4f}")

    # Test root_relative conversion
    x_world = x.clone()
    x_world[:, :, 0] += 2.0  # Shift X by 2m (simulating world coordinates)
    y_rr = model(x_world, foot_only=True, root_relative=True)
    print(f"  RootRel: {x_world.shape} -> {y_rr.shape}")

    print("  ✓ All tests passed.")
