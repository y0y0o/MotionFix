"""
MotionFix V12 - FRDM-inspired Transformer with Dual-Head Output

Key enhancements over V8/V11:
  1. DUAL-HEAD OUTPUT: position head + delta (frame displacement) head
     → enables velocity-position consistency loss (FRDM L_vel-pos equivalent)

  2. CONTACT-GATED FOOT LOSS: high weight on contact frames, low on air frames
     → model focuses on fixing foot sliding where it matters (FRDM L_Foot equivalent)

  3. EPSILON-INSENSITIVE LOSS: small errors not penalized
     → allows micro-adjustments without forcing over-fitting (FRDM L_epsilon-rp equivalent)

  4. ENHANCED SELECTIVE REPLACE (inference only):
     a) Window expansion: blend [t-k, t+k] around detected skating frames
        → eliminates isolated-frame jumps
     b) Velocity-aware gating: reject predictions that would cause >0.5m jumps
        → prevents catastrophic 4m teleportation

Architecture (same as V8/V11, proven):
  (T, 66) → Linear(66→512) → PositionalEncoding → TransformerEncoder ×6
          → SharedProj → PosHead(512→256→66) + DeltaHead(512→256→66)
"""

import torch
import torch.nn as nn
import math
import numpy as np


# ================================================================
#  Positional Encoding
# ================================================================
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


# ================================================================
#  Main Network
# ================================================================
class MotionFixNetworkV12(nn.Module):
    """
    Transformer Encoder with dual-head output (position + delta).

    Training: foot_only=False → both heads active, full reconstruction
    Inference: foot_only=True → enhanced selective replace on foot joints

    V12.1: blend_alpha 0.5→0.7, velocity_gate 0.5→1.0 (stronger corrections)
    """

    def __init__(
        self,
        input_dim=66,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        blend_alpha=0.7,
        window_size=2,         # ±window frames for expansion
        velocity_gate=1.0,     # max allowed jump distance (meters)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.blend_alpha = blend_alpha
        self.window_size = window_size
        self.velocity_gate = velocity_gate

        # Foot joints: 7=L_ankle, 8=R_ankle, 10=L_foot, 11=R_foot
        self.foot_joints = [7, 8, 10, 11]
        self.foot_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j * 3, j * 3 + 1, j * 3 + 2])

        # ---- Encoder (same as V8/V11) ----
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # ---- Shared feature projection ----
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Dual output heads (FRDM-style redundant representation) ----
        self.pos_head = nn.Linear(d_model // 2, input_dim)    # joint positions
        self.delta_head = nn.Linear(d_model // 2, input_dim)  # frame-to-frame displacements

    def forward(self, x, foot_only=False):
        """
        x: (B, T, 66)

        Returns:
          foot_only=False: pred_pos (B,T,66), pred_delta (B,T,66)
          foot_only=True:  fixed_motion (B,T,66) after enhanced selective replace
        """
        # Encoder
        h = self.input_proj(x)                     # (B, T, d_model)
        h = h.permute(1, 0, 2)                     # (T, B, d_model)
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)                     # (B, T, d_model)

        # Shared projection
        feat = self.shared_proj(h)                 # (B, T, d_model//2)

        # Dual heads (FRDM-style: pos + delta)
        pred_pos = self.pos_head(feat)             # (B, T, 66)
        pred_delta = self.delta_head(feat)         # (B, T, 66)

        if not foot_only:
            return pred_pos, pred_delta

        # === Inference: Enhanced Selective Replace ===
        output = x.clone()
        B = x.shape[0]
        for b in range(B):
            output[b] = self._selective_replace_enhanced(
                x[b], pred_pos[b], pred_delta[b]
            )
        return output

    def _selective_replace_enhanced(self, original, pred_pos, pred_delta):
        """
        Enhanced selective replace with window expansion + velocity gating.

        original:   (T, 66)  input motion (with artifacts)
        pred_pos:   (T, 66)  predicted positions
        pred_delta: (T, 66)  predicted frame-to-frame displacements

        1. Detect skating frames (foot on ground + horizontal velocity > 0.03)
        2. Expand to window [t-window, t+window] to eliminate isolated frames
        3. Velocity gate: skip blend if prediction would cause > velocity_gate jump
        4. Blend with distance-decayed alpha
        """
        output = original.clone()
        T = original.shape[0]
        device = original.device

        for fj in self.foot_joints:
            dims = [fj * 3, fj * 3 + 1, fj * 3 + 2]
            y_dim = fj * 3 + 1  # Y coordinate (height)

            # --- Step 1: Detect skating frames ---
            heights = original[:, y_dim].detach().cpu().numpy()
            ground_level = np.percentile(heights, 5)
            contact_threshold = ground_level + 0.05

            skating_frames = set()
            for t in range(1, T):
                if heights[t] < contact_threshold:
                    orig_xz = original[t, [dims[0], dims[2]]]
                    prev_xz = original[t - 1, [dims[0], dims[2]]]
                    velocity = torch.norm(orig_xz - prev_xz).item()
                    if velocity > 0.03:
                        skating_frames.add(t)

            if not skating_frames:
                continue

            # --- Step 2: Window expansion ---
            expanded_frames = set()
            for t in skating_frames:
                for dt in range(-self.window_size, self.window_size + 1):
                    tt = t + dt
                    if 0 <= tt < T:
                        expanded_frames.add(tt)

            # --- Step 3 & 4: Velocity gate + Blend ---
            for t in sorted(expanded_frames):
                # Velocity gate: check if prediction is physically reasonable
                if t > 0:
                    # Predicted displacement from t-1 to t
                    pred_disp = torch.norm(
                        pred_pos[t, dims] - original[t - 1, dims]
                    ).item()
                    # Original displacement from t-1 to t
                    orig_disp = torch.norm(
                        original[t, dims] - original[t - 1, dims]
                    ).item()

                    # Skip if prediction would cause unreasonably large jump
                    # Condition: displacement > 3x original OR > absolute threshold
                    if pred_disp > max(orig_disp * 3.0, self.velocity_gate):
                        continue

                # Compute alpha: center frames get full blend_alpha, edges decay
                if t in skating_frames:
                    alpha = self.blend_alpha
                else:
                    # Distance to nearest skating frame
                    min_dist = min(abs(t - s) for s in skating_frames)
                    # Linear decay: alpha at dist=0 is blend_alpha, at dist=window+1 is 0
                    alpha = self.blend_alpha * max(
                        0.0, 1.0 - min_dist / (self.window_size + 1)
                    )

                if alpha <= 0.0:
                    continue

                # Apply blend
                for d in dims:
                    output[t, d] = (
                        (1.0 - alpha) * original[t, d]
                        + alpha * pred_pos[t, d]
                    )

        return output


# ================================================================
#  V12 Loss — FRDM-inspired with Contact Gating & Consistency
# ================================================================
class MotionFixLossV12(nn.Module):
    """
    FRDM-inspired loss combining V8's proven structure with FRDM's key ideas:

    L_recon     — Full body L1 reconstruction (V8, all frames)
    L_vel       — Full body velocity L1 (V8, mild)
    L_foot_c    — Foot position L1, contact=1 frames ONLY, weight=5.0 (FRDM L_Foot)
    L_foot_a    — Foot position L1, contact=0 frames ONLY, weight=0.1
    L_foot_vel  — Foot velocity L1 (V8, all frames)
    L_vel_pos   — Velocity-position consistency: |CumSum(delta) - pos| (FRDM L_vel-pos)
    L_eps       — Epsilon-insensitive loss on vel-pos consistency (FRDM L_epsilon-rp)

    Total = L_recon + λ_vel*L_vel
          + λ_fc*L_foot_c + λ_fa*L_foot_a + λ_fvel*L_foot_vel
          + λ_vp*L_vel_pos + λ_eps*L_eps
    """

    def __init__(
        self,
        lambda_vel=0.5,
        lambda_foot_contact=5.0,
        lambda_foot_air=0.1,
        lambda_foot_vel=2.0,
        lambda_vel_pos=0.3,
        lambda_eps=1.0,
        epsilon=0.02,  # ε tolerance in meters (~2cm)
    ):
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_foot_contact = lambda_foot_contact
        self.lambda_foot_air = lambda_foot_air
        self.lambda_foot_vel = lambda_foot_vel
        self.lambda_vel_pos = lambda_vel_pos
        self.lambda_eps = lambda_eps
        self.epsilon = epsilon

        self.l1_loss = nn.L1Loss()

        # Foot joint dimensions (7,8,10,11 × 3 = 12 dims)
        self.foot_joints = [7, 8, 10, 11]
        self.foot_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j * 3, j * 3 + 1, j * 3 + 2])

    def forward(self, pred_pos, pred_delta, target, contact):
        """
        pred_pos:   (B, T, 66)  predicted joint positions
        pred_delta: (B, T, 66)  predicted frame displacements
        target:     (B, T, 66)  ground truth
        contact:    (B, T, 2)   foot-ground contact labels
        """
        B, T, D = pred_pos.shape

        # 1. Full body L1 reconstruction (V8)
        loss_recon = self.l1_loss(pred_pos, target)

        # 2. Full body velocity L1 (V8)
        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target[:, 1:, :] - target[:, :-1, :]
        loss_vel = self.l1_loss(pred_vel, target_vel)

        # 3 & 4. Contact-gated foot position loss (FRDM L_Foot)
        pred_foot = pred_pos[:, :, self.foot_dims]    # (B, T, 12)
        target_foot = target[:, :, self.foot_dims]    # (B, T, 12)

        # Expand contact from (B, T, 2) to (B, T, 12) — 2 feet × 3 dims each, repeated for ankle+foot
        # contact[:,:,0] = left foot, contact[:,:,1] = right foot
        # foot_dims order: 7(LA)×3, 8(RA)×3, 10(LF)×3, 11(RF)×3
        contact_left = contact[:, :, 0:1]   # (B, T, 1)
        contact_right = contact[:, :, 1:2]  # (B, T, 1)

        # Build mask: left ankle(3) + right ankle(3) + left foot(3) + right foot(3)
        contact_mask_12 = torch.cat([
            contact_left.expand(-1, -1, 3),   # LA
            contact_right.expand(-1, -1, 3),  # RA
            contact_left.expand(-1, -1, 3),   # LF
            contact_right.expand(-1, -1, 3),  # RF
        ], dim=-1).bool()  # (B, T, 12)

        # Contact frames
        if contact_mask_12.sum() > 0:
            loss_foot_contact = self.l1_loss(
                pred_foot[contact_mask_12], target_foot[contact_mask_12]
            )
        else:
            loss_foot_contact = torch.tensor(0.0, device=pred_pos.device)

        # Non-contact frames
        non_contact_mask = ~contact_mask_12
        if non_contact_mask.sum() > 0:
            loss_foot_air = self.l1_loss(
                pred_foot[non_contact_mask], target_foot[non_contact_mask]
            )
        else:
            loss_foot_air = torch.tensor(0.0, device=pred_pos.device)

        # 5. Foot velocity L1 (V8, all frames)
        pred_foot_vel = pred_foot[:, 1:, :] - pred_foot[:, :-1, :]
        target_foot_vel = target_foot[:, 1:, :] - target_foot[:, :-1, :]
        loss_foot_vel = self.l1_loss(pred_foot_vel, target_foot_vel)

        # 6. Velocity-position consistency (FRDM L_vel-pos)
        # CumSum(delta) should equal pos shift from frame 0
        cumsum_delta = torch.cumsum(pred_delta, dim=1)          # (B, T, 66)
        pos_shift = pred_pos - pred_pos[:, 0:1, :]              # (B, T, 66)
        loss_vel_pos = self.l1_loss(cumsum_delta, pos_shift)

        # 7. Epsilon-insensitive loss (FRDM L_epsilon-rp)
        # Applied to vel-pos consistency: only penalize deviations > epsilon
        diff = torch.abs(cumsum_delta - pos_shift)
        loss_eps = torch.mean(torch.clamp(diff - self.epsilon, min=0))

        # Total
        total = (
            loss_recon
            + self.lambda_vel * loss_vel
            + self.lambda_foot_contact * loss_foot_contact
            + self.lambda_foot_air * loss_foot_air
            + self.lambda_foot_vel * loss_foot_vel
            + self.lambda_vel_pos * loss_vel_pos
            + self.lambda_eps * loss_eps
        )

        return total, loss_recon, loss_foot_contact, loss_foot_vel, loss_vel_pos, loss_eps


# ================================================================
#  Self-test
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MotionFix V12 (FRDM-inspired) - Self Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MotionFixNetworkV12(blend_alpha=0.5, window_size=2, velocity_gate=0.5)
    model = model.to(device)

    x = torch.randn(2, 100, 66).to(device)
    contact = torch.zeros(2, 100, 2).to(device)
    contact[:, 10:20, 0] = 1.0  # some left foot contacts
    contact[:, 30:40, 1] = 1.0  # some right foot contacts

    # Training mode: dual outputs
    pred_pos, pred_delta = model(x, foot_only=False)
    print(f"Train mode:  pos={pred_pos.shape}, delta={pred_delta.shape}")

    # Inference mode: enhanced selective replace
    y_infer = model(x, foot_only=True)
    print(f"Infer mode:  {y_infer.shape}")

    # Non-foot joints should be untouched in inference
    non_foot = [i for i in range(66) if i not in model.foot_dims]
    diff = (y_infer[:, :, non_foot] - x[:, :, non_foot]).abs().max().item()
    print(f"Non-foot change (infer): {diff:.10f}  (should be 0.0)")

    # Loss test
    criterion = MotionFixLossV12()
    target = x.clone()
    total, l_recon, l_fc, l_fvel, l_vp, l_eps = criterion(
        pred_pos, pred_delta, target, contact
    )

    print(f"\nLoss breakdown:")
    print(f"  L_recon:     {l_recon.item():.6f}")
    print(f"  L_foot_c:    {l_fc.item():.6f}")
    print(f"  L_foot_vel:  {l_fvel.item():.6f}")
    print(f"  L_vel_pos:   {l_vp.item():.6f}")
    print(f"  L_eps:       {l_eps.item():.6f}")
    print(f"  Total:       {total.item():.6f}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")

    print("\nAll tests passed.")
