"""
V16: Self-Supervised Foot Skating Fix with Anti-Gaming Loss
===========================================================
The KEY insight from V8-V15 failures:
  L1-to-target loss is dominated by the 88% of dimensions that should
  stay identical → gradient signal for foot-XZ-at-contact is washed out
  → model degenerates to identity mapping.

V16 fix: DROP the L1-to-target paradigm entirely. Optimize the goal DIRECTLY.

  Loss = λ_skate    · soft_FSR(output)        # reduce skating (THE GOAL)
       + λ_smooth   · jitter(output)          # stay smooth (ANTI-TWITCH guard)
       + λ_anchor   · ‖foot - input‖          # foot stays near original (ANTI-FLY guard)
       + λ_preserve · ‖nonfoot/footY - input‖ # don't touch the rest (ANTI-WRECK guard)

Self-supervised: NO target needed. Trains directly on raw MoMask output.

Anti-gaming design (addresses the "metric gaming → twitching" concern):
  - soft_FSR and jitter are in TENSION: abruptly freezing a sliding foot
    creates an acceleration spike that jitter penalizes. The model is forced
    to reduce sliding GRADUALLY → natural correction, not a twitch.
  - anchor stops the foot from teleporting to a convenient low-FSR position.
  - preserve hard-pins non-foot joints and foot height.
  - Floating/BoneCV/Penetration metrics (computed at eval) catch any residual gaming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the proven V14 architecture (Transformer Encoder ×6, root-relative)
from models.v14 import MotionFixNetworkV14, PositionalEncoding

# V16 uses the exact same network as V14 — only the LOSS and TRAINING differ.
MotionFixNetworkV16 = MotionFixNetworkV14


class V16Loss(nn.Module):
    """
    Self-supervised anti-gaming loss for foot skating correction.

    Operates on world-coordinate motion (B, T, 66). No target required —
    the loss is computed entirely from the model output and the original input.
    """

    def __init__(self,
                 lambda_skate=10.0,     # reduce foot sliding (the goal)
                 lambda_smooth=5.0,     # penalize foot acceleration (anti-twitch)
                 lambda_anchor=2.0,     # keep foot near original position (anti-fly)
                 lambda_preserve=20.0,  # pin non-foot joints + foot height (anti-wreck)
                 h_thresh=0.05,         # contact height threshold (m)
                 skate_thresh=0.03,     # horizontal velocity threshold (m/frame)
                 contact_temp=0.02):    # soft-contact sigmoid temperature
        super().__init__()
        self.lambda_skate = lambda_skate
        self.lambda_smooth = lambda_smooth
        self.lambda_anchor = lambda_anchor
        self.lambda_preserve = lambda_preserve
        self.h_thresh = h_thresh
        self.skate_thresh = skate_thresh
        self.contact_temp = contact_temp

        # Joint layout (22 joints × 3, flattened to 66)
        self.ankle_joints = [7, 8]                 # for contact + skate detection
        self.foot_joints = [7, 8, 10, 11]          # all foot joints
        self.non_foot_joints = [j for j in range(22) if j not in self.foot_joints]

        # Flattened dim indices
        self.foot_xz_dims = []
        self.foot_y_dims = []
        for j in self.foot_joints:
            self.foot_xz_dims.extend([j*3, j*3+2])  # X, Z
            self.foot_y_dims.append(j*3 + 1)        # Y
        self.non_foot_dims = []
        for j in self.non_foot_joints:
            self.non_foot_dims.extend([j*3, j*3+1, j*3+2])

    def _soft_contact_weights(self, motion_jts):
        """
        Soft contact weight per ankle per frame: high when foot near ground.

        motion_jts: (B, T, 22, 3)
        Returns: (B, T, len(ankle_joints)) in [0, 1]
        """
        B, T, _, _ = motion_jts.shape
        weights = []
        for j in self.ankle_joints:
            foot_y = motion_jts[:, :, j, 1]                       # (B, T)
            # Ground level = 5th percentile over time (detached reference)
            ground = torch.quantile(foot_y.detach(), 0.05, dim=1, keepdim=True)  # (B, 1)
            # Soft contact: sigmoid rises as foot drops below (ground + h_thresh)
            w = torch.sigmoid((ground + self.h_thresh - foot_y) / self.contact_temp)
            weights.append(w)
        return torch.stack(weights, dim=-1)                      # (B, T, n_ankles)

    def forward(self, output, original):
        """
        Args:
            output:   (B, T, 66) model output (world coords)
            original: (B, T, 66) input MoMask motion (world coords)

        Returns:
            total, dict of components
        """
        B, T, D = output.shape
        out_jts = output.reshape(B, T, 22, 3)
        orig_jts = original.reshape(B, T, 22, 3)

        # ── 1. Soft FSR loss: contact-weighted excess foot velocity ──
        # Compute on ANKLES (FSR is defined on ankles 7,8)
        contact_w = self._soft_contact_weights(orig_jts)         # (B, T, 2) from ORIGINAL
        skate_loss = 0.0
        for i, j in enumerate(self.ankle_joints):
            vel_xz = out_jts[:, 1:, j, [0, 2]] - out_jts[:, :-1, j, [0, 2]]  # (B, T-1, 2)
            speed = torch.norm(vel_xz, dim=-1)                   # (B, T-1)
            # Only penalize speed ABOVE threshold (legit slow movement is free)
            excess = F.relu(speed - self.skate_thresh)           # (B, T-1)
            w = contact_w[:, 1:, i]                              # (B, T-1)
            skate_loss = skate_loss + (w * excess).sum() / (w.sum() + 1e-6)
        skate_loss = skate_loss / len(self.ankle_joints)

        # ── 2. Smoothness (jitter) loss: foot acceleration ──
        foot_out = out_jts[:, :, self.foot_joints, :]            # (B, T, 4, 3)
        vel = foot_out[:, 1:] - foot_out[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        smooth_loss = (acc ** 2).mean()

        # ── 3. Anchor loss: foot stays near original position ──
        foot_out_pos = output[:, :, self.foot_xz_dims]
        foot_orig_pos = original[:, :, self.foot_xz_dims]
        anchor_loss = (foot_out_pos - foot_orig_pos).abs().mean()

        # ── 4. Preserve loss: non-foot joints + foot Y must NOT change ──
        nonfoot_diff = (output[:, :, self.non_foot_dims]
                        - original[:, :, self.non_foot_dims]).abs().mean()
        footy_diff = (output[:, :, self.foot_y_dims]
                      - original[:, :, self.foot_y_dims]).abs().mean()
        preserve_loss = nonfoot_diff + footy_diff

        # ── Total ──
        total = (self.lambda_skate * skate_loss
                 + self.lambda_smooth * smooth_loss
                 + self.lambda_anchor * anchor_loss
                 + self.lambda_preserve * preserve_loss)

        return total, {
            'skate': skate_loss.item() if torch.is_tensor(skate_loss) else skate_loss,
            'smooth': smooth_loss.item(),
            'anchor': anchor_loss.item(),
            'preserve': preserve_loss.item(),
        }


if __name__ == "__main__":
    print("Testing V16 model + loss...")
    model = MotionFixNetworkV16(blend_alpha=0.5)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 100, 66)
    # Make feet near ground with some sliding
    x = x * 0.1
    x[:, :, [7*3+1, 8*3+1]] = 0.02  # ankles near ground

    out = model(x, foot_only=False, root_relative=False)
    print(f"  Forward: {x.shape} -> {out.shape}")

    criterion = V16Loss()
    total, comps = criterion(out, x)
    print(f"  Loss total={total.item():.4f}")
    print(f"  Components: {comps}")
    print("  ✓ V16 test passed.")
