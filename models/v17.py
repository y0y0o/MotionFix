"""
V17: Hybrid Physics + Learned Smoother
=======================================
The stabilized FINAL method for the dissertation.

Pipeline (two stages, both ablatable):
  Stage 1 — Physics correction (rule-based, model-agnostic):
            Cap foot horizontal velocity at contact frames → FSR 14.1%→9.5%
            Side effect: abrupt freezing creates jitter (×2.7).
  Stage 2 — Lightweight learned smoother (the trained component):
            A small 1D temporal CNN predicts a RESIDUAL on foot XZ that
            smooths the abrupt physics transitions WITHOUT regrowing skating.

Why this is defensible (vs the failed pure-learning V8/V14/V15/V16):
  - The constraint (low FSR) is enforced by physics, not learned → can't degenerate.
  - The smoother only has a small, well-posed job (reduce jitter), anchored to
    the physics output → can't game its way into smooth-but-wrong trajectories.
  - 3 components (physics / smoother / their combination) give clean ablations.

Design choices that prevent gaming (the user's core concern):
  - Residual is applied ONLY to foot XZ (8 dims); foot Y and all non-foot
    joints pass through physics unchanged → semantic content preserved by construction.
  - Fidelity loss anchors the smoother output to the physics output → keeps the FSR gain.
  - Anti-skate loss penalizes contact-frame velocity regrowth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


FOOT_JOINTS = [7, 8, 10, 11]          # L-Ankle, R-Ankle, L-Foot, R-Foot
ANKLE_JOINTS = [7, 8]                  # FSR is defined on ankles
# Flattened XZ dim indices for the 4 foot joints (8 dims)
FOOT_XZ_DIMS = []
for j in FOOT_JOINTS:
    FOOT_XZ_DIMS.extend([j*3, j*3+2])  # X, Z (skip Y)


class FootSmoother(nn.Module):
    """
    Lightweight 1D temporal CNN that predicts a residual on foot XZ trajectories.

    Input:  foot XZ trajectory (B, T, 8)  [4 joints × (X,Z)]
    Output: residual          (B, T, 8)

    The residual is added to the physics-corrected foot XZ. Small receptive
    field temporal convolutions act as a learned smoothing filter.
    """
    def __init__(self, in_dim=8, hidden=64, kernel=5, n_layers=3):
        super().__init__()
        pad = kernel // 2
        layers = []
        c_in = in_dim
        for i in range(n_layers - 1):
            layers += [nn.Conv1d(c_in, hidden, kernel, padding=pad), nn.ReLU()]
            c_in = hidden
        layers += [nn.Conv1d(c_in, in_dim, kernel, padding=pad)]
        self.net = nn.Sequential(*layers)

        # Initialize last layer to ~zero so initial output ≈ physics (identity start)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, foot_xz):
        # foot_xz: (B, T, 8) → conv expects (B, C, T)
        x = foot_xz.permute(0, 2, 1)
        res = self.net(x)
        return res.permute(0, 2, 1)      # (B, T, 8) residual

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def gaussian_smooth_traj(xz, sigma=2.0, radius=4):
    """
    Gaussian temporal smoothing of a foot trajectory (numpy).
    xz: (T, C) → smoothed (T, C). Used to build the smoother's target reference.
    """
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-(t ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    out = np.zeros_like(xz)
    padded = np.pad(xz, ((radius, radius), (0, 0)), mode='edge')
    for c in range(xz.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode='valid')
    return out


class SmootherLoss(nn.Module):
    """
    Target-guided loss for the foot smoother.

    The target is a Gaussian-smoothed version of the physics output (a clear,
    strong gradient signal that the residual-acc² term could not provide).
    The anti-skate term prevents the smoothing from regrowing contact-frame
    sliding — so the model learns: smooth in air / transitions, stay put at contact.

    L = λ_match · ‖final_xz - smoothed_target‖²     # learn to smooth
      + λ_skate · soft_skate(final_xz)              # but don't slide at contact
    """
    def __init__(self, lambda_match=1.0, lambda_skate=8.0,
                 h_thresh=0.05, skate_thresh=0.03, contact_temp=0.02):
        super().__init__()
        self.lambda_match = lambda_match
        self.lambda_skate = lambda_skate
        self.h_thresh = h_thresh
        self.skate_thresh = skate_thresh
        self.contact_temp = contact_temp

    def _contact_weights(self, foot_y, ground):
        return torch.sigmoid((ground + self.h_thresh - foot_y) / self.contact_temp)

    def forward(self, final_xz, target_xz, foot_y_orig):
        """
        Args:
            final_xz:    (B, T, 8)  smoothed foot XZ (physics + residual)
            target_xz:   (B, T, 8)  Gaussian-smoothed physics XZ (reference)
            foot_y_orig: (B, T, 4)  foot heights (for contact detection)
        """
        # ── Match: pull toward smoothed reference ──
        match_loss = ((final_xz - target_xz) ** 2).mean()

        # ── Anti-skate: contact-weighted excess velocity (ankles only) ──
        skate_loss = 0.0
        for ai in (0, 1):                       # ankle 7 → cols 0,1 ; ankle 8 → cols 2,3
            cx = ai * 2
            xz = final_xz[:, :, cx:cx+2]
            speed = torch.norm(xz[:, 1:] - xz[:, :-1], dim=-1)
            fy = foot_y_orig[:, :, ai]
            ground = torch.quantile(fy.detach(), 0.05, dim=1, keepdim=True)
            w = self._contact_weights(fy, ground)[:, 1:]
            excess = F.relu(speed - self.skate_thresh)
            skate_loss = skate_loss + (w * excess).sum() / (w.sum() + 1e-6)
        skate_loss = skate_loss / 2.0

        total = self.lambda_match * match_loss + self.lambda_skate * skate_loss
        return total, {
            'match': match_loss.item(),
            'skate': skate_loss.item() if torch.is_tensor(skate_loss) else skate_loss,
        }


# ═══════════════════════════════════════════════════════════════════
# Inference helper: full hybrid pipeline
# ═══════════════════════════════════════════════════════════════════

def hybrid_fix(motion_world, smoother, device, physics_fn, damp_factor=0.0):
    """
    Full V17 pipeline on a single motion.

    Args:
        motion_world: (T, 22, 3) raw generator output
        smoother:     trained FootSmoother
        physics_fn:   physics_foot_fix function
        damp_factor:  physics damping (0.0 = freeze)

    Returns:
        final: (T, 22, 3) physics-corrected + smoothed
        physics: (T, 22, 3) physics-only (for ablation)
    """
    T = motion_world.shape[0]

    # Stage 1: physics
    physics, _ = physics_fn(motion_world, damp_factor=damp_factor, return_stats=True)

    # Stage 2: smoother on foot XZ
    phys_flat = physics.reshape(T, -1).astype(np.float32)
    foot_xz = phys_flat[:, FOOT_XZ_DIMS]                       # (T, 8)
    foot_xz_t = torch.from_numpy(foot_xz).unsqueeze(0).to(device)

    smoother.eval()
    with torch.no_grad():
        residual = smoother(foot_xz_t).squeeze(0).cpu().numpy()  # (T, 8)

    final_flat = phys_flat.copy()
    final_flat[:, FOOT_XZ_DIMS] = foot_xz + residual          # add residual to XZ
    final = final_flat.reshape(T, 22, 3)

    return final, physics


if __name__ == "__main__":
    print("Testing V17 components...")
    sm = FootSmoother()
    print(f"  FootSmoother params: {sm.n_params():,}")
    x = torch.randn(2, 100, 8)
    res = sm(x)
    print(f"  Smoother: {x.shape} -> {res.shape}")
    print(f"  Initial residual max (should be ~0): {res.abs().max().item():.2e}")

    loss = SmootherLoss()
    final_xz = torch.randn(2, 100, 8) * 0.1
    phys_xz = torch.randn(2, 100, 8) * 0.1
    fy = torch.rand(2, 100, 4) * 0.1
    total, comps = loss(final_xz, phys_xz, fy)
    print(f"  Loss: {total.item():.4f}, comps: {comps}")
    print("  ✓ V17 test passed.")
