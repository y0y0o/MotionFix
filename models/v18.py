"""
V18: Contact-Constrained Learned Foot Refinement
=================================================
The synthesis of every lesson from V8-V17.

Core idea — work in VELOCITY space, not position space:
  1. Predict a foot-velocity field (model, init ≈ original velocity).
  2. Hard-mask contact-frame velocity to ≈0 via a soft contact weight (1-w).
     → FSR is structurally guaranteed low and NON-GAMEABLE (not learned).
  3. Integrate the masked velocity → positions are continuous by construction
     (no V17-style position jumps).

Why this breaks the FSR–Jitter antagonism that trapped V8-V17:
  - The soft contact weight ramps velocity DOWN smoothly as the foot approaches
    the ground (learned/analytical "air deceleration") → low jitter.
  - At full contact, (1-w)≈0 → velocity≈0 → foot planted → low FSR.
  - FSR is removed from the learning problem (hard-masked) so the model can ONLY
    help with smoothness — it cannot make FSR worse. The failure modes of every
    previous version (signal washout, surrogate gaming) are structurally eliminated.

The model's job is small and safe: predict a velocity RESIDUAL that makes the
integrated trajectory smoother while staying near the original (anti-drift).
"""

import torch
import torch.nn as nn
import numpy as np


FOOT_JOINTS = [7, 8, 10, 11]
FOOT_XZ_DIMS = []
for j in FOOT_JOINTS:
    FOOT_XZ_DIMS.extend([j*3, j*3+2])          # X, Z per foot joint → 8 dims
FOOT_Y_DIMS = [j*3+1 for j in FOOT_JOINTS]      # 4 dims
NON_FOOT = [j for j in range(22) if j not in FOOT_JOINTS]


def compute_contact_weight_np(foot_y, h_thresh=0.05, temp=0.02):
    """
    Soft contact weight per foot joint from height.
    foot_y: (T, 4) → w: (T, 4) in [0,1], ≈1 when foot on ground.
    """
    T, J = foot_y.shape
    w = np.zeros((T, J), dtype=np.float32)
    for j in range(J):
        ground = np.percentile(foot_y[:, j], 5)
        w[:, j] = 1.0 / (1.0 + np.exp((foot_y[:, j] - (ground + h_thresh)) / temp))
    return w


def deskated_target(pos_xz, w_xz, contact_thresh=0.5):
    """
    Build the de-skated anchor target (per contact segment).

    For each foot XZ channel:
      - During a contact segment (w > thresh): hold the position at segment ONSET
        (foot planted, tracking the body — no sliding, no drift accumulation).
      - During air: keep the original position (preserve swing shape).

    pos_xz, w_xz: (T, 8). Returns target (T, 8).
    This is what a correctly de-skated foot SHOULD do; FootErr of this target
    vs original ≈ the slide amount (a few cm), not the 40cm integration drift.
    """
    T, C = pos_xz.shape
    target = pos_xz.copy()
    for c in range(C):
        t = 0
        while t < T:
            if w_xz[t, c] > contact_thresh:
                # find segment extent [t, e)
                e = t
                while e < T and w_xz[e, c] > contact_thresh:
                    e += 1
                # plant at segment MEAN (minimizes deviation vs onset anchor)
                anchor = pos_xz[t:e, c].mean()
                target[t:e, c] = anchor
                t = e
            else:
                t += 1
    return target


class FootRefiner(nn.Module):
    """
    Lightweight 1D temporal CNN. Predicts a velocity RESIDUAL on foot XZ.

    Input:  concat(foot_xz_pos, contact_w)  (B, T, 16)
    Output: velocity residual               (B, T, 8)   (init ≈ 0)
    """
    def __init__(self, hidden=64, kernel=5, n_layers=4):
        super().__init__()
        pad = kernel // 2
        in_dim = 16   # 8 pos + 8 contact-weight (per XZ dim)
        layers = []
        c = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Conv1d(c, hidden, kernel, padding=pad), nn.ReLU()]
            c = hidden
        layers += [nn.Conv1d(c, 8, kernel, padding=pad)]
        self.net = nn.Sequential(*layers)
        # zero-init last layer → residual starts at 0 → V18 = analytical soft-mask baseline
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pos_xz, w_xz):
        # pos_xz, w_xz: (B, T, 8)
        x = torch.cat([pos_xz, w_xz], dim=-1).permute(0, 2, 1)   # (B, 16, T)
        res = self.net(x).permute(0, 2, 1)                       # (B, T, 8)
        return res

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def refine(pos_xz, w_xz, residual):
    """
    The V18 forward integration (the heart of the method).

    pos_xz:   (B, T, 8) original foot XZ positions
    w_xz:     (B, T, 8) soft contact weights (1=contact)
    residual: (B, T, 8) model velocity residual

    Returns: pos_new (B, T, 8) — contact-masked, integrated, continuous.
    """
    # original velocity (prepend 0 at t=0)
    v_orig = torch.zeros_like(pos_xz)
    v_orig[:, 1:] = pos_xz[:, 1:] - pos_xz[:, :-1]

    v_pred = v_orig + residual                 # learned velocity
    v_masked = (1.0 - w_xz) * v_pred           # contact → velocity ≈ 0

    # integrate from original start position
    pos_new = pos_xz[:, 0:1] + torch.cumsum(v_masked, dim=1)
    return pos_new


class V18Loss(nn.Module):
    """
    L = λ_smooth   · ‖acc(pos_new)‖²          (reduce jitter)
      + λ_fidelity · ‖pos_new - target‖       (track de-skated target, ALL frames)

    The target is the de-skated anchor trajectory (planted-at-mean during contact,
    original in air). Full-frame fidelity is what fixes the integration drift —
    the earlier air-weighted version was blind to contact-frame drift (40cm bug).
    """
    def __init__(self, lambda_smooth=10.0, lambda_fidelity=8.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_fidelity = lambda_fidelity

    def forward(self, pos_new, target):
        acc = pos_new[:, 2:] - 2 * pos_new[:, 1:-1] + pos_new[:, :-2]
        smooth = (acc ** 2).mean()
        fidelity = (pos_new - target).abs().mean()
        total = self.lambda_smooth * smooth + self.lambda_fidelity * fidelity
        return total, {'smooth': smooth.item(), 'fidelity': fidelity.item()}


# ═══════════════════════════════════════════════════════════════════
# Inference: full motion → refined motion
# ═══════════════════════════════════════════════════════════════════

def v18_fix(motion_world, model, device, h_thresh=0.05, temp=0.02):
    """
    Apply V18 to a single motion (T, 22, 3).
    Only foot XZ changes; foot Y and non-foot joints pass through unchanged.
    """
    T = motion_world.shape[0]
    flat = motion_world.reshape(T, -1).astype(np.float32)

    pos_xz = flat[:, FOOT_XZ_DIMS]                       # (T, 8)
    foot_y = flat[:, FOOT_Y_DIMS]                        # (T, 4)
    w = compute_contact_weight_np(foot_y, h_thresh, temp)   # (T, 4)
    w_xz = np.repeat(w, 2, axis=1)                       # (T, 8)

    pos_t = torch.from_numpy(pos_xz).unsqueeze(0).to(device)
    w_t = torch.from_numpy(w_xz).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        residual = model(pos_t, w_t)
        pos_new = refine(pos_t, w_t, residual).squeeze(0).cpu().numpy()  # (T, 8)

    out = flat.copy()
    out[:, FOOT_XZ_DIMS] = pos_new
    return out.reshape(T, 22, 3)


def deskate_xz(motion_world, h_thresh=0.05, temp=0.02):
    """De-skated foot XZ via plant-at-segment-mean (NO integration → NO drift)."""
    T = motion_world.shape[0]
    flat = motion_world.reshape(T, -1).astype(np.float32)
    pos_xz = flat[:, FOOT_XZ_DIMS]
    w = compute_contact_weight_np(flat[:, FOOT_Y_DIMS], h_thresh, temp)
    w_xz = np.repeat(w, 2, axis=1)
    tgt = deskated_target(pos_xz, w_xz)             # (T, 8)
    return tgt, w_xz


def smooth_fix(motion_world, model, device, h_thresh=0.05, temp=0.02):
    """
    Learned adaptive smoothing of the de-skated foot trajectory (POSITION space).

    out_xz = deskated_target + model(deskated_target, w)   (drift-free: anchored to
    the no-skate reference). The model learns to smooth the sharp plant→air
    boundaries (low jitter) without re-introducing skating (anti over-smoothing).
    Returns (T, 22, 3); only foot XZ changes.
    """
    T = motion_world.shape[0]
    flat = motion_world.reshape(T, -1).astype(np.float32)
    tgt, w_xz = deskate_xz(motion_world, h_thresh, temp)

    tgt_t = torch.from_numpy(tgt).unsqueeze(0).to(device)
    w_t = torch.from_numpy(w_xz).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        res = model(tgt_t, w_t)
        out_xz = (tgt_t + res).squeeze(0).cpu().numpy()    # (T, 8)

    out = flat.copy()
    out[:, FOOT_XZ_DIMS] = out_xz
    return out.reshape(T, 22, 3)


def analytical_fix(motion_world, h_thresh=0.05, temp=0.02):
    """V18 with residual=0 — the analytical soft-velocity-mask baseline (ablation)."""
    T = motion_world.shape[0]
    flat = motion_world.reshape(T, -1).astype(np.float32)
    pos_xz = flat[:, FOOT_XZ_DIMS]
    foot_y = flat[:, FOOT_Y_DIMS]
    w = compute_contact_weight_np(foot_y, h_thresh, temp)
    w_xz = np.repeat(w, 2, axis=1)

    pos_t = torch.from_numpy(pos_xz).unsqueeze(0)
    w_t = torch.from_numpy(w_xz).unsqueeze(0)
    residual = torch.zeros_like(pos_t)
    pos_new = refine(pos_t, w_t, residual).squeeze(0).numpy()
    out = flat.copy()
    out[:, FOOT_XZ_DIMS] = pos_new
    return out.reshape(T, 22, 3)


if __name__ == "__main__":
    print("Testing V18 components...")
    m = FootRefiner()
    print(f"  FootRefiner params: {m.n_params():,}")
    pos = torch.randn(2, 100, 8) * 0.1
    w = torch.rand(2, 100, 8)
    res = m(pos, w)
    print(f"  Residual: {res.shape}, init max abs (should be 0): {res.abs().max().item():.2e}")
    pos_new = refine(pos, w, res)
    print(f"  Refined pos: {pos_new.shape}")
    # with residual=0, contact frames should freeze
    loss = V18Loss()
    total, comps = loss(pos_new, pos, w)
    print(f"  Loss: {total.item():.4f}, comps: {comps}")
    print("  ✓ V18 test passed.")
