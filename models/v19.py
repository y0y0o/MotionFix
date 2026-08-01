"""
V19 — IK-in-the-loop, contact-partitioned adaptive foot smoother
================================================================
V19 changes NOTHING about the pipeline shape (de-skate → smoother → 2-bone IK).
It fixes three structural defects found in the V18+IK *training* setup on
2026-07-20. The pipeline is identical; only what the smoother is asked to do
during training changes.

────────────────────────────────────────────────────────────────────────────
A1 — IK IS NOW IN THE TRAINING LOOP
────────────────────────────────────────────────────────────────────────────
V18 trained the smoother on its raw output, but evaluated it AFTER 2-bone IK.
Measured cost of that mismatch on 10 held-out MoMask motions:

    FSR    12.0%  --IK-->  12.9%     (+0.8pp)
    Jitter 0.00970 --IK--> 0.01143   (x1.18)

The learned smoother lost to a tuned Gaussian by 1.8pp FSR — and ~half of
that gap was introduced by a stage it was never trained through. V19 makes
the ankle clamp differentiable (`torch_ankle_ik`) and puts it in the forward
pass, so the loss is computed on the trajectory that is actually evaluated.

Only the ANKLE path of the IK needs to be differentiable: FSR is defined on
ankles (7,8) alone, and `apply_ik` moves the toe RIGIDLY with the ankle
(`toe = ankle_new + (toe-ankle)_orig`). The cosine-rule knee solve does not
touch any metric, so it is omitted from the training graph (still applied at
inference by `models.v18_ik.apply_ik`, unchanged).

────────────────────────────────────────────────────────────────────────────
A3 — THE LOSS NOW ASKS FOR THE BEHAVIOUR ONLY A LEARNED FILTER CAN GIVE
────────────────────────────────────────────────────────────────────────────
The smoother sees the contact weight `w`; a global Gaussian does not. That is
its ONLY structural advantage, and V18's loss never cashed it in:

  V18:  skate = mean(|v| * w^3)          ← a MEAN over velocities
  FSR:  fraction of contact frames with  |v| > 0.03   ← a COUNT over a threshold

Most de-skated contact frames sit far below 0.03, so they dominated the mean
while contributing nothing to FSR. The model spent its budget on frames that
were already passing.

V19 partitions the objective by contact phase, which is exactly the behaviour
a single global sigma provably cannot reproduce:

  L = lam_air   * jitter(acc^2) weighted by (1-w)   ← smooth HARD in the air
    + lam_all   * jitter(acc^2) unweighted           ← keep boundaries continuous
    + lam_skate * soft_count(|v| > 0.03) at contact  ← threshold-ALIGNED, not a mean
    + lam_anch  * |out - deskated|                   ← anti-drift

WARNING — V16 tried a differentiable soft-FSR and it failed catastrophically
(FSR 35.8%, feet 12-25cm off-body: the model gamed the proxy). V19 is only
safe because the escape route V16 used is now physically blocked:
  - the residual is applied ON TOP of the de-skated reference (not a free
    trajectory), and
  - `torch_ankle_ik` clamps the ankle to the leg's reach INSIDE the loss.
FootErr is logged every epoch as the tripwire. If it climbs, the proxy is
being gamed again — see docs/v19_devlog.md.

────────────────────────────────────────────────────────────────────────────
Also fixed
────────────────────────────────────────────────────────────────────────────
  * Output is 4 dims (ankle XZ), not 8. V18 predicted 8 dims but `apply_ik`
    overwrites the toe with a rigid follow — half of V18's capacity and half
    of its loss gradient went to outputs that never reached a metric.
  * All loss terms are mask-normalised by true sequence length. V18 edge-padded
    to 196 frames; the constant tail contributes 0 to every numerator but was
    counted in the mean's denominator, so each motion got a different effective
    learning rate.
"""

import numpy as np
import torch
import torch.nn as nn

# HumanML3D legs: hip -> knee -> ankle -> toe
LEGS = [
    {"hip": 1, "knee": 4, "ankle": 7, "toe": 10},
    {"hip": 2, "knee": 5, "ankle": 8, "toe": 11},
]
ANKLES = [7, 8]
TOES = [10, 11]
EPS = 1e-6

# Must match utils.metrics.compute_fsr / compute_contact_labels exactly.
SKATE_THRESH = 0.03      # m/frame — the FSR decision boundary
H_THRESH = 0.05          # m above ground counts as contact
TEMP = 0.02              # contact-weight sigmoid temperature


def _safe_norm(x, dim=-1):
    """Norm with a gradient-safe floor (plain .norm() is NaN at 0)."""
    return torch.sqrt((x * x).sum(dim=dim).clamp(min=EPS))


def torch_ankle_ik(hip, ankle_tgt, L1, L2):
    """
    Differentiable ankle clamp — the ONLY part of 2-bone IK that moves a metric.

    Mirrors `models.v18_ik.two_bone_ik`'s ankle path exactly:
      1. clamp the HORIZONTAL reach (preserving the foot's height Y), then
      2. clamp the hip->ankle distance into [|L1-L2|, L1+L2].

    hip, ankle_tgt : (B, T, 3)
    L1, L2         : (B, T)    per-frame thigh / shin lengths from the original
    returns        : (B, T, 3) reachable ankle position
    """
    reach = (L1 + L2) - 1e-4

    dy = ankle_tgt[..., 1] - hip[..., 1]
    horiz = ankle_tgt - hip
    horiz = horiz * torch.tensor([1.0, 0.0, 1.0], device=horiz.device, dtype=horiz.dtype)
    r = _safe_norm(horiz)

    dy_cl = torch.clamp(dy, min=-reach, max=reach)
    max_r = torch.sqrt((reach ** 2 - dy_cl ** 2).clamp(min=EPS))
    r_cl = torch.minimum(r, max_r)

    ankle_h = hip + horiz / r.unsqueeze(-1) * r_cl.unsqueeze(-1)
    ankle_h = torch.stack(
        [ankle_h[..., 0], hip[..., 1] + dy_cl, ankle_h[..., 2]], dim=-1
    )

    AB = ankle_h - hip
    d = _safe_norm(AB)
    dmin = (L1 - L2).abs() + 1e-4
    d_cl = torch.clamp(d, min=EPS).clamp(max=reach)
    d_cl = torch.maximum(d_cl, dmin)

    return hip + AB / d.unsqueeze(-1) * d_cl.unsqueeze(-1)


class V19Smoother(nn.Module):
    """
    1D temporal CNN. Predicts a POSITION residual on the two ankles' XZ only.

    Input  (B, T, 12): deskated ankle XZ (4) | original ankle XZ (4) | contact w (4)
      - giving it BOTH the de-skated and the original lets it see how much slide
        was removed, i.e. how much room it has to smooth.
    Output (B, T, 4) : ankle XZ residual, zero-initialised
      -> at init the model is exactly the de-skate baseline (safe start).
    """

    def __init__(self, hidden=64, kernel=5, n_layers=4, in_dim=12, out_dim=4):
        super().__init__()
        pad = kernel // 2
        layers, c = [], in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Conv1d(c, hidden, kernel, padding=pad), nn.ReLU()]
            c = hidden
        layers += [nn.Conv1d(c, out_dim, kernel, padding=pad)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, deskated_xz, orig_xz, w):
        x = torch.cat([deskated_xz, orig_xz, w], dim=-1).permute(0, 2, 1)
        return self.net(x).permute(0, 2, 1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class V19Loss(nn.Module):
    """
    Contact-partitioned, threshold-aligned, mask-normalised.

    All terms are computed on the POST-IK ankle trajectory and the rigidly
    derived toe, i.e. on exactly what `utils.metrics` will see.
    """

    def __init__(self, lam_air=1.0, lam_all=1.0, lam_skate=1.0, lam_anch=1.0,
                 tau=0.008):
        super().__init__()
        self.lam_air, self.lam_all = lam_air, lam_all
        self.lam_skate, self.lam_anch = lam_skate, lam_anch
        self.tau = tau

    @staticmethod
    def _wmean(x, weight):
        """Weighted mean with a safe denominator (mask normalisation, B6)."""
        return (x * weight).sum() / weight.sum().clamp(min=1.0)

    def terms(self, ankle_xz, toe_xz, deskated_xz, w, mask):
        """
        ankle_xz, toe_xz, deskated_xz : (B, T, 4)   [legL_x, legL_z, legR_x, legR_z]
        w    : (B, T, 4)  contact weight, repeated per XZ dim
        mask : (B, T)     1 = real frame, 0 = padding
        """
        m = mask.unsqueeze(-1)

        # ── jitter: 2nd difference, on BOTH ankles and toes (matches compute_jitter)
        def acc(x):
            return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]

        a_ank, a_toe = acc(ankle_xz), acc(toe_xz)
        m2 = m[:, 2:]
        w2 = w[:, 2:]
        air2 = 1.0 - w2

        jit_all = self._wmean(a_ank ** 2, m2) + self._wmean(a_toe ** 2, m2)
        jit_air = self._wmean(a_ank ** 2, m2 * air2) + self._wmean(a_toe ** 2, m2 * air2)

        # ── skate: soft COUNT of contact frames over the FSR threshold (ankles only)
        d = ankle_xz[:, 1:] - ankle_xz[:, :-1]                      # (B,T-1,4)
        vx, vz = d[..., 0::2], d[..., 1::2]                         # per leg
        speed = torch.sqrt((vx ** 2 + vz ** 2).clamp(min=EPS))      # (B,T-1,2)
        over = torch.sigmoid((speed - SKATE_THRESH) / self.tau)
        w_ank = w[:, 1:, 0::2]                                      # (B,T-1,2)
        skate = self._wmean(over, mask[:, 1:].unsqueeze(-1) * w_ank ** 3)

        # ── anchor: stay on the de-skated reference (anti-drift / anti-gaming)
        anch = self._wmean((ankle_xz - deskated_xz).abs(), m)

        return {'jit_air': jit_air, 'jit_all': jit_all, 'skate': skate, 'anch': anch}

    def forward(self, ankle_xz, toe_xz, deskated_xz, w, mask):
        t = self.terms(ankle_xz, toe_xz, deskated_xz, w, mask)
        total = (self.lam_air * t['jit_air'] + self.lam_all * t['jit_all']
                 + self.lam_skate * t['skate'] + self.lam_anch * t['anch'])
        return total, {k: float(v) for k, v in t.items()}


# ═══════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════

def smooth_fix_v19(motion_world, model, device, h_thresh=H_THRESH, temp=TEMP):
    """
    De-skate -> V19 smoother, ankles only. Returns (T, 22, 3) PRE-IK.
    Feed the result to `models.v18_ik.apply_ik(original, this)` exactly as the
    V18 pipeline does — the IK stage itself is unchanged.
    """
    from models.v18 import deskate_xz, FOOT_XZ_DIMS

    T = motion_world.shape[0]
    flat = motion_world.reshape(T, -1).astype(np.float32)
    tgt8, w8 = deskate_xz(motion_world, h_thresh, temp)   # (T,8) both

    # FOOT_XZ_DIMS order is [7x,7z, 8x,8z, 10x,10z, 11x,11z] -> ankles are 0:4
    des4 = tgt8[:, 0:4]
    w4 = w8[:, 0:4]
    orig4 = flat[:, [ANKLES[0] * 3, ANKLES[0] * 3 + 2,
                     ANKLES[1] * 3, ANKLES[1] * 3 + 2]]

    to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        res = model(to(des4), to(orig4), to(w4)).squeeze(0).cpu().numpy()
    out4 = des4 + res

    out = flat.copy()
    out[:, FOOT_XZ_DIMS[0:4]] = out4          # ankles; toes are rebuilt by apply_ik
    return out.reshape(T, 22, 3)


def pipeline_fix_v19(motion_orig, model, device):
    """Full V19 pipeline: de-skate -> V19 smoother -> 2-bone IK."""
    from models.v18_ik import apply_ik
    return apply_ik(motion_orig, smooth_fix_v19(motion_orig, model, device))


if __name__ == "__main__":
    # ── check torch_ankle_ik reproduces the numpy IK's ankle to float precision
    from models.v18_ik import two_bone_ik

    rng = np.random.default_rng(0)
    T = 200
    hip = rng.normal(0, 0.3, (T, 3)).astype(np.float64)
    knee = hip + rng.normal(0, 0.3, (T, 3))
    ankle = knee + rng.normal(0, 0.3, (T, 3))
    L1 = np.linalg.norm(knee - hip, axis=-1)
    L2 = np.linalg.norm(ankle - knee, axis=-1)
    tgt = ankle + rng.normal(0, 1.5, (T, 3))      # deliberately unreachable
    tgt[:, 1] = ankle[:, 1]                        # V18 only moves XZ

    _, ankle_np = two_bone_ik(hip, knee, tgt.copy(), L1, L2)
    ankle_t = torch_ankle_ik(
        torch.tensor(hip)[None], torch.tensor(tgt)[None],
        torch.tensor(L1)[None], torch.tensor(L2)[None],
    )[0].numpy()

    err = np.abs(ankle_np - ankle_t).max()
    print(f"  torch vs numpy IK ankle — max abs diff: {err:.3e}")
    assert err < 1e-6, "differentiable IK does not match the numpy IK"

    g = torch.tensor(hip, requires_grad=True)
    out = torch_ankle_ik(g[None], torch.tensor(tgt)[None],
                         torch.tensor(L1)[None], torch.tensor(L2)[None])
    out.sum().backward()
    assert torch.isfinite(g.grad).all(), "IK produced non-finite gradients"
    print("  gradients finite ✓")

    m = V19Smoother()
    print(f"  V19Smoother params: {m.n_params():,}")
    des = torch.randn(2, 100, 4) * 0.1
    r = m(des, des, torch.rand(2, 100, 4))
    print(f"  residual init max |r| (must be 0): {r.abs().max().item():.2e}")
    assert r.abs().max().item() == 0.0
    print("  ✓ V19 components OK")
