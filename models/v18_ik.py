"""
V18 + 2-bone IK — Bone-Consistent Foot Refinement
=================================================
V18 (velocity-space contact mask) gives us ankles that are planted (low FSR)
and smooth (low Jitter) — BUT it moves the ankle/foot freely in XZ, which:
  (a) stretches the shin (knee→ankle) up to 2m on rotation motions, and
  (b) flips the foot, because the toe joint (10/11) is moved independently.

The IK stage repairs the SKELETON while keeping V18's planted ankle target:

  For each leg  hip → knee → ankle  (+ rigid toe):
    1. Take V18's corrected ankle XZ as the IK target (Y kept from original).
    2. CLAMP the target to the leg's reach (|hip-ankle| ≤ thigh + shin).
       → this is what STOPS the unbounded stretch / drift.
    3. Solve the knee with analytic 2-bone IK (cosine rule), using the ORIGINAL
       knee to fix the bend plane (knee keeps bending the natural way).
    4. Move the toe RIGIDLY with the ankle (toe_new = ankle_new + (toe-ankle)_orig)
       → bone length + foot orientation preserved, no flip.

Bone lengths are taken PER-FRAME from the original motion, so the corrected
skeleton reproduces the source bone lengths exactly (BoneCV ≈ original).

Pelvis, hips, spine, arms, head: untouched. Only knees/ankles/toes change.
"""

import numpy as np

# HumanML3D skeleton — leg chains
# Left : hip 1 → knee 4 → ankle 7 → toe 10
# Right: hip 2 → knee 5 → ankle 8 → toe 11
LEGS = [
    {"hip": 1, "knee": 4, "ankle": 7, "toe": 10},
    {"hip": 2, "knee": 5, "ankle": 8, "toe": 11},
]
EPS = 1e-6


def _norm(v):
    return np.linalg.norm(v, axis=-1)


def two_bone_ik(hip, knee_orig, ankle_tgt, L1, L2):
    """
    Analytic 2-bone IK, vectorised over time.

    hip, knee_orig, ankle_tgt : (T, 3)   world positions
    L1 : (T,)  thigh length  |hip - knee|   (per-frame, from original)
    L2 : (T,)  shin  length  |knee - ankle| (per-frame, from original)

    Returns:
      knee_new  : (T, 3)
      ankle_new : (T, 3)   = ankle_tgt clamped to reachable distance
    """
    reach = (L1 + L2) - 1e-4                      # max straight-leg length

    # Clamp the HORIZONTAL drift only, preserving the foot's height (Y).
    # This bounds the leg-tear without lifting a planted foot off the ground.
    dy = ankle_tgt[:, 1] - hip[:, 1]             # vertical offset (kept)
    horiz = ankle_tgt - hip
    horiz[:, 1] = 0.0
    r = _norm(horiz)                             # horizontal distance to ankle

    # if the leg can't even reach vertically, clamp dy and zero the horizontal reach
    dy_cl = np.clip(dy, -reach, reach)
    max_r = np.sqrt(np.clip(reach**2 - dy_cl**2, 0.0, None))
    r_cl = np.minimum(r, max_r)

    safe_r = np.where(r > EPS, r, 1.0)
    horiz_unit = horiz / safe_r[:, None]
    ankle_h = hip + horiz_unit * r_cl[:, None]
    ankle_h[:, 1] = hip[:, 1] + dy_cl            # horizontally-clamped, height-preserved

    AB = ankle_h - hip
    d = _norm(AB)

    # cosine-rule reachable range (lower bound = legs can't fully fold)
    dmin = np.abs(L1 - L2) + 1e-4
    d_cl = np.clip(d, dmin, reach)

    safe_d = np.where(d > EPS, d, 1.0)
    n = AB / safe_d[:, None]
    ankle_new = hip + n * d_cl[:, None]          # final ankle (consistent with cosine geom)

    # cosine rule: distance from hip to the knee's foot-of-perpendicular
    a = (d_cl**2 + L1**2 - L2**2) / (2.0 * d_cl)
    h2 = L1**2 - a**2
    h = np.sqrt(np.clip(h2, 0.0, None))          # perpendicular height

    P = hip + n * a[:, None]                      # base point on hip→ankle line

    # bend direction from the ORIGINAL knee (component perpendicular to n)
    kr = knee_orig - hip
    kr_perp = kr - (kr * n).sum(-1, keepdims=True) * n
    m_norm = _norm(kr_perp)

    # fallback bend direction where the original knee is collinear with the leg
    fallback = np.tile(np.array([0.0, 0.0, 1.0]), (hip.shape[0], 1))
    fb_perp = fallback - (fallback * n).sum(-1, keepdims=True) * n
    fb_norm = _norm(fb_perp)
    fb_safe = np.where(fb_norm[:, None] > EPS, fb_perp / np.where(fb_norm > EPS, fb_norm, 1.0)[:, None],
                       np.tile(np.array([1.0, 0.0, 0.0]), (hip.shape[0], 1)))

    use_orig = m_norm > EPS
    m = np.where(use_orig[:, None],
                 kr_perp / np.where(use_orig, m_norm, 1.0)[:, None],
                 fb_safe)

    knee_new = P + m * h[:, None]
    return knee_new, ankle_new


def apply_ik(motion_orig, motion_v18):
    """
    Combine V18's planted ankles with bone-consistent IK.

    motion_orig : (T, 22, 3)  original generator output (source skeleton)
    motion_v18  : (T, 22, 3)  V18-corrected motion (ankle/toe moved in XZ)

    Returns (T, 22, 3): pelvis/hips/spine/arms/head = original,
                        knees/ankles/toes = IK-repaired.
    """
    out = motion_orig.copy()

    for leg in LEGS:
        hip = motion_orig[:, leg["hip"], :]          # fixed
        knee_o = motion_orig[:, leg["knee"], :]
        ankle_o = motion_orig[:, leg["ankle"], :]
        toe_o = motion_orig[:, leg["toe"], :]

        # original per-frame bone lengths (these get preserved exactly)
        L1 = _norm(knee_o - hip)
        L2 = _norm(ankle_o - knee_o)

        # V18 target: corrected XZ, original Y (V18 only touches XZ)
        ankle_tgt = ankle_o.copy()
        ankle_tgt[:, 0] = motion_v18[:, leg["ankle"], 0]
        ankle_tgt[:, 2] = motion_v18[:, leg["ankle"], 2]

        knee_new, ankle_new = two_bone_ik(hip, knee_o, ankle_tgt, L1, L2)

        # rigid toe: preserve original ankle→toe bone vector (no stretch, no flip)
        toe_new = ankle_new + (toe_o - ankle_o)

        out[:, leg["knee"], :] = knee_new
        out[:, leg["ankle"], :] = ankle_new
        out[:, leg["toe"], :] = toe_new

    return out


def v18_ik_fix(motion_orig, model, device, h_thresh=0.05, temp=0.02):
    """Full pipeline: V18 velocity-mask → 2-bone IK. Returns (T,22,3)."""
    from models.v18 import v18_fix
    motion_v18 = v18_fix(motion_orig, model, device, h_thresh, temp)
    return apply_ik(motion_orig, motion_v18)


def pipeline_fix(motion_orig, model, device, h_thresh=0.05, temp=0.02):
    """
    FINAL method: de-skate (plant-at-mean) → learned adaptive smoother → 2-bone IK.
      - de-skate: removes foot skating with no integration drift (reachable target)
      - learned smoother: rounds plant→air boundaries (low jitter), adaptively
      - IK: keeps thigh/shin/foot bones rigid (no leg-tear, no foot-flip)
    Returns (T, 22, 3).
    """
    from models.v18 import smooth_fix
    motion_smooth = smooth_fix(motion_orig, model, device, h_thresh, temp)
    return apply_ik(motion_orig, motion_smooth)


if __name__ == "__main__":
    # sanity: a synthetic stretched leg gets clamped back to bone length
    T = 50
    hip = np.zeros((T, 3))
    knee_o = np.tile([0.0, -0.4, 0.05], (T, 1))   # bent forward
    ankle_o = np.tile([0.0, -0.8, 0.0], (T, 1))
    # target pulls ankle 2m away (impossible)
    ankle_tgt = np.tile([2.0, -0.8, 0.0], (T, 1))
    L1 = _norm(knee_o - hip)
    L2 = _norm(ankle_o - knee_o)
    knee_new, ankle_new = two_bone_ik(hip, knee_o, ankle_tgt, L1, L2)
    d_new = _norm(ankle_new - hip)
    L1_new = _norm(knee_new - hip)
    L2_new = _norm(ankle_new - knee_new)
    print(f"  reach max = {(L1+L2)[0]:.3f}, clamped |hip-ankle| = {d_new.mean():.3f}")
    print(f"  thigh: orig {L1[0]:.3f} -> ik {L1_new.mean():.3f}")
    print(f"  shin : orig {L2[0]:.3f} -> ik {L2_new.mean():.3f}")
    assert np.allclose(L1, L1_new, atol=1e-3), "thigh length broken"
    assert np.allclose(L2, L2_new, atol=1e-3), "shin length broken"
    print("  ✓ bone lengths preserved, target clamped to reach.")
