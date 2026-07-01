# V17 — Hybrid (Physics + Learned Smoother) — Version Log

**Date:** 2026-06-24
**Method:** Two-stage hybrid foot-skating correction
**Eval:** 10 held-out MoMask motions (same split as V15/V16)

## Pipeline

```
Generator output (T,22,3)
   │
   ├─ Stage 1: Physics correction (rule-based, model-agnostic)
   │           Cap foot horizontal velocity at contact frames (damp=0.0)
   │
   └─ Stage 2: FootSmoother (learned, 25.7K params, 1D temporal CNN)
               Residual on foot XZ, contact-aware Gaussian target,
               anti-skate guarded. Foot Y + non-foot pass through.
```

## Results — 3-way Ablation (n=10 held-out)

| Metric | Original | Physics-only | Hybrid (V17) |
|--------|----------|--------------|--------------|
| **FSR** ↓ | 16.3% | **11.9%** | 20.2% |
| **Jitter** ↓ | **0.0140** | 0.0319 | 0.0244 |
| Floating ↓ | 0.0% | 0.0% | 0.0% |
| FootErr | — | 0.0140 | 0.0600 |
| ContactAcc ↑ | 100% | 100% | 100% |
| BoneCV ↓ | 0.0050 | 0.0310 | 0.0489 |
| Penetration ↓ | 0.0111 | 0.0111 | 0.0111 |

**Semantic preservation (Hybrid vs Original):**
- Non-foot joints max change: 0.000000 m (upper body untouched by construction)
- Joint-frames modified: 17.2% (only feet at skating frames)
- Avg foot displacement: 0.060 m

## Key Finding: FSR–Jitter Antagonism

The ablation reveals a **fundamental trade-off** at contact frames:
- Low FSR requires foot velocity → 0 at contact → flat segments with sharp
  boundaries → **high jitter**.
- Low jitter requires smooth foot velocity → foot moves at contact → **high FSR**.

The learned smoother reduces physics jitter (0.0319 → 0.0244, −24%) but this
re-introduces contact-frame velocity, raising FSR (11.9% → 20.2%). The hybrid
is **not Pareto-optimal** with the current (uniform-smoothing) design.

## Diagnosis & Next Step

The smoother smooths foot *position* uniformly, which moves contact frames.
The principled fix: smooth by **decelerating the foot in the AIR before contact**
(air frames don't count toward FSR), so the foot arrives slow and stays still —
smooth approach + zero contact velocity = low jitter AND low FSR simultaneously.

## Files
- `models/v17.py` — FootSmoother + SmootherLoss + hybrid_fix
- `training/v17.py` — smoother training (contact-aware Gaussian target)
- `testing/v17.py` — 3-way ablation + semantic proxy + plots
- `checkpoints/v17/best.pth` — trained smoother
- `analysis/v17_viz/` — trajectory + velocity plots + summary.json
- `outputs/fixed/v17/*.npy` — 10 hybrid-corrected held-out motions
