# MotionFix — Detailed Outline & Evidence Map
> Output of `academic-paper` **outline-only** mode. Structure pattern: **IMRaD**
> (chapter-form, MSc thesis). Target ~15,000 words. Framing: method-contribution.
> `criteria_binding_unavailable` — no venue review-target supplied; no
> venue-alignment claim is made.
>
> **Citations below are name-only placeholders for works the author already uses;
> DOIs/years are NOT asserted here and MUST be verified in the citation phase
> (IRON RULE: no fabricated references).** Chapter 6 is an optional module.

---

## Overview

The paper introduces a generator-agnostic, post-hoc pipeline that corrects
foot-skating in text-to-motion output without retraining the generator. It flows
IMRaD-style: motivate the gap (Ch1) → position against generators, representation,
and prior de-skating (Ch2) → specify the three-stage method with IK-in-the-loop
(Ch3) → describe the leak-free multi-generator protocol and the corrected FID
reference (Ch4) → report the ablation showing physics does most of the work and a
46K learned smoother only matches a tuned Gaussian, without breaking text
alignment (Ch5) → optionally validate perceptually (Ch6) → discuss what the
finding localises and the honest limits (Ch7) → conclude (Ch8).

## Structure Pattern: IMRaD (chapter-form)

Outline depth (>10k words): Level 2 = 3–5 per chapter; Level 3 used freely;
Level 4 only in the method chapter.

---

## Detailed Outline

### 1. Introduction (~1,600 words)
**Purpose:** establish the problem, the gap, and the three contributions (incl.
the honest finding) up front.
**Serves:** framing (no sub-question binding).
**Content:**
- 1.1 Motivation (~500)
  - Foot-skating is a pervasive artefact in text-to-motion output.
  - It is usually fixed *per generator* (retraining / contact-aware loss) —
    expensive, non-portable.
- 1.2 Problem statement & research question (~350)
  - RQ: can one post-hoc, generator-agnostic module correct feet without
    retraining and without breaking text alignment — and does a *learned*
    smoother beat a well-tuned classical one?
- 1.3 Contributions (~500)
  - (1) generator-agnostic pipeline; (2) IK-in-the-loop training; (3) leak-free
    multi-generator protocol + corrected FID reference + honest central finding
    (physics does most of the work; learned ≈ tuned Gaussian).
- 1.4 Thesis structure (~250)
**Sources:** HumanML3D [verify], T2M-GPT [verify], MoMask [verify], MDM [verify].
**Transition:** the gap named here motivates the background needed to place it —
generators, representation, and prior fixes (Ch2).

### 2. Background & Related Work (~2,400 words)
**Purpose:** give the minimum to read the results and set up the metric critique.
**Serves:** framing / positioning.
**Content:**
- 2.1 Text-to-motion generation (~700)
  - T2M-GPT (VQ-VAE + transformer); MoMask (RVQ + mask/residual transformer);
    MDM (diffusion). Only enough detail to interpret outputs.
- 2.2 The HumanML3D motion representation (~600)
  - 22-joint skeleton, 263-d features, `recover_from_ric`, contact channels.
- 2.3 Foot-skating: definition and prior corrections (~700)
  - FSR definition; classical IK; smoothing; contact-aware training. Why post-hoc,
    generator-agnostic correction is under-explored.
- 2.4 Evaluation metrics and their pitfalls (~400)
  - FSR, Jitter, FID, R-precision, MM-Dist, Diversity; FID's rank/reference
    sensitivity — foreshadows §4.4.
**Sources:** all four above + a text-motion-matching evaluator ref [verify].
**Transition:** with generators and metrics fixed, the method chapter specifies
the corrector.

### 3. Method: the V19 pipeline (~2,900 words)
**Purpose:** specify the three-stage corrector; IK-in-the-loop is the novelty.
**Serves:** RQ (mechanism).
**Content:**
- 3.1 Overview (~400)
  - Three stages; only the middle is neural; de-skate and IK are zero-parameter.
    Pipeline figure.
- 3.2 Physics de-skate (~450)
  - Plant-at-mean XZ; no drift; large FSR reduction. (`data/prep/v19.py`)
- 3.3 Learned smoother (~650)
  - 46,276-param 1D-CNN, ankles only; loss design — threshold-aligned soft-count
    anti-skate term, not a mean proxy. (`models/v19.py::V19Loss`)
- 3.4 IK-in-the-loop training (~800) **[core novelty]**
  - 3.4.1 differentiable ankle clamp in the forward pass; loss on the evaluated
    trajectory (matches numpy IK to 2.2e-16).
  - 3.4.2 λ auto-calibration by per-term gradient norm.
  - 3.4.3 validation-loss checkpointing. (`models/v19.py::torch_ankle_ik`,
    `training/v19.py`)
- 3.5 2-bone IK (~350)
  - hip→knee→ankle + rigid toe; bone-length rigidity; contact guarantee.
- 3.6 Delivery point 088a10 (~250)
  - `--jit-share 0.88 --anch-share 0.10`; a low-jitter operating point.
**Sources:** own method (`docs/motionfix_v19_full_spec.md`); IK / smoothing
references [verify].
**Transition:** the method fixed, we define how it is evaluated leak-free.

### 4. Experimental Setup (~1,900 words)
**Purpose:** show the evaluation is leak-free, multi-generator, and honestly
scoped.
**Serves:** RQ (validity).
**Content:**
- 4.1 Data & no-leakage argument (~450)
  - Trained on ~3.4–4k HumanML3D; MoMask/MDM/T2M-GPT all OOD by construction.
    (`training/v19.py` B7)
- 4.2 Physics metrics & invariant-by-construction disclosure (~450)
  - Seven metrics; only FSR/Jitter/FootErr move; Floating/ContactAcc/Penetration/
    BoneCV constant by construction — stated here, once. (`testing/v19_eval.py`)
- 4.3 Semantic evaluator (~350)
  - HumanML3D text-motion-matching (FID/R-prec/MM-Dist/Diversity), paired
    bootstrap.
- 4.4 FID reference correction (~450) **[methodological highlight]**
  - old n=26 rank-deficient in 512-d → FID noise; rebuilt from `test.txt` = 1215
    GT (full-rank). (`testing/v19_fid_ref.py`, `semantic_largeref.json`)
- 4.5 Perceptual study design (~200)
  - blind 2-AFC, 12 motions × 3 comparisons, controls 000076/000119.
    (`outputs/perceptual/rating.html`)
**Sources:** evaluator ref [verify]; FID ref [verify].
**Transition:** with the protocol set, the ablation results follow.

### 5. Results & Ablation (~2,900 words)
**Purpose:** show physics dominates FSR reduction; learned ≈ tuned Gaussian;
feet-fixing preserves text alignment.
**Serves:** RQ (evidence).
**Content:**
- 5.1 Main 5-way ablation × 3 generators (~800)
  - FSR/Jitter at n=200 (T2M-GPT/MoMask); de-skate drives FSR down, smoothing
    trades a little back to kill jitter.
    (`v19_088a10_expanded_results.json`, `momask_pool_results.json`)
- 5.2 FSR–Jitter frontier (~550)
  - learned ≈ tuned Gaussian across the whole frontier, not one point.
    (`frontier.json`)
- 5.3 Per-category generalisation (~550)
  - 7 categories; none regress; rotation retains most residual. (`by_category.json`)
- 5.4 Semantic / FID with corrected reference (~700)
  - de-skate hurts realism most; smoother repairs to ≈ original; MM-Dist/R-prec
    changes not significant. (`semantic_largeref.json`)
- 5.5 Jitter-spike analysis (~300)
  - p99/max spike rate; RMS masks twitches. (`jitter_stats.json`)
    **[BLOCKER: re-run for 088a10 before drafting — disk json is v19_045]**
**Sources:** own data (on-disk json).
**Transition:** results interpreted next; the perceptual test (Ch6) probes the
one unverified assumption.

### 6. Perceptual Study *(optional module, ~1,000 words)*
**Purpose:** test "lower jitter = looks more natural".
**Serves:** RQ (the trade-off's hidden premise).
**Content:**
- 6.1 Hypothesis & design (~400) — blind 2-AFC; attention-check controls.
- 6.2 Analysis (~300) — per-comparison v19 preference; binomial test.
  (`analysis/perceptual_analysis.py`)
- 6.3 Results **or** protocol + expected outcome + placeholder (~300).
**Sources:** own tooling.
**Transition:** whether shipped or protocol-only, feeds the discussion of what the
finding means.

### 7. Discussion & Limitations (~1,500 words)
**Purpose:** locate the contribution and state honest limits.
**Serves:** interpretation.
**Content:**
- 7.1 What "learned ≈ Gaussian" means (~500) — value = pipeline + IK-in-the-loop
  + protocol; a 46K model matching a strong baseline while fixing jitter spikes
  and semantic drift is the result, not a defeat.
- 7.2 MDM not reproducible (~400) — needs `--use_ema` + batch ≤ 10; FID ≈ 16
  confirms off-distribution; kept at n=50, excluded from method conclusions.
- 7.3 Data / sample-size limits (~300) — local HumanML3D 8177/14616; only affected
  FID (now fixed).
- 7.4 Method boundary (~300) — rotation residual; [+ perceptual as future work if
  Ch6 cut].
**Sources:** `supervisor_report_20260822.md` §5; memory `mdm-generation-gotchas`.
**Transition:** limits acknowledged, the conclusion recaps the contribution.

### 8. Conclusion & Future Work (~800 words)
**Purpose:** recap and point forward.
**Content:**
- 8.1 Summary of contributions (~450) — the three, restated tightly.
- 8.2 Future work (~350) — stronger learned stage, contact-aware training signal,
  more generators, completed perceptual study.
**Sources:** —
**Transition:** end.

---

## Evidence Map — external literature (verify in citation phase)

| Section | Assigned sources (name-only, [verify]) | Evidence type |
|---|---|---|
| 1.1, 2.1 | T2M-GPT; MoMask; MDM | generators, problem context |
| 2.2 | HumanML3D | representation |
| 2.3 | classical IK / smoothing / contact-aware works [verify] | prior corrections |
| 2.4, 4.3–4.4 | text-motion-matching evaluator; FID [verify] | metric definitions |
| 3.x | own method spec + IK/smoothing refs [verify] | method justification |
| 7.1 | prior post-hoc / classical-baseline works [verify] | comparison |

> No source here carries an asserted DOI/year. Run `/ars-citation-check` or the
> citation phase of `/ars-full` to verify every reference before submission.

## Evidence Map — on-disk data artifacts (authoritative)

| Section | Artifact |
|---|---|
| 5.1 | `analysis/v19/v19_088a10_expanded_results.json`, `momask_pool_results.json` |
| 5.2 | `analysis/v19/frontier.json` |
| 5.3 | `analysis/v19/by_category.json` |
| 5.4 | `analysis/v19/semantic_largeref.json` (script `testing/v19_fid_ref.py`) |
| 5.5 | `analysis/v19/jitter_stats.json` **[re-run for 088a10]** |
| 3.x | `models/v19.py`, `training/v19.py`, `data/prep/v19.py`, `docs/motionfix_v19_full_spec.md` |
| 4.1–4.2 | `testing/v19_eval.py` |
| 6.x | `outputs/perceptual/rating.html`, `analysis/perceptual_analysis.py` |

## Word Count Summary

| Section | Target words |
|---|---:|
| Abstract (bilingual) | 250 |
| 1. Introduction | 1,600 |
| 2. Background & Related Work | 2,400 |
| 3. Method | 2,900 |
| 4. Experimental Setup | 1,900 |
| 5. Results & Ablation | 2,900 |
| 6. Perceptual Study (optional) | 1,000 |
| 7. Discussion & Limitations | 1,500 |
| 8. Conclusion & Future Work | 800 |
| **Total (with Ch6)** | **15,000** |
| Total (Ch6 cut, +300 to §7.4) | 14,300 |

`criteria_binding_unavailable`

---

## Quality-gate status & open items

- Every section has a Purpose and a Transition. ✔
- Word counts sum to target (±0%). ✔
- On-disk artifacts each assigned to a section. ✔
- External citations are placeholders pending verification. ⚠ (citation phase)
- **User approval required before Phase 3 (argument building) / drafting.**

**Pre-drafting blockers:**
- [x] ~~re-run jitter trace for 088a10~~ **DONE 2026-08-31** (`analysis/v19/spike_ratios_088a10.json`):
  v19/orig p99·max = T2M-GPT 0.93·0.98, MoMask **1.03·1.07**, MDM 0.90·0.83. Paper §5.E
  "0.91×/0.94× on all three generators" is WRONG — MoMask spikes rise. Rewrite per generator.
- [x] ~~per-category for 088a10~~ **DONE 2026-08-31** (`analysis/v19/by_category_088a10.json`,
  identical to old json → 088a10 all along): backward (0.190→0.196) and jumping (0.033→0.037)
  REGRESS; largest residual is backward (0.196), not rotation (0.175). Paper §5.C "no category
  regresses" and "rotation largest residual" are both WRONG.
- [ ] resolve V8-baseline number conflict (CLAUDE.md −2.9% vs research_log 14.1%→15.6%).
- [ ] re-render 088a10 videos; decide Chapter 6.
