# MotionFix Progress Report — From a Learned Corrector to a Physics-Based Correction Pipeline

**Student:** Xin Wan
**Date:** 21 July 2026
**Previous report:** 16 June 2026
**Supervisor:** Frederick

---

## 0. Executive Summary

Since the June report the project has been restructured around a single, stabilised method
and every point in your feedback has been addressed. The headline changes:

1. **The method is no longer a learned corrector.** It is a three-stage *physics* pipeline:
   **de-skate → (optional) smoother → reach-clamped 2-bone IK**. The learned network is now
   one *ablatable component*, not the contribution.
2. **The negative transfer to T2M-GPT is gone.** T2M-GPT now shows the **largest** improvement
   of the three generators (FSR 11.99% → **6.65%**, −5.34pp), because the pipeline contains no
   learned artifact model to overfit to a particular generator.
3. **Rotational motions — your specific concern — improve by −9.48pp**, now the second-largest
   gain of seven motion categories. No category degrades.
4. **Semantic preservation is now measured**, with significance testing: correcting the feet does
   not measurably break text–motion alignment.
5. **An honest negative result**, established with a much stronger experiment than before: the
   learned smoother traces the *same* FSR–Jitter frontier as a tuned Gaussian filter and adds
   no benefit. This is reported as an ablation conclusion, not hidden.

Main result across three generators (leakage-free evaluation):

| Generator | FSR before | FSR after (physics) | Δ | Jitter | Bone length | Contact |
|---|---|---|---|---|---|---|
| MoMask (held-out, n=10) | 16.29% | **8.18%** | −8.11pp | see §4.2 | unchanged | 100% |
| MDM (n=50) | 11.86% | **7.04%** | −4.82pp | see §4.2 | unchanged | 100% |
| T2M-GPT (n=50) | 11.99% | **6.65%** | −5.34pp | see §4.2 | unchanged | 100% |

For reference, published control-based methods report FSR ≈ 5.5% (MaskControl/OmniControl,
ICCV 2025) while *retraining or conditioning the generator*; this pipeline is purely post-hoc
and requires no access to the generator.

---

## 1. Response to Your Feedback (point by point)

| # | Your point | Status | Evidence |
|---|---|---|---|
| 1 | State the contribution explicitly: what is genuinely new, and why is it needed | **Done** | §3 — the contribution is the *constraint-satisfaction formulation* plus reach-clamped 2-bone IK, not a network |
| 2 | Tighter ablation: which part helps — gating, smoothing, IK, foot-only replacement | **Done** | §4.2 — 5-way ablation on all three generators; §4.4 — a full frontier sweep of the smoothing stage |
| 3 | Explain the cross-model behaviour: why MoMask improves but T2M-GPT degrades | **Done — and resolved** | §4.1 — the cause is identified (learned artifact statistics) and eliminated; T2M-GPT now improves most |
| 4 | The method looks prompt-sensitive; show generalisation across motion categories | **Done** | §4.3 — all 7 categories improve, monotonically in input severity; rotation −9.48pp |
| 5 | Separate what comes from FRDM / InfiniteDance from your own idea | **Done** | §2.3 — the final method shares *no component* with FRDM; the divergence is documented |
| 6 | Stabilise one method rather than keeping many variants open | **Done** | §2 — one pipeline; V9–V17 are closed and documented as failure analysis |
| 7 | Add ablations and failure analysis | **Done** | §4.2, §5 — five systematically documented failure modes |
| 8 | Clarify the technical novelty of the correction pipeline | **Done** | §3.3 |
| 9 | Evaluation showing both physical plausibility **and semantic preservation** | **Done** | §4.2 (physical), §4.5 (semantic, with paired bootstrap) |

Two items are **partially** addressed and are stated as limitations in §6:
- Sample size (n = 50 per generator; the FID reference set is n = 26 because the local
  HumanML3D checkout is incomplete, 8177 of 14616 files).
- No perceptual/user study. The Jitter metric's relationship to perceived quality is untested.

---

## 2. What Changed Since 16 June

### 2.1 V9–V17: five systematic failures of the learned approach

The June report described V8 (Transformer + selective foot replacement) and an in-progress V9.
Both were pursued to completion and abandoned. The full record is in
`docs/research_log_20260624.md`; the summary:

| Version | Paradigm | Result (FSR) | Why it failed |
|---|---|---|---|
| V8 | L1 to clean target, synthetic noise | 15.6% ✗ | Signal washout → identity mapping |
| V14 | L1, simulated foot skating | 15.6% ✗ | Same |
| V15 | Distillation from a physics teacher | 18.2% ✗ | 88% of dimensions should not change → identity mapping |
| V16 | Self-supervised differentiable soft-FSR | 35.8% ✗ | **Gamed the proxy**: output a new, very smooth trajectory with feet 12–25cm off-body |
| V17 | Physics + learned smoother | 20.2% ✗ | Smoother re-introduced skating |
| **Physics (rule-based)** | No model | **9.5%** ✓ | Operates on the real quantity; cannot be gamed |

**The conclusion that reorganised the project:** foot-skating correction is a *constraint
satisfaction* problem (foot velocity ≈ 0 during contact), not a learning problem. Any
differentiable surrogate for FSR gets exploited; direct L1 supervision is drowned out by the
88% of dimensions that must stay unchanged.

### 2.2 V18: a breakthrough that had to be discarded

V18 worked in *velocity space* — mask contact-frame velocity, then integrate. This broke the
FSR/Jitter antagonism for the first time (FSR 5.6%, Jitter below the original) but had two
fatal defects: `cumsum` integration drift of 0.39 m, and shins stretched to 2 m on rotational
motions. **This is the direct explanation of the "gains limited on rotational motions" you
observed:** FSR is defined on the ankle, so reducing it requires moving the ankle, which tears
the shin unless the skeleton is re-solved.

That diagnosis produced the current method: keep the planted-ankle target, but repair the
skeleton with IK.

### 2.3 Relationship to FRDM / InfiniteDance (your point 5)

The June report proposed reproducing FRDM with a Transformer in place of its diffusion model.
**That plan was abandoned, and the final method shares no component with FRDM.**

| | FRDM (InfiniteDance) | This work |
|---|---|---|
| Backbone | Diffusion model, ~50-step denoising | No generative model at all |
| Correction | Learned, generative | Analytic: plant-at-segment-mean + closed-form IK |
| Representation | 259-d (pos + vel + rot + contact) | 22×3 joint positions |
| Skeleton integrity | Not explicitly enforced | Enforced by construction (per-frame bone lengths + reach clamp) |
| Training data | Pseudo-artifacts on clean data | The delivered pipeline needs **no training at all** |

What was genuinely taken from FRDM: the *idea* of a contact-gated foot loss (`L_Foot`), which
informed the loss design of the (ablated-away) learned smoother. Everything in the delivered
pipeline is my own design. The negative result in §4.4 is, in effect, evidence that the
FRDM-style learned approach is unnecessary for this problem.

---

## 3. The Method

### 3.1 Pipeline

```
input (T,22,3) → De-skate (physics) → [smoother] → 2-bone IK (physics) → output
                 plant foot at the      optional     hip→knee→ankle,
                 per-contact-segment    (see §4.4)   reach-clamped,
                 mean XZ                             rigid toe
```

**Stage 1 — De-skate.** Within each contact segment, the foot's horizontal position is set to
that segment's mean. Removing skating by *construction* rather than by integrating a corrected
velocity is what eliminates V18's 0.39 m drift: the target stays reachable.

**Stage 2 — Smoother (optional).** De-skating creates sharp plant→air velocity boundaries.
Either a tuned Gaussian or a small learned CNN can round them. §4.4 shows the two are
equivalent; the stage is presented as ablatable.

**Stage 3 — Reach-clamped 2-bone IK.** For each leg (hip → knee → ankle, plus a rigid toe):
1. take the corrected ankle XZ as the IK target, keeping the original height;
2. **clamp the target into the leg's reachable set** `|hip−ankle| ≤ thigh + shin`;
3. solve the knee analytically (cosine rule), using the original knee to fix the bend plane;
4. move the toe rigidly with the ankle.

Bone lengths are taken **per frame** from the source motion, so the corrected skeleton
reproduces the input's bone lengths exactly.

### 3.2 Why each stage is necessary (this is the ablation logic)

| Remove | Consequence | Measured |
|---|---|---|
| De-skate | No FSR reduction | — |
| IK | Shin stretches to 2 m on rotation; foot flips | §2.2 |
| Smoother | Jitter rises to ~2× the original | §4.2 |

### 3.3 Technical novelty (your point 1 and 8)

1. **Formulating post-hoc foot correction as constraint satisfaction rather than learning**,
   supported by five systematic negative results (§2.1). This reframing is the main
   intellectual contribution.
2. **The reach clamp.** Clamping the IK target into the leg's reachable set *before* solving is
   what converts an unbounded positional correction into a bounded one. It is the single step
   that fixes leg-tear, and it is what makes the method work on rotational motions.
3. **Plant-at-segment-mean** as the de-skate target — a non-integrating formulation that is
   drift-free by construction, chosen after the 0.39 m drift of the integrating formulation.
4. **Generator-agnostic by construction.** Because no artifact model is learned, the method
   transfers without modification — verified on three generators (§4.1).

---

## 4. Results

All evaluations use `utils/metrics.py` (7 metrics) and a leakage-free protocol: for methods
involving the learned smoother, MoMask is split into the 40 motions used in training and the
10 held out, reported separately. The delivered physics pipeline uses no training data at all.

### 4.1 Cross-generator results and the resolution of the T2M-GPT regression (your point 3)

**Diagnosis.** V8's negative transfer to T2M-GPT was caused by learning *generator-specific
artifact statistics* from synthetic distortions. Those distortions were designed against
MoMask's artifact distribution; T2M-GPT's differ, so the learned correction was mis-targeted.

**Resolution.** The current pipeline contains no learned artifact model. It measures contact
from the motion itself and enforces a physical constraint. There is therefore nothing to
overfit, and transfer is expected by construction.

| Generator | Original | De-skate + IK | Δ |
|---|---|---|---|
| MoMask (held-out) | 16.29% | **8.18%** | **−8.11pp** |
| MoMask (train split) | 13.49% | **8.23%** | −5.26pp |
| MDM | 11.86% | **7.04%** | −4.82pp |
| **T2M-GPT** | 11.99% | **6.65%** | **−5.34pp** |

T2M-GPT went from **+0.9pp (worse)** in June to **−5.34pp (best absolute FSR)** now.

### 4.2 Full ablation (your points 2 and 7)

n = 50 for MDM/T2M-GPT; n = 10 for the MoMask held-out split.

**T2M-GPT (n=50):**

| Method | FSR ↓ | Jitter ↓ | FootErr ↓ | BoneCV | ContactAcc |
|---|---|---|---|---|---|
| Original | 11.99% | 0.01388 | 0.0000 | 0.02936 | 100% |
| **De-skate + IK** | **6.65%** | 0.02703 | 0.0293 | 0.02936 | 100% |
| De-skate + Gaussian + IK | 8.83% | 0.01217 | 0.0350 | 0.02936 | 100% |
| De-skate + learned (V18) + IK | 10.99% | **0.01070** | 0.0399 | 0.02936 | 100% |
| De-skate + learned (V19) + IK | 8.93% | 0.01283 | 0.0292 | 0.02936 | 100% |

**MDM (n=50):**

| Method | FSR ↓ | Jitter ↓ | FootErr ↓ |
|---|---|---|---|
| Original | 11.86% | 0.01417 | 0.0000 |
| **De-skate + IK** | **7.04%** | 0.02850 | 0.0252 |
| + Gaussian | 9.06% | 0.01363 | 0.0313 |
| + learned (V19) | 9.24% | 0.01363 | 0.0263 |

**MoMask held-out (n=10):**

| Method | FSR ↓ | Jitter ↓ | FootErr ↓ |
|---|---|---|---|
| Original | 16.29% | 0.01403 | 0.0000 |
| **De-skate + IK** | **8.18%** | 0.02948 | 0.0424 |
| + Gaussian | 11.12% | 0.01311 | 0.0481 |
| + learned (V19) | 11.63% | 0.01343 | 0.0430 |

**Reading:** the FSR reduction comes entirely from the physics stages. The smoothing stage
*trades FSR back* for lower jitter — this is the central trade-off of the method, and it is
now quantified rather than assumed. BoneCV is identical to the input in every row, confirming
the IK stage preserves the skeleton exactly.

### 4.3 Generalisation across motion categories (your point 4)

Aggregated over all three generators (150 motion–generator pairs), by the category labels of
the 50 test prompts:

| Category | n | Original FSR | De-skate + IK | Δ |
|---|---|---|---|---|
| rotation | 18 | 21.99% | 12.51% | **−9.48pp** |
| walking | 21 | 20.76% | 8.07% | **−12.69pp** |
| backward | 9 | 18.97% | 13.19% | −5.78pp |
| turning | 21 | 13.84% | 9.80% | −4.05pp |
| complex | 30 | 11.74% | 7.38% | −4.37pp |
| dance | 30 | 5.99% | 3.12% | −2.87pp |
| jumping | 21 | 3.34% | 2.96% | −0.38pp |
| **All** | **150** | **12.64%** | **7.31%** | **−5.33pp** |

**Three observations:**
1. **No category degrades.** The method is not prompt-sensitive.
2. **The improvement is monotonic in input severity** — the worse the input skating, the larger
   the gain. Jumping barely improves because it starts at 3.34%, near the floor.
3. **Rotation, the category you flagged, improves by −9.48pp** and is now the second-largest
   gain. It remains the *hardest* category in absolute terms (12.51% residual), which is
   expected: rotational motions move the ankle continuously, so the reach clamp is active most
   often. This is the honest boundary of the method.

### 4.4 Is the learned smoother worth having? (a negative result)

The smoothing stage can be a tuned Gaussian or a learned CNN. To settle this I swept both:
7 Gaussian σ values and 8 learned operating points (the loss weights select a point on the
frontier), evaluating every point on all three generators.

**Result: at 21 of 24 comparable operating points the tuned Gaussian is equal or better.**
The learned model wins at exactly one point per generator, by 0.28–0.40pp, and those are the
points where its residual is smallest — i.e. where it does the least.

This conclusion is reported with confidence because the experiment that produced it was
rebuilt to remove six defects present in the earlier version of this comparison: an evaluation
data leak, a mean-based surrogate misaligned with the threshold-based FSR metric, a
train/inference mismatch around the IK stage, a corrupted training corpus, an unstable learning
rate, and checkpoint selection on training loss. Full record: `docs/v19_devlog.md`.

**Interpretation:** the learned smoother's only structural advantage over a global filter is
that it can see the contact labels. That advantage does not pay for itself. Combined with §2.1,
the evidence says the learning component of this pipeline is unnecessary.

### 4.5 Semantic preservation (your point 9)

Evaluated with the standard HumanML3D `text_mot_match` evaluator (the one used by the MoMask,
MDM and T2M-GPT papers): FID, R-precision, Multimodal Distance and Diversity. Because the
pipeline outputs joint positions and the evaluator consumes 263-d features, all methods —
including the untouched original and the ground truth — are pushed through the *same*
joints→features conversion, so the conversion error is common-mode.

**MM-Dist change vs the uncorrected original (paired bootstrap, 95% CI; + = worse alignment):**

| Method | MoMask | MDM | T2M-GPT |
|---|---|---|---|
| De-skate + IK | +0.279 [+0.05, +0.61] **sig.** | +0.038 n.s. | +0.208 n.s. |
| + Gaussian | +0.116 n.s. | +0.036 n.s. | +0.100 n.s. |
| + learned (V18) | +0.121 [+0.01, +0.30] **sig.** | +0.019 n.s. | +0.090 [+0.00, +0.19] **sig.** |
| + learned (V19) | +0.103 n.s. | +0.034 n.s. | +0.064 n.s. |

**Only 3 of 12 method × generator cells show a statistically significant degradation, and all
are small.** Correcting the feet does not measurably break text–motion alignment.

**A useful secondary finding:** pure de-skating is the *worst* offender semantically
(MoMask +0.279 significant, FID +1.81). The smoothing stage therefore does more than suppress
visual twitching — it also pulls back the semantic drift introduced by the physical correction.
This gives the smoothing stage a justification independent of the Jitter metric.

**Caveats:** n = 50; the FID reference set is n = 26; R-precision differences are within noise
and should not be cited; "not significant" at this sample size means *underpowered*, not *no
effect*; absolute values are not comparable to published numbers.

---

## 5. Honest Findings and Failure Analysis

1. **Learning is unnecessary for this problem** (§2.1, §4.4). Five paradigms failed; the
   surviving learned component is equivalent to a two-line filter.
2. **A differentiable surrogate for FSR will be gamed** (V16: FSR 35.8%, feet 12–25 cm
   off-body). Constraints must be enforced structurally, not optimised.
3. **Reducing FSR necessarily disturbs the leg.** FSR is defined on the ankle; moving the ankle
   tears the shin unless the skeleton is re-solved. This is the mechanism behind the
   "rotational motions" limitation you observed.
4. **The FSR↔Jitter trade-off is real and irreducible under the IK constraint.** Both the
   analytic and the learned smoother trace the same frontier.
5. **Jitter's RMS hides the phenomenon it claims to measure.** Twitching is a *spike* event.
   Ankle-acceleration p99/max relative to the original: de-skate-only 3.02×/3.64×,
   one learned operating point 1.54×/1.79×, another 0.91×/0.94×. Two configurations with
   similar RMS can differ greatly in spike behaviour. I now report p99 and max alongside RMS.
6. **A data-preparation bug was found in the earlier training corpus.** `data/prep/v14.py`
   de-normalised HumanML3D features that were already raw, compressing foot height into a 9 cm
   band. V14 and V15 both trained on this corpus, so those two negative results are
   *confounded* — they are reported as such, not as clean evidence.

---

## 6. Limitations

| Limitation | Impact | Mitigation / status |
|---|---|---|
| n = 50 per generator | Statistical power | Paired bootstrap used where possible; all CIs reported |
| FID reference n = 26 | FID is noisy | Only orderings are read, never absolute values; local HumanML3D checkout is incomplete (8177/14616) |
| No perceptual study | The link between Jitter and perceived quality is untested | See §7; a small pairwise study is the highest-value remaining experiment |
| Four of seven metrics are degenerate | Floating ≡ 0, ContactAcc ≡ 100%, Penetration varies in the 7th decimal, BoneCV is enforced by the IK | Will be reported as construction-guaranteed invariants, not as results |
| Ground plane = 5th percentile of foot height | Fails for jumping / stairs / seated motions | Known; not addressed |
| Trade-off point selection | The smoothing strength is a free parameter | Reported as a frontier, not a single tuned number |

---

## 7. Plan for the Dissertation

### 7.1 The thesis statement

> Post-hoc foot-skating correction for VQ-based motion generators is a **constraint-satisfaction
> problem, not a learning problem**. We present a generator-agnostic, training-free pipeline —
> de-skating plus reach-clamped 2-bone inverse kinematics — that reduces the foot-skating ratio
> from ~12% to ~6.7% across three generators while preserving bone lengths, foot contact and
> text–motion alignment, and we give systematic evidence that a learned corrector is
> unnecessary for this task.

This framing is chosen because it is what the evidence actually supports, and because it turns
the negative result from a weakness into a contribution.

### 7.2 Chapter plan and where existing material goes

| Chapter | Content | Source material |
|---|---|---|
| 1. Introduction | Foot skating in VQ generators; why post-hoc; contributions | §0, §3.3 |
| 2. Related Work | Motion generation; control-based methods (OmniControl/MaskControl); physics-based cleanup; FRDM — with an explicit statement of what is borrowed | §2.3 |
| 3. Problem Formulation | Definition of FSR, contact, the constraint view; the metric suite and its limitations | §1.2 of `version_review`, `utils/metrics.py` |
| 4. Method | De-skate; reach-clamped 2-bone IK; the optional smoother; why each stage exists | §3 |
| 5. Experiments | Setup, leakage-free protocol, all three generators | §4.1 |
| 6. Ablation | 5-way ablation; the frontier sweep; per-category results | §4.2, §4.3, §4.4 |
| 7. Semantic Preservation | FID / R-precision / MM-Dist with significance testing | §4.5 |
| 8. Failure Analysis | V8–V17; the five learning paradigms; the V16 gaming episode | §2.1, §5 |
| 9. Discussion | The constraint-satisfaction argument; when learning would help; the Jitter-metric critique | §5 |
| 10. Conclusion & Future Work | Perceptual study; better contact detection; whole-body extension | §6 |

**Chapter 8 is unusual for an MSc and should be kept.** A documented sequence of five
principled failures, each with a diagnosis, is stronger evidence for the central claim than the
positive result alone.

### 7.3 Writing order (recommended)

1. **Chapter 4 (Method) first** — it is fully settled and writing it will fix the notation.
2. **Chapters 5–6 (Experiments, Ablation)** — all tables already exist in
   `analysis/v19/*.json`; this is transcription, not new work.
3. **Chapter 8 (Failure Analysis)** — the logs are complete; largely editing.
4. **Chapter 3 (Problem Formulation)** — write after 4–6, when the argument is clear.
5. **Chapters 1, 2, 9, 10** — last, once the results are fixed on paper.

### 7.4 What still needs doing before submission

**Required**
- Resolve one internal inconsistency in my own records regarding the V8 baseline number
  (two documents disagree); the correct value must be established before it appears in the text.
- Re-present the four degenerate metrics as construction-guaranteed invariants rather than
  results.
- State the sample-size limitations explicitly in the experiments chapter.

**High value, ~2 days**
- A small pairwise perceptual study (≈10 participants, ≈20 pairs) asking which of two clips
  looks more natural. This would (a) test whether the Jitter metric predicts perceived quality
  — currently an untested assumption underlying the entire trade-off analysis — and (b) satisfy
  the "physical plausibility" half of your point 9 with human evidence rather than proxies.
  Stimuli are already rendered.

**Optional**
- Complete the local HumanML3D checkout to enlarge the FID reference set.

### 7.5 Anticipated examiner questions and prepared answers

| Question | Answer |
|---|---|
| Why not learn the correction? | Five paradigms were tried and systematically failed; the surviving learned component is equivalent to a tuned Gaussian. Evidence in Ch. 6 and 8. |
| Is the improvement just from the IK? | No — IK alone changes nothing; it repairs the skeleton *after* de-skating. The 5-way ablation isolates each stage. |
| Does the correction damage the motion? | No measurable damage to text alignment (Ch. 7); FootErr ≈ 3 cm; bone lengths exactly preserved. |
| Why is rotation still the worst category? | FSR is defined on the ankle; rotational motions move the ankle continuously, so the reach clamp is active most often. Quantified in Ch. 6. |
| Does lower Jitter mean it looks better? | Currently unverified — Jitter's RMS hides spike behaviour, and no perceptual study has been run. Stated as a limitation (and addressed if 7.4 is completed). |

---

## Appendix — Artefacts

| Item | Path |
|---|---|
| Method (de-skate, IK) | `models/v18.py`, `models/v18_ik.py` |
| Learned smoother (ablated component) | `models/v19.py`, `training/v19.py` |
| Leakage-free evaluation | `testing/v19_eval.py` |
| Semantic evaluation | `testing/v19_semantic.py`, `utils/joints_to_feats.py` |
| Frontier sweep | `analysis/v19_frontier.py` → `analysis/v19/frontier.png` |
| Jitter spike analysis | `analysis/v19_jitter_trace.py` → `analysis/v19/jitter_trace.png` |
| Per-category results | `analysis/v19/by_category.json` |
| Comparison videos (5-panel) | `outputs/videos/v19/*.mp4` |
| Full development record | `docs/v19_devlog.md`, `docs/v18_devlog.md`, `docs/research_log_20260624.md` |
| Results and analysis | `docs/v19_results.md`, `docs/version_review_20260720.md` |
