# MotionFix — MSc Thesis Outline / 论文框架

> Framing: **method-contribution** (方法贡献型). The thesis' contribution is the
> pipeline + IK-in-the-loop training + a leak-free multi-generator evaluation
> protocol; the "learned smoother ≈ tuned Gaussian" result is reported honestly as
> an ablation finding, and is framed as *localising where the value is* (physics
> de-skate + IK), not as a failure.
>
> 定位=方法贡献型。贡献是管线 + IK-进-回路训练 + 无泄漏多生成器评估协议;
> 「学习≈高斯」作为诚实的消融结论,用来*定位价值所在*,不是失败。
>
> Working title: **Generator-Agnostic Post-Hoc Correction of Foot-Skating in
> Text-to-Motion Generation**
>
> Every chapter below lists the on-disk evidence it draws from. All numbers must
> come from `analysis/v19/*.json`, not memory.

---

## Contribution list (state these in the Intro, up front)

1. **A generator-agnostic, post-hoc foot-skating correction pipeline** (de-skate →
   learned smoother → 2-bone IK) that needs no retraining of the generator and
   applies unchanged to T2M-GPT, MoMask, and MDM.
2. **IK-in-the-loop training**: the differentiable ankle IK is inside the forward
   pass, so the loss is computed on the trajectory that is actually evaluated
   (matches numpy IK to 2.2e-16). This is what lets a 46K-parameter learned model
   *match* a strong classical baseline.
3. **A leak-free, multi-generator evaluation protocol** with an honest central
   finding: physics de-skate does most of the FSR reduction; the learned smoother
   is on par with a well-tuned Gaussian. Includes a corrected FID reference
   (n=26 rank-deficient → n=1215 full-rank) and a diagnosis of which metrics are
   invariant-by-construction.

> Put the negative/nuanced result in the contribution list itself. Owning it up
> front reads as rigour; letting a reviewer discover it reads as a gap.

---

## Chapter 1 — Introduction

- 1.1 Motivation: foot-skating is pervasive in text-to-motion output and is
  usually fixed per-generator (retrain / contact-aware loss) — expensive, not
  portable.
- 1.2 Problem statement: can one post-hoc, generator-agnostic module correct feet
  without retraining and without breaking text alignment?
- 1.3 Contributions (the 3 above).
- 1.4 Thesis structure.

## Chapter 2 — Background & Related Work

- 2.1 Text-to-motion generation: T2M-GPT (VQ-VAE + transformer), MoMask
  (RVQ + mask/residual transformer), MDM (diffusion). Only enough to read results.
- 2.2 HumanML3D representation: 22-joint, 263-d features, `recover_from_ric`,
  contact channels. (source: CLAUDE.md)
- 2.3 Foot-skating: definitions (FSR), classical fixes (IK, smoothing,
  contact-aware training), and why post-hoc is under-explored.
- 2.4 Evaluation metrics and their pitfalls: FSR, Jitter, FID, R-precision,
  MM-Dist, Diversity — set up §4's critique of FID reference size.

## Chapter 3 — Method: the V19 pipeline

- 3.1 Overview (three-stage figure): de-skate → learned smoother → 2-bone IK.
  Only the middle stage is neural; the other two are zero-parameter physics.
- 3.2 De-skate: plant-at-mean XZ, no drift, low FSR. (`data/prep/v19.py`)
- 3.3 Learned smoother: 46,276-param 1D-CNN, ankles only; loss design
  (threshold-aligned soft-count anti-skate term, not a mean proxy).
  (`models/v19.py::V19Loss`)
- 3.4 **IK-in-the-loop training** (headline of the method): differentiable ankle
  clamp in the forward pass; loss on the evaluated trajectory; λ auto-calibration
  by per-term gradient norm; validation-loss checkpointing.
  (`models/v19.py::torch_ankle_ik`, `training/v19.py`)
- 3.5 2-bone IK: hip→knee→ankle + rigid toe; bone-length rigidity, contact
  guarantee.
- 3.6 Delivery point `v19_088a10`: `--jit-share 0.88 --anch-share 0.10`, a
  low-jitter operating point on the frontier.

## Chapter 4 — Experimental Setup

- 4.1 Data + **no-leakage argument**: trained on ~3.4-4k HumanML3D; MoMask / MDM /
  T2M-GPT are all out-of-distribution by construction. (`training/v19.py` B7)
- 4.2 Seven physics metrics + **invariant-by-construction disclosure**: only
  FSR / Jitter / FootErr move; Floating / ContactAcc / Penetration / BoneCV are
  constant by construction — say this once, here, so results aren't over-read.
  (`testing/v19_eval.py`)
- 4.3 Semantic evaluator: HumanML3D `text_mot_match` (FID / R-prec / MM-Dist /
  Diversity), paired bootstrap.
- 4.4 **FID reference correction** (methodological highlight, own subsection):
  old reference n=26 was rank-deficient in the 512-d embedding → FID was noise;
  rebuilt from `test.txt` = 1215 GT motions (full-rank). (`testing/v19_fid_ref.py`)
- 4.5 Perceptual study design: blind 2-AFC, 12 motions × 3 comparisons, controls
  (000076 / 000119, FSR≈0), binomial test. (`outputs/perceptual/rating.html`)

## Chapter 5 — Results & Ablation

- 5.1 Main 5-way ablation × 3 generators (FSR / Jitter, n=200 for T2M-GPT/MoMask):
  **physics de-skate drives FSR down; smoothing trades a little FSR back to remove
  the jitter de-skate introduces.** (`v19_088a10_expanded_results.json`,
  `momask_pool_results.json`)
- 5.2 FSR–Jitter frontier: "learned ≈ tuned Gaussian" holds across the whole
  frontier, not at a cherry-picked point. (`frontier.json`)
- 5.3 Per-category generalisation (7 categories): every category improves, none
  regresses; rotation retains the most residual. (`by_category.json`)
- 5.4 Semantic / FID (corrected reference): de-skate hurts realism most, smoother
  repairs to ≈ original; correcting the feet does not break text alignment
  (MM-Dist / R-prec changes not significant). (`semantic_largeref.json`)
- 5.5 Jitter-spike analysis: p99 / max, spike rate; RMS alone masks twitches.
  (`jitter_stats.json` — NB: re-run for 088a10 if citing spike numbers; current
  json is v19_045.)

## Chapter 6 — Perceptual Study

- Hypothesis: "lower jitter = looks more natural" is the one untested assumption
  under the whole trade-off narrative.
- Design, binomial test, attention-check controls, results.
- **If human data not yet collected: write protocol + expected outcome, leave a
  results placeholder** so it drops in cleanly.

## Chapter 7 — Discussion & Limitations

- 7.1 What "learned ≈ Gaussian" means: value is in the pipeline + IK-in-the-loop
  training that lets a 46K model match a strong baseline while fixing jitter spikes
  and semantic drift — not a claim that learning dominates.
- 7.2 **MDM not reproducible on this cluster**: needs `--use_ema` + batch ≤ 10,
  still shows contact sliding; FID ≈ 16 confirms off-distribution. Environment
  limitation, kept at n=50, **not** used for method conclusions.
- 7.3 Data / sample-size limitations (local HumanML3D 8177/14616; only affected
  FID, now fixed).
- 7.4 Method boundary: rotation-category residual (reach-clamp triggers most).

## Chapter 8 — Conclusion & Future Work

- Recap the three contributions.
- Future: stronger learned smoother, contact-aware training signal, more
  generators, completed perceptual study.

---

## Writing-order suggestion / 建议写作顺序

3 (Method) → 4 (Setup) → 5 (Results) first — they're the most concrete and are
fully backed on disk. Then 2 (Related Work), then 1 (Intro) and 8 (Conclusion)
last, once the story is fixed. 6 (Perceptual) in parallel as data arrives.

## Open items that touch the text / 写前需确认

- [ ] Re-run `analysis/v19_jitter_trace.py` for 088a10 if citing spike numbers
      (disk json is still v19_045). (status doc §4 item 2)
- [ ] Resolve the V8-baseline conflict: CLAUDE.md "−2.9%" vs research_log
      "14.1%→15.6%". (status doc §4 item 6)
- [ ] Re-render `outputs/videos/v19/*.mp4` from 088a10 for any thesis figures
      (current renders are v19_045). (status doc §4 item 1)
- [ ] Collect perceptual human data or commit to protocol-only Chapter 6.
