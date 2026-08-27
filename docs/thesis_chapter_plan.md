# MotionFix — Chapter Plan & INSIGHT Collection
> Output of `academic-paper` **plan** mode (Socratic chapter-by-chapter planning).
> Framing: method-contribution. Target: MSc thesis, ~15,000 words. Chapter 6
> (perceptual study) is an **optional module** — becomes a full chapter if human
> data arrives, else folds into Discussion/Future Work.
> Every claim below is anchored to on-disk evidence; no number is from memory.

---

## INSIGHT Collection

- **[INSIGHT: research_gap]** Existing foot-skating fixes either retrain the
  generator or are single-model physics post-processing; there is no
  *generator-agnostic, retrain-free* corrector that also preserves text
  alignment, and no honest measurement of whether a *learned* smoother beats a
  well-tuned classical one.

- **[INSIGHT: thesis_statement]** (user-endorsed) A generator-agnostic, post-hoc
  pipeline (de-skate → learned smoother → IK), trained leak-free and evaluated
  across three generators, corrects foot-skating without breaking text alignment;
  a controlled ablation localises the correction to the physics stages and shows a
  46K learned smoother only matches a well-tuned Gaussian.

- **[INSIGHT: contribution_claim]** Three contributions: (1) the generator-agnostic
  post-hoc pipeline; (2) IK-in-the-loop training of the smoother; (3) a leak-free
  multi-generator evaluation protocol with an honest central finding (physics does
  most of the work; learned ≈ tuned Gaussian) and a corrected FID reference.

- **[INSIGHT: open_question]** Whether "lower jitter = looks more natural" holds
  perceptually is not yet verified (Chapter 6 module).

---

## Word budget (~15,000 words)

| Ch | Title | Words | Core argument (1 sentence) |
|----|-------|------:|----------------------------|
| — | Abstract (bilingual) | 250 | — |
| 1 | Introduction | 1,600 | Foot-skating is pervasive and fixed per-generator; a generator-agnostic post-hoc corrector is the gap we fill. |
| 2 | Background & Related Work | 2,400 | Generators, the HumanML3D representation, and prior de-skating fixes leave post-hoc correction under-explored and its metrics mis-used. |
| 3 | Method: the V19 pipeline | 2,900 | Three stages — physics de-skate, a 46K learned smoother, 2-bone IK — with IK inside the training loop so loss lands on the evaluated trajectory. |
| 4 | Experimental Setup | 1,900 | A leak-free, multi-generator protocol with 7 metrics (only 3 non-trivial), a fixed FID reference, and a blind perceptual design. |
| 5 | Results & Ablation | 2,900 | Physics de-skate drives FSR down; smoothing trades a little back to kill jitter; learned ≈ tuned Gaussian; feet-fixing does not break text alignment. |
| 6 | Perceptual Study *(optional module)* | 1,000 | Blind 2-AFC tests the one unverified assumption behind the trade-off narrative. |
| 7 | Discussion & Limitations | 1,500 | The value is the pipeline + protocol, not the learned component; MDM and perceptual data are honest limits. |
| 8 | Conclusion & Future Work | 800 | A small, honest, portable corrector; future work strengthens the learned stage and completes the perceptual test. |

If Chapter 6 is omitted: +300 words to §7.4 as protocol-ready future work; total ~14,300.

---

## Chapter 1 — Introduction (~1,600)

- **Core argument:** the field fixes feet per-generator; a generator-agnostic
  post-hoc module is missing and worth building.
- **Key moves:** motivate foot-skating → state the gap (INSIGHT above) → the 3
  contributions **including** the honest learned≈Gaussian finding, stated up front
  → thesis structure.
- **Evidence:** `docs/thesis_outline.md` §1; CLAUDE.md representation notes.
- **Weakest point / defense:** a reviewer may say "post-hoc is trivial" — pre-empt
  by naming leak-free multi-generator OOD + IK-in-the-loop as the non-trivial parts.

## Chapter 2 — Background & Related Work (~2,400)

- **Core argument:** three generators + HumanML3D representation + the metric
  landscape set up why post-hoc correction is open and why FID is easy to misuse.
- **Sub-stories (3):** (a) text-to-motion generators (T2M-GPT / MoMask / MDM);
  (b) HumanML3D 22-joint / 263-d / `recover_from_ric`; (c) foot-skating metrics &
  fixes (IK, smoothing, contact-aware training).
- **Evidence:** `docs/motionfix_architecture_and_data.md`; CLAUDE.md.
- **Lands on:** post-hoc, generator-agnostic correction is under-explored → §3.

## Chapter 3 — Method: the V19 pipeline (~2,900)

- **Core argument:** a three-stage corrector where only the middle stage is
  neural, and IK-in-the-loop is the technical novelty.
- **Sections:** 3.1 overview figure · 3.2 de-skate (plant-at-mean XZ) · 3.3
  learned smoother (46,276-param 1D-CNN, ankles only, threshold-aligned soft-count
  loss) · **3.4 IK-in-the-loop** (differentiable ankle clamp, matches numpy IK to
  2.2e-16, λ auto-calibration, val-loss checkpointing) · 3.5 2-bone IK · 3.6
  delivery point 088a10 (`--jit-share 0.88 --anch-share 0.10`).
- **Evidence:** `models/v19.py`, `training/v19.py`, `data/prep/v19.py`,
  `docs/motionfix_v19_full_spec.md`.
- **Weakest point / defense:** "why not just Gaussian?" — answer belongs in §5, but
  flag here that the pipeline is designed so the learned stage is a drop-in the
  ablation can isolate.

## Chapter 4 — Experimental Setup (~1,900)

- **Core argument:** the evaluation is leak-free, multi-generator, and its metrics
  are stated honestly.
- **Sections:** 4.1 data + no-leakage (4k train → all generators OOD) · 4.2 seven
  metrics + **invariant-by-construction disclosure** (only FSR/Jitter/FootErr move)
  · 4.3 semantic evaluator (text_mot_match, paired bootstrap) · **4.4 FID reference
  correction** (n=26 rank-deficient → n=1215 full-rank; own subsection) · 4.5 blind
  2-AFC perceptual design.
- **Evidence:** `testing/v19_eval.py`, `testing/v19_fid_ref.py`,
  `analysis/v19/semantic_largeref.json`, `outputs/perceptual/rating.html`.

## Chapter 5 — Results & Ablation (~2,900)

- **Core argument:** physics does most of the FSR reduction; the learned smoother
  matches a tuned Gaussian; correcting feet does not break text alignment.
- **Sections & evidence:**
  - 5.1 5-way ablation ×3 generators (FSR/Jitter, n=200 for T2M-GPT/MoMask) →
    `v19_088a10_expanded_results.json`, `momask_pool_results.json`
  - 5.2 FSR–Jitter frontier → `frontier.json`
  - 5.3 per-category generalisation (7 cats, none regress; rotation worst) →
    `by_category.json`
  - 5.4 semantic / FID (corrected ref: de-skate hurts most, smoother repairs;
    MM-Dist/R-prec not significant) → `semantic_largeref.json`
  - 5.5 jitter spikes (p99/max) → `jitter_stats.json` **[re-run for 088a10 first]**
- **Unexpected result to explain:** learned ≈ Gaussian — frame as *localisation*,
  not failure.
- **Counter-evidence to state:** MDM FID ≈ 16 (off-distribution) → §7.

## Chapter 6 — Perceptual Study *(optional module, ~1,000)*

- **Core argument:** the trade-off narrative rests on "lower jitter looks more
  natural"; a blind 2-AFC tests exactly that.
- **Sections:** hypothesis · design (12 motions × 3 comparisons, controls
  000076/000119) · binomial test · results **or** protocol + expected outcome +
  placeholder.
- **Evidence:** `outputs/perceptual/rating.html`, `analysis/perceptual_analysis.py`.

## Chapter 7 — Discussion & Limitations (~1,500)

- **Core argument:** the contribution is the pipeline + protocol; learned≈Gaussian
  tells us *where* the value is.
- **Sections:** 7.1 what learned≈Gaussian means · 7.2 MDM not reproducible
  (env limit, FID≈16, kept at n=50, not used for method conclusions) · 7.3
  data/sample limits (local HumanML3D 8177/14616; only affected FID, now fixed) ·
  7.4 method boundary (rotation residual) [+ perceptual as future work if Ch6 cut].
- **Evidence:** `supervisor_report_20260822.md` §5; memory `mdm-generation-gotchas`.

## Chapter 8 — Conclusion & Future Work (~800)

- **Core argument:** a small (46K), honest, portable foot corrector; recap 3
  contributions; future = stronger learned stage, contact-aware signal, more
  generators, completed perceptual study.

---

## Argument stress test (Step 3)

| Weakest point | Reviewer attack | Defense in the plan |
|---|---|---|
| Learned ≈ Gaussian | "So the ML part is pointless." | Reframed as the finding (§5.1/5.2 frontier shows it across all operating points, not cherry-picked); value = pipeline + IK-in-the-loop + protocol. Owned in Intro contribution list. |
| Only 3 of 7 metrics move | "You padded the metric table." | §4.2 discloses invariant-by-construction up front; results only discuss FSR/Jitter/FootErr. |
| MDM off-distribution | "Your third generator is broken." | §7.2 documents as environment limitation; MDM excluded from method conclusions, main tables use T2M-GPT/MoMask n=200. |
| No human validation yet | "Jitter↓ ≠ better-looking." | Ch6 module tests it directly; if unshipped, stated as the one open assumption in §7.4, not hidden. |
| Reverse the argument | "If smoothing helps, drop de-skate." | §5.1 shows de-skate alone does most FSR reduction and smoothing *raises* FSR — the order is load-bearing, not arbitrary. |

## Pre-writing blockers (resolve before drafting §5/§7)

- [ ] Re-run `analysis/v19_jitter_trace.py` for 088a10 (disk json is v19_045).
- [ ] Resolve V8-baseline conflict: CLAUDE.md −2.9% vs research_log 14.1%→15.6%.
- [ ] Re-render `outputs/videos/v19/*.mp4` from 088a10 for figures.
- [ ] Decide Chapter 6: collect perceptual data or ship protocol-only.

---

## Next step

Chapter Plan complete. To draft, switch mode:
- **`/ars-outline`** — expand this into a section-level outline + evidence map, or
- **`/ars-full`** — draft the full paper chapter by chapter (needs Phase 0 config:
  paper type, citation format e.g. IEEE, output format e.g. LaTeX).
- **`/ars-abstract`** — bilingual abstract once the draft exists.
Recommended: draft Ch3 (Method) or Ch5 (Results) first — most concrete, fully backed.
