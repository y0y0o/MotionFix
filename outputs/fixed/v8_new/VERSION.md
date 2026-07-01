# V8_new — Version Log

**Date:** 2026-06-23
**Model:** V8 (Transformer Encoder ×6, d_model=512, nhead=8, 19.1M params)
**Checkpoint:** `checkpoints/v8/best.pth` (epoch 50, train_loss=0.0114)

## Changes from V8 baseline

### Fix 1: Root-Relative Coordinate Conversion
- **Problem:** Training data is root-relative (X range [-0.34, 0.32]), but MoMask output is world-coordinate (X range [-3.6, 0.3]). Model maps any input to training coordinate range → output squashed to origin.
- **Fix:** `forward()` now accepts `root_relative=True` (default). Before Transformer: subtract pelvis (joint 0) from all joints. After Transformer: add pelvis back. Uses `repeat(1,1,22)` for correct joint-interleaved tiling.
- **Code:** `models/v8.py:69-87` (`_to_root_relative`, `_from_root_relative`)

### Fix 2: Y-Axis Protection in _selective_replace
- **Problem:** Model's full_output predicts foot Y ≈ 0.95m (pelvis height) for all frames. Blending Y during skating correction pulled feet up to knee height → visible knee-jumping artifact.
- **Fix:** In `_selective_replace`, when foot is on ground and skating is detected, only blend X and Z (horizontal plane). Y (height) `continue` — kept as original.
- **Code:** `models/v8.py:168-170` (`if d == y_dim: continue`)

### Bug Fixed: repeat_interleave → repeat
- **Bug:** `pelvis.repeat_interleave(22, dim=2)` gives `[px×22, py×22, pz×22]` instead of `[px,py,pz, px,py,pz, ...]`. Caused 2/3 of coordinate subtractions to use wrong pelvis component.
- **Fix:** Changed to `pelvis.repeat(1, 1, 22)` (correct joint-interleaved layout).

## Inference Settings
- `foot_only=True`: selective foot replacement
- `root_relative=True`: coordinate space conversion
- `blend_alpha=0.5`: 50% original + 50% predicted blend on XZ only
- Skating detection: foot height < ground+5cm AND horizontal velocity > 0.03 m/frame

## Results — MoMask 50 Prompts

| Metric | Original | V8_new | Change |
|--------|----------|--------|--------|
| **Avg FSR** | 14.1% | 15.6% | +1.5pp |
| **Avg Jitter** | 0.0128 | 0.0278 | +2.2× |
| **Avg FootErr** | — | 0.0098m | — |

### Per-motion results

| # | Name | FSR_bef | FSR_aft | ΔFSR | Jit_bef | Jit_aft | FtErr |
|---|------|---------|---------|------|---------|---------|-------|
| 1 | p000021_rotation_person_is_walking_normally_in_ | 22.5% | 26.8% | +4.2% | 0.0110 | 0.0283 | 0.0174 |
| 2 | p000818_rotation_a_person_is_spinning_in_a | 48.6% | 53.6% | +5.0% | 0.0452 | 0.0604 | 0.0165 |
| 3 | p001120_walking_person_is_walking_forward | 9.9% | 12.3% | +2.5% | 0.0094 | 0.0206 | 0.0033 |
| 4 | p001168_walking_a_person_walks_forward_casually | 4.0% | 3.0% | -1.0% | 0.0074 | 0.0175 | 0.0042 |
| 5 | p001448_turning_the_figure_steps_forward_then_t | 12.5% | 15.5% | +3.0% | 0.0139 | 0.0314 | 0.0098 |
| 6 | p001567_dance_a_person_throws_their_hands_outwa | 5.2% | 5.9% | +0.7% | 0.0061 | 0.0341 | 0.0087 |
| 7 | p001969_turning_a_man_walks_forward_then_turns | 19.2% | 21.5% | +2.3% | 0.0115 | 0.0387 | 0.0155 |
| 8 | p002104_rotation_person_is_walking_in_circles | 28.4% | 33.6% | +5.2% | 0.0143 | 0.0410 | 0.0185 |
| 9 | p002530_walking_a_person_walks_towards_the_came | 14.0% | 16.1% | +2.1% | 0.0078 | 0.0277 | 0.0114 |
| 10 | p002606_dance_a_man_is_shadowboxing_while_stand | 0.0% | 0.0% | +0.0% | 0.0044 | 0.0044 | 0.0000 |
| 11 | p002627_dance_a_person_dancing_starting_in_a | 39.7% | 38.2% | -1.5% | 0.0175 | 0.0319 | 0.0161 |
| 12 | p003005_dance_he_does_a_salsa_dance | 6.5% | 8.0% | +1.5% | 0.0087 | 0.0201 | 0.0066 |
| 13 | p003111_dance_the_person_is_doing_a_dance | 7.0% | 9.6% | +2.6% | 0.0245 | 0.0395 | 0.0072 |
| 14 | p003424_jumping_man_jumps_twice_in_place | 0.8% | 1.6% | +0.8% | 0.0173 | 0.0188 | 0.0005 |
| 15 | p003566_rotation_person_quickly_walks_in_a_cloc | 21.3% | 26.6% | +5.2% | 0.0157 | 0.0378 | 0.0157 |
| 16 | p003784_rotation_a_man_walks_in_clockwise_direc | 12.4% | 14.5% | +2.1% | 0.0108 | 0.0306 | 0.0102 |
| 17 | p004040_dance_a_man_stands_and_brings_both | 3.3% | 2.9% | -0.4% | 0.0225 | 0.0337 | 0.0041 |
| 18 | p004311_complex_a_person_stumbles_around_like_t | 21.3% | 21.9% | +0.6% | 0.0114 | 0.0281 | 0.0166 |
| 19 | p004488_turning_a_person_walks_forward_at_a | 12.2% | 14.4% | +2.1% | 0.0097 | 0.0227 | 0.0103 |
| 20 | p004822_walking_a_person_is_walking_in_place | 40.3% | 21.3% | -19.0% | 0.0104 | 0.0367 | 0.0224 |
| 21 | p004965_complex_a_person_walks_forward_to_the | 2.7% | 3.2% | +0.5% | 0.0054 | 0.0108 | 0.0026 |
| 22 | p006652_jumping_the_man_is_doing_star_jumps | 0.0% | 0.0% | +0.0% | 0.0080 | 0.0088 | 0.0003 |
| 23 | p006658_backward_the_person_walks_backwards_in_ | 11.1% | 12.1% | +1.0% | 0.0082 | 0.0277 | 0.0104 |
| 24 | p006701_turning_a_person_walks_forward_then_tur | 12.1% | 14.6% | +2.5% | 0.0075 | 0.0272 | 0.0113 |
| 25 | p007561_complex_a_man_walks_in_a_curve | 21.9% | 26.3% | +4.4% | 0.0144 | 0.0377 | 0.0164 |
| 26 | p007767_dance_a_person_slowly_kicks_with_their | 18.8% | 21.6% | +2.9% | 0.0198 | 0.0411 | 0.0125 |
| 27 | p008292_turning_a_person_turns_to_his_right | 12.5% | 15.6% | +3.1% | 0.0111 | 0.0275 | 0.0127 |
| 28 | p008382_jumping_squatting_motion_for_exercise_w | 0.0% | 0.0% | +0.0% | 0.0038 | 0.0125 | 0.0004 |
| 29 | p008463_complex_a_man_walks_forward_then_squats | 12.3% | 15.1% | +2.8% | 0.0111 | 0.0287 | 0.0119 |
| 30 | p009161_walking_a_person_walks_forward_from_one | 21.4% | 26.2% | +4.8% | 0.0127 | 0.0213 | 0.0041 |
| 31 | p009377_complex_the_person_is_walking_forward_a | 15.8% | 17.7% | +1.9% | 0.0090 | 0.0232 | 0.0097 |
| 32 | p009539_walking_the_person_was_walking_forward_ | 5.1% | 7.7% | +2.6% | 0.0085 | 0.0097 | 0.0008 |
| 33 | p009613_backward_the_man_runs_backwards | 28.8% | 32.4% | +3.6% | 0.0280 | 0.0536 | 0.0185 |
| 34 | p009958_backward_a_person_walks_backward_in_a | 52.1% | 56.7% | +4.5% | 0.0115 | 0.0359 | 0.0378 |
| 35 | p010665_walking_a_person_takes_a_few_steps | 9.5% | 11.4% | +2.0% | 0.0081 | 0.0256 | 0.0076 |
| 36 | p010819_turning_the_person_is_walking_forward_a | 16.9% | 20.8% | +3.9% | 0.0130 | 0.0298 | 0.0144 |
| 37 | p011028_jumping_a_man_jumps_then_kicks_the | 14.0% | 15.8% | +1.9% | 0.0241 | 0.0347 | 0.0105 |
| 38 | p011441_complex_a_person_walks_forward_in_a | 19.2% | 21.5% | +2.3% | 0.0125 | 0.0396 | 0.0161 |
| 39 | p011673_jumping_subject_sits_flat_on_the_ground | 1.4% | 0.7% | -0.7% | 0.0065 | 0.0189 | 0.0033 |
| 40 | p011684_jumping_a_person_jumps_up_and_then | 2.2% | 2.8% | +0.6% | 0.0138 | 0.0155 | 0.0005 |
| 41 | p011978_turning_a_person_walked_forward_by_chan | 20.3% | 24.0% | +3.7% | 0.0147 | 0.0357 | 0.0143 |
| 42 | p011997_complex_a_person_walks_up_some_stairs | 12.7% | 14.1% | +1.4% | 0.0086 | 0.0109 | 0.0016 |
| 43 | p012309_jumping_someone_jumps_around_takes_four | 10.6% | 13.6% | +2.9% | 0.0170 | 0.0299 | 0.0080 |
| 44 | p012498_complex_pretend_to_hold_a_ball_in | 17.6% | 17.9% | +0.3% | 0.0200 | 0.0351 | 0.0156 |
| 45 | p012529_rotation_a_person_is_dancing_the_waltz | 19.3% | 20.9% | +1.6% | 0.0097 | 0.0314 | 0.0182 |
| 46 | p012657_dance_kick_left_leg_step_back | 2.0% | 4.1% | +2.0% | 0.0202 | 0.0461 | 0.0039 |
| 47 | p012805_complex_a_man_crouches_down_while_quick | 2.3% | 2.3% | +0.0% | 0.0099 | 0.0164 | 0.0016 |
| 48 | p012883_dance_person_is_working_on_their_boxing | 3.9% | 4.4% | +0.6% | 0.0050 | 0.0196 | 0.0046 |
| 49 | p013207_complex_a_person_sprinting_ahead_and_th | 6.9% | 9.1% | +2.3% | 0.0180 | 0.0271 | 0.0056 |
| 50 | p014457_dance_the_person_swings_a_golf_club | 0.0% | 0.0% | +0.0% | 0.0027 | 0.0027 | 0.0000 |
| **AVG** | | **14.1%** | **15.6%** | **+1.5%** | **0.0128** | **0.0278** | **0.0098** |

## Model Code Changes (`models/v8.py`)
- `_to_root_relative()` — convert world → root-relative (subtract pelvis)
- `_from_root_relative()` — reverse conversion
- `forward(root_relative=True)` — enable coordinate conversion at inference
- `_selective_replace()` — Y-axis skip during skating correction

## Test Scripts
- `testing/v8_new.py` — batch test on MoMask 50
- `analysis/test_root_relative_fix.py` — A/B comparison script
- `analysis/training_vs_inference_gap.py` — training/inference distribution analysis

## Analysis Reports
- `analysis/v8_vs_v13/root_cause_report.md` — root cause: coordinate system mismatch
- `analysis/v8_vs_v13/comparison_report.md` — V8 vs V13 comparison
- `analysis/root_relative_fix/` — A/B test results + visualizations

## Checksum
MD5 of all 50 output files: `909c9124bccc6435672040061e6df6a8`
