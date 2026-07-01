# V14 — Version Log

**Date:** 2026-06-23 19:54:50
**Model:** V14 Transformer Encoder ×6, d_model=512, nhead=8, 19.1M params
**Checkpoint:** `checkpoints/v14/best.pth` (epoch 24)

## Training Paradigm

- **Input:** HumanML3D + simulated foot skating (horizontal drift at contact frames)
- **Target:** Clean HumanML3D
- **Distortion:** Only foot joints [7,8,10,11], only XZ plane, only contact frames
- **Loss:** V14Loss (foot-XZ-focused, λ_foot=3.0, λ_foot_y=0.5)

## Inference Settings

- `foot_only=True`: selective foot replacement
- `root_relative=True`: world→root-relative→world conversion
- `blend_alpha=0.5`: 50% original + 50% predicted on XZ only
- Y-axis protected: height never modified

## Results — MoMask 50 Prompts

| Metric | Value |
|--------|-------|
| **FSR** | 15.6% (was 14.1%, Δ=+1.6%) |
| **Jitter** | 0.0286 m/frame² (was 0.0128) |
| **Floating** | 0.0% |
| **Foot Error** | 0.0102 m |
| **Contact Accuracy** | 100.0% |
| **Bone Length CV** | 0.0212 |
| **Penetration (mean)** | 0.0046 m |
| **Penetration (max)** | 0.0144 m |

### Per-Motion Results

| # | Name | FSR_bef | FSR_aft | ΔFSR | Jit_bef | Jit_aft | Float | FtErr | CtAcc |
|---|------|---------|---------|------|---------|---------|-------|-------|-------|
| 1 | p000021_rotation_person_is_walking_norma | 22.5% | 26.8% | +4.2% | 0.0110 | 0.0289 | 0.0% | 0.0178 | 100.0% |
| 2 | p000818_rotation_a_person_is_spinning_in | 48.6% | 52.9% | +4.3% | 0.0452 | 0.0604 | 0.0% | 0.0167 | 100.0% |
| 3 | p001120_walking_person_is_walking_forwar | 9.9% | 12.3% | +2.5% | 0.0094 | 0.0223 | 0.0% | 0.0036 | 100.0% |
| 4 | p001168_walking_a_person_walks_forward_c | 4.0% | 3.0% | -1.0% | 0.0074 | 0.0179 | 0.0% | 0.0044 | 100.0% |
| 5 | p001448_turning_the_figure_steps_forward | 12.5% | 14.9% | +2.4% | 0.0139 | 0.0320 | 0.0% | 0.0099 | 100.0% |
| 6 | p001567_dance_a_person_throws_their_hand | 5.2% | 5.9% | +0.7% | 0.0061 | 0.0356 | 0.0% | 0.0090 | 100.0% |
| 7 | p001969_turning_a_man_walks_forward_then | 19.2% | 21.5% | +2.3% | 0.0115 | 0.0395 | 0.0% | 0.0156 | 100.0% |
| 8 | p002104_rotation_person_is_walking_in_ci | 28.4% | 33.6% | +5.2% | 0.0143 | 0.0413 | 0.0% | 0.0189 | 100.0% |
| 9 | p002530_walking_a_person_walks_towards_t | 14.0% | 16.1% | +2.1% | 0.0078 | 0.0305 | 0.0% | 0.0128 | 100.0% |
| 10 | p002606_dance_a_man_is_shadowboxing_whil | 0.0% | 0.0% | +0.0% | 0.0044 | 0.0044 | 0.0% | 0.0000 | 100.0% |
| 11 | p002627_dance_a_person_dancing_starting_ | 39.7% | 38.2% | -1.5% | 0.0175 | 0.0320 | 0.0% | 0.0161 | 100.0% |
| 12 | p003005_dance_he_does_a_salsa_dance | 6.5% | 8.0% | +1.5% | 0.0087 | 0.0205 | 0.0% | 0.0066 | 100.0% |
| 13 | p003111_dance_the_person_is_doing_a_danc | 7.0% | 9.6% | +2.6% | 0.0245 | 0.0398 | 0.0% | 0.0073 | 100.0% |
| 14 | p003424_jumping_man_jumps_twice_in_place | 0.8% | 1.6% | +0.8% | 0.0173 | 0.0195 | 0.0% | 0.0005 | 100.0% |
| 15 | p003566_rotation_person_quickly_walks_in | 21.3% | 26.6% | +5.2% | 0.0157 | 0.0382 | 0.0% | 0.0159 | 100.0% |
| 16 | p003784_rotation_a_man_walks_in_clockwis | 12.4% | 14.5% | +2.1% | 0.0108 | 0.0310 | 0.0% | 0.0105 | 100.0% |
| 17 | p004040_dance_a_man_stands_and_brings_bo | 3.3% | 2.9% | -0.4% | 0.0225 | 0.0346 | 0.0% | 0.0042 | 100.0% |
| 18 | p004311_complex_a_person_stumbles_around | 21.3% | 21.6% | +0.3% | 0.0114 | 0.0288 | 0.0% | 0.0169 | 100.0% |
| 19 | p004488_turning_a_person_walks_forward_a | 12.2% | 14.4% | +2.1% | 0.0097 | 0.0237 | 0.0% | 0.0108 | 100.0% |
| 20 | p004822_walking_a_person_is_walking_in_p | 40.3% | 22.9% | -17.4% | 0.0104 | 0.0379 | 0.0% | 0.0234 | 100.0% |
| 21 | p004965_complex_a_person_walks_forward_t | 2.7% | 3.2% | +0.5% | 0.0054 | 0.0119 | 0.0% | 0.0030 | 100.0% |
| 22 | p006652_jumping_the_man_is_doing_star_ju | 0.0% | 0.0% | +0.0% | 0.0080 | 0.0091 | 0.0% | 0.0004 | 100.0% |
| 23 | p006658_backward_the_person_walks_backwa | 11.1% | 12.1% | +1.0% | 0.0082 | 0.0291 | 0.0% | 0.0116 | 100.0% |
| 24 | p006701_turning_a_person_walks_forward_t | 12.1% | 14.6% | +2.5% | 0.0075 | 0.0285 | 0.0% | 0.0122 | 100.0% |
| 25 | p007561_complex_a_man_walks_in_a_curve | 21.9% | 26.3% | +4.4% | 0.0144 | 0.0380 | 0.0% | 0.0164 | 100.0% |
| 26 | p007767_dance_a_person_slowly_kicks_with | 18.8% | 21.6% | +2.9% | 0.0198 | 0.0417 | 0.0% | 0.0127 | 100.0% |
| 27 | p008292_turning_a_person_turns_to_his_ri | 12.5% | 15.6% | +3.1% | 0.0111 | 0.0281 | 0.0% | 0.0130 | 100.0% |
| 28 | p008382_jumping_squatting_motion_for_exe | 0.0% | 0.0% | +0.0% | 0.0038 | 0.0124 | 0.0% | 0.0004 | 100.0% |
| 29 | p008463_complex_a_man_walks_forward_then | 12.3% | 15.1% | +2.8% | 0.0111 | 0.0292 | 0.0% | 0.0120 | 100.0% |
| 30 | p009161_walking_a_person_walks_forward_f | 21.4% | 26.2% | +4.8% | 0.0127 | 0.0219 | 0.0% | 0.0044 | 100.0% |
| 31 | p009377_complex_the_person_is_walking_fo | 15.8% | 18.4% | +2.6% | 0.0090 | 0.0248 | 0.0% | 0.0104 | 100.0% |
| 32 | p009539_walking_the_person_was_walking_f | 5.1% | 7.7% | +2.6% | 0.0085 | 0.0103 | 0.0% | 0.0010 | 100.0% |
| 33 | p009613_backward_the_man_runs_backwards | 28.8% | 33.1% | +4.3% | 0.0280 | 0.0557 | 0.0% | 0.0200 | 100.0% |
| 34 | p009958_backward_a_person_walks_backward | 52.1% | 56.7% | +4.5% | 0.0115 | 0.0373 | 0.0% | 0.0386 | 100.0% |
| 35 | p010665_walking_a_person_takes_a_few_ste | 9.5% | 11.4% | +2.0% | 0.0081 | 0.0272 | 0.0% | 0.0084 | 100.0% |
| 36 | p010819_turning_the_person_is_walking_fo | 16.9% | 20.8% | +3.9% | 0.0130 | 0.0305 | 0.0% | 0.0152 | 100.0% |
| 37 | p011028_jumping_a_man_jumps_then_kicks_t | 14.0% | 15.8% | +1.9% | 0.0241 | 0.0355 | 0.0% | 0.0108 | 100.0% |
| 38 | p011441_complex_a_person_walks_forward_i | 19.2% | 21.5% | +2.3% | 0.0125 | 0.0412 | 0.0% | 0.0165 | 100.0% |
| 39 | p011673_jumping_subject_sits_flat_on_the | 1.4% | 0.7% | -0.7% | 0.0065 | 0.0192 | 0.0% | 0.0034 | 100.0% |
| 40 | p011684_jumping_a_person_jumps_up_and_th | 2.2% | 2.8% | +0.6% | 0.0138 | 0.0160 | 0.0% | 0.0006 | 100.0% |
| 41 | p011978_turning_a_person_walked_forward_ | 20.3% | 24.4% | +4.1% | 0.0147 | 0.0358 | 0.0% | 0.0144 | 100.0% |
| 42 | p011997_complex_a_person_walks_up_some_s | 12.7% | 14.1% | +1.4% | 0.0086 | 0.0112 | 0.0% | 0.0018 | 100.0% |
| 43 | p012309_jumping_someone_jumps_around_tak | 10.6% | 13.6% | +2.9% | 0.0170 | 0.0301 | 0.0% | 0.0080 | 100.0% |
| 44 | p012498_complex_pretend_to_hold_a_ball_i | 17.6% | 17.6% | +0.0% | 0.0200 | 0.0368 | 0.0% | 0.0166 | 100.0% |
| 45 | p012529_rotation_a_person_is_dancing_the | 19.3% | 21.2% | +1.9% | 0.0097 | 0.0332 | 0.0% | 0.0201 | 100.0% |
| 46 | p012657_dance_kick_left_leg_step_back | 2.0% | 4.1% | +2.0% | 0.0202 | 0.0466 | 0.0% | 0.0039 | 100.0% |
| 47 | p012805_complex_a_man_crouches_down_whil | 2.3% | 2.3% | +0.0% | 0.0099 | 0.0166 | 0.0% | 0.0016 | 100.0% |
| 48 | p012883_dance_person_is_working_on_their | 3.9% | 4.4% | +0.6% | 0.0050 | 0.0212 | 0.0% | 0.0050 | 100.0% |
| 49 | p013207_complex_a_person_sprinting_ahead | 6.9% | 9.7% | +2.9% | 0.0180 | 0.0281 | 0.0% | 0.0061 | 100.0% |
| 50 | p014457_dance_the_person_swings_a_golf_c | 0.0% | 0.0% | +0.0% | 0.0027 | 0.0027 | 0.0% | 0.0000 | 100.0% |

**Average** | | 14.1% | 15.6% | +1.6% | 0.0128 | 0.0286 | 0.0% | 0.0102 | 100.0% |
