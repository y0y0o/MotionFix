"""
Prepare V15 Training Data: MoMask → Physics-Fixed Teacher
===========================================================
Physics correction (damp=0.0) provides the "correct answer" for
each MoMask output. The model learns to replicate this correction.

Pipeline:
  1. Load 50 MoMask motions (world coords)
  2. Apply physics fix (damp=0.0) → teacher target
  3. Split: 40 train, 10 test (stratified by FSR)
  4. Augment training data (noise variants → 200 pairs)
  5. Convert to root-relative for model training
  6. Save as (distorted, target) pairs (both root-relative)
"""

import numpy as np
import glob, os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.physics_fix import physics_foot_fix
from utils.metrics import compute_fsr

INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
OUTPUT_DIR = "data/training/v15"
TEST_LIST = "data/training/v15/test_names.json"
TRAIN_RATIO = 0.8
AUGMENT_PER_SAMPLE = 5  # Each MoMask → 5 variants → 200 train pairs
os.makedirs(OUTPUT_DIR, exist_ok=True)


def to_root_relative(motion):
    """Convert world coords → root-relative by subtracting pelvis (joint 0)."""
    pelvis = motion[:, 0:1, :].copy()
    return motion - pelvis


def augment_motion(motion, seed=0):
    """Create a slight variation: temporal noise + small spatial jitter."""
    rng = np.random.RandomState(seed)
    T = motion.shape[0]
    aug = motion.copy()

    # Tiny Gaussian noise (only on non-foot joints simulates VQ artifacts)
    noise = rng.normal(0, 0.002, motion.shape)
    aug += noise

    # Slight temporal shift (roll 1-2 frames)
    shift = rng.randint(-2, 3)
    if shift != 0:
        aug = np.roll(aug, shift, axis=0)
        if shift > 0:
            aug[:shift] = motion[:shift]
        else:
            aug[shift:] = motion[shift:]

    return aug


def main():
    print("=" * 70)
    print("  V15 — Physics-Teacher Training Data")
    print("=" * 70)

    files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))
    print(f"  Source: {len(files)} MoMask motions")

    # ── Phase 1: Apply physics fix to all motions ──
    print("\n  Phase 1: Applying physics correction (damp=0.0)...")
    all_data = []
    for i, fp in enumerate(files):
        name = os.path.basename(fp).replace('.npy', '')
        motion = np.load(fp).astype(np.float32)
        if motion.ndim == 4:
            motion = motion[0]

        fixed, stats = physics_foot_fix(motion, damp_factor=0.0, return_stats=True)
        fsr_orig = compute_fsr(motion)[0]
        fsr_fixed = compute_fsr(fixed)[0]

        all_data.append({
            'name': name,
            'original': motion,      # world coords (with skating)
            'target': fixed,          # world coords (physics-corrected)
            'fsr_orig': fsr_orig,
            'fsr_fixed': fsr_fixed,
        })

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(files)}")

    print(f"    Done. FSR: {np.mean([d['fsr_orig'] for d in all_data]):.1%} → "
          f"{np.mean([d['fsr_fixed'] for d in all_data]):.1%}")

    # ── Phase 2: Stratified split (40 train / 10 test) ──
    print("\n  Phase 2: Stratified train/test split...")

    # Sort by FSR improvement
    all_data.sort(key=lambda d: d['fsr_fixed'] - d['fsr_orig'])

    # Interleave for stratification: take every Nth for test
    test_indices = []
    step = len(all_data) // 10
    for i in range(10):
        test_indices.append(i * step + step // 2)

    train_data = [d for i, d in enumerate(all_data) if i not in test_indices]
    test_data = [d for i, d in enumerate(all_data) if i in test_indices]

    # Save test names
    test_names = [d['name'] for d in test_data]
    with open(TEST_LIST, 'w') as f:
        json.dump(test_names, f, indent=2)

    print(f"    Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"    Test names → {TEST_LIST}")

    # ── Phase 3: Augment and save training pairs ──
    print(f"\n  Phase 3: Augmenting ({AUGMENT_PER_SAMPLE}× per sample → "
          f"~{len(train_data)*AUGMENT_PER_SAMPLE} pairs)...")

    pair_count = 0
    for d_idx, d in enumerate(train_data):
        orig_world = d['original']
        targ_world = d['target']

        for aug_idx in range(AUGMENT_PER_SAMPLE):
            seed = d_idx * 1000 + aug_idx * 137

            if aug_idx == 0:
                # Pure version: no augmentation
                input_world = orig_world
                target_world = targ_world
            else:
                # Augmented: MoMask + slight noise → physics-fixed (clean)
                input_world = augment_motion(orig_world, seed=seed)
                # Target stays the physics-fixed version
                target_world = targ_world

            # Convert both to root-relative for training
            input_rr = to_root_relative(input_world)
            target_rr = to_root_relative(target_world)

            # Flatten for saving
            input_flat = input_rr.reshape(input_rr.shape[0], -1).astype(np.float32)
            target_flat = target_rr.reshape(target_rr.shape[0], -1).astype(np.float32)

            np.save(f"{OUTPUT_DIR}/distorted_{pair_count:06d}.npy", input_flat)
            np.save(f"{OUTPUT_DIR}/target_{pair_count:06d}.npy", target_flat)
            pair_count += 1

    print(f"    Saved {pair_count} training pairs → {OUTPUT_DIR}/")

    # ── Phase 4: Save test data as packed .pt file ──
    print("\n  Phase 4: Saving test data...")

    test_pairs = []
    for d in test_data:
        orig_rr = to_root_relative(d['original'])
        targ_rr = to_root_relative(d['target'])
        test_pairs.append({
            'name': d['name'],
            'original_world': d['original'],
            'target_world': d['target'],
            'original_rr': orig_rr,
            'target_rr': targ_rr,
            'fsr_orig': float(d['fsr_orig']),
            'fsr_fixed': float(d['fsr_fixed']),
        })

    # Save as pickle for easy loading
    import pickle
    with open(f"{OUTPUT_DIR}/test_data.pkl", 'wb') as f:
        pickle.dump(test_pairs, f)

    print(f"    Saved {len(test_pairs)} test motions → {OUTPUT_DIR}/test_data.pkl")

    # ── Summary ──
    print(f"\n{'─'*70}")
    print(f"  📊 V15 Training Data Summary")
    print(f"{'─'*70}")
    print(f"  Train pairs:  {pair_count} ({len(train_data)} base × {AUGMENT_PER_SAMPLE})")
    print(f"  Test motions: {len(test_data)} (held out)")
    print(f"")
    print(f"  Train FSR range: {min(d['fsr_orig'] for d in train_data):.1%} — "
          f"{max(d['fsr_orig'] for d in train_data):.1%}")
    print(f"  Test FSR range:  {min(d['fsr_orig'] for d in test_data):.1%} — "
          f"{max(d['fsr_orig'] for d in test_data):.1%}")
    print(f"")
    print(f"  Format: root-relative (T,66) flat arrays")
    print(f"  Input → Target: MoMask (skating) → Physics-fixed (corrected)")
    print(f"\n  💾 {OUTPUT_DIR}/")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
