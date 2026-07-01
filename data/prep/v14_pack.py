"""
Pack V14 training data into a single .pt file for fast loading.
Solves the disk I/O bottleneck of reading 15,980 individual .npy files.
"""

import torch
import numpy as np
import glob
import os
import time

DATA_DIR = "data/training/v14"
OUTPUT_FILE = "data/training/v14_packed.pt"
MAX_SAMPLES = 4000  # Use subset for fast training


def main():
    print(f"Packing V14 training data from {DATA_DIR}/")
    print(f"Output: {OUTPUT_FILE} (max {MAX_SAMPLES} samples)")
    t0 = time.time()

    distorted_files = sorted(glob.glob(f"{DATA_DIR}/distorted_*.npy"))[:MAX_SAMPLES]
    n = len(distorted_files)
    print(f"Loading {n} pairs...")

    all_distorted = []
    all_target = []

    for i, df in enumerate(distorted_files):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}")

        tf = df.replace('distorted_', 'target_')
        d = np.load(df)      # (T, 22, 3)
        t = np.load(tf)

        # Flatten to (T, 66)
        d_flat = d.reshape(d.shape[0], -1).astype(np.float32)
        t_flat = t.reshape(t.shape[0], -1).astype(np.float32)

        # Pad/truncate to max_len=196
        T = d_flat.shape[0]
        max_len = 196
        if T > max_len:
            d_flat = d_flat[:max_len]
            t_flat = t_flat[:max_len]
        elif T < max_len:
            d_flat = np.pad(d_flat, ((0, max_len - T), (0, 0)), mode='constant')
            t_flat = np.pad(t_flat, ((0, max_len - T), (0, 0)), mode='constant')

        all_distorted.append(torch.from_numpy(d_flat))
        all_target.append(torch.from_numpy(t_flat))

    # Stack into single tensors
    distorted_tensor = torch.stack(all_distorted)  # (N, 196, 66)
    target_tensor = torch.stack(all_target)        # (N, 196, 66)

    torch.save({
        'distorted': distorted_tensor,
        'target': target_tensor,
    }, OUTPUT_FILE)

    elapsed = time.time() - t0
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"  distorted: {distorted_tensor.shape}")
    print(f"  target:    {target_tensor.shape}")
    print(f"  Size:      {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")
    print(f"  Time:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
