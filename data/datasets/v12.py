"""
MotionFix V12 - Dataset

FRDM-style data format: distorted_*.npy + target_*.npy + contact_*.npy (triplets)
Contact labels are MANDATORY for V12 — used for contact-gated loss.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os


class MotionFixDatasetV12(Dataset):
    def __init__(self, data_dir, max_len=196):
        self.data_dir = data_dir
        self.max_len = max_len

        self.distorted_files = sorted(
            glob.glob(f"{data_dir}/distorted_*.npy")
        )

        # Verify contact files exist
        if self.distorted_files:
            first_contact = self.distorted_files[0].replace('distorted_', 'contact_')
            if not os.path.exists(first_contact):
                raise RuntimeError(
                    f"V12 requires contact labels! Missing: {first_contact}\n"
                    f"Run prepare_data_v12.py first."
                )

        print(f"Dataset: {len(self.distorted_files)} pairs from {data_dir}")
        print(f"  max_len={max_len}, contact_labels=YES (mandatory for V12)")

    def __len__(self):
        return len(self.distorted_files)

    def __getitem__(self, idx):
        distorted_path = self.distorted_files[idx]
        target_path = distorted_path.replace('distorted_', 'target_')
        contact_path = distorted_path.replace('distorted_', 'contact_')

        distorted = np.load(distorted_path).astype(np.float32)   # (T, 22, 3)
        target = np.load(target_path).astype(np.float32)         # (T, 22, 3)
        contact = np.load(contact_path).astype(np.float32)       # (T, 2)

        T = distorted.shape[0]

        # Flatten joint positions
        distorted = distorted.reshape(T, -1)   # (T, 66)
        target = target.reshape(T, -1)         # (T, 66)

        # Pad or truncate
        if T > self.max_len:
            distorted = distorted[:self.max_len]
            target = target[:self.max_len]
            contact = contact[:self.max_len]
        elif T < self.max_len:
            pad_len = self.max_len - T
            distorted = np.pad(distorted, ((0, pad_len), (0, 0)), mode='constant')
            target = np.pad(target, ((0, pad_len), (0, 0)), mode='constant')
            contact = np.pad(contact, ((0, pad_len), (0, 0)), mode='constant')

        return (
            torch.FloatTensor(distorted),
            torch.FloatTensor(target),
            torch.FloatTensor(contact),
        )


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/training/v12"
    dataset = MotionFixDatasetV12(data_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    for distorted, target, contact in loader:
        print(f"Batch distorted: {distorted.shape}")
        print(f"Batch target:    {target.shape}")
        print(f"Batch contact:   {contact.shape}")
        print(f"Contact mean:    {contact.mean().item():.4f}")
        break

    print("Dataset test passed.")
