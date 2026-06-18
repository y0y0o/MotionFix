"""
MotionFix V10 - Dataset

返回三元组: (distorted, target, contact)
  distorted: (max_len, 66)  仅下半身有瑕疵的关节位置
  target:    (max_len, 66)  干净关节位置
  contact:   (max_len, 2)   左右脚着地标签
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os


class MotionFixDataset(Dataset):
    def __init__(self, data_dir, max_len=196):
        self.data_dir = data_dir
        self.max_len = max_len

        self.distorted_files = sorted(
            glob.glob(f"{data_dir}/distorted_*.npy")
        )
        print(f"Dataset: {len(self.distorted_files)} pairs from {data_dir}")
        print(f"  max_len={max_len}, with contact labels")

    def __len__(self):
        return len(self.distorted_files)

    def __getitem__(self, idx):
        distorted_path = self.distorted_files[idx]
        target_path = distorted_path.replace('distorted_', 'target_')
        contact_path = distorted_path.replace('distorted_', 'contact_')

        distorted = np.load(distorted_path)    # (T, 22, 3)
        target = np.load(target_path)          # (T, 22, 3)
        contact = np.load(contact_path)        # (T, 2)

        # 展平: (T, 22, 3) -> (T, 66)
        T = distorted.shape[0]
        distorted = distorted.reshape(T, -1).astype(np.float32)
        target = target.reshape(T, -1).astype(np.float32)
        contact = contact.astype(np.float32)

        # 统一长度到 max_len
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
    dataset = MotionFixDataset("training_data_v10")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for distorted, target, contact in loader:
        print(f"Batch distorted: {distorted.shape}")
        print(f"Batch target:    {target.shape}")
        print(f"Batch contact:   {contact.shape}")
        break

    print("Dataset test passed.")
