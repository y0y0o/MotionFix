"""
MotionFix V11 - Dataset

支持两种数据格式:
  1. 新格式 (V10): distorted_*.npy + target_*.npy + contact_*.npy → 三元组
  2. 旧格式 (V2):  distorted_*.npy + target_*.npy → 二元组 (contact=None)
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

        # 检测是否有 contact 文件
        first_contact = self.distorted_files[0].replace('distorted_', 'contact_') if self.distorted_files else ''
        self.has_contact = os.path.exists(first_contact)

        print(f"Dataset: {len(self.distorted_files)} pairs from {data_dir}")
        print(f"  max_len={max_len}, contact_labels={'YES' if self.has_contact else 'NO'}")

    def __len__(self):
        return len(self.distorted_files)

    def __getitem__(self, idx):
        distorted_path = self.distorted_files[idx]
        target_path = distorted_path.replace('distorted_', 'target_')
        contact_path = distorted_path.replace('distorted_', 'contact_')

        distorted = np.load(distorted_path)    # (T, 22, 3)
        target = np.load(target_path)          # (T, 22, 3)

        # 展平
        T = distorted.shape[0]
        distorted = distorted.reshape(T, -1).astype(np.float32)
        target = target.reshape(T, -1).astype(np.float32)

        # 统一长度
        if T > self.max_len:
            distorted = distorted[:self.max_len]
            target = target[:self.max_len]
        elif T < self.max_len:
            pad_len = self.max_len - T
            distorted = np.pad(distorted, ((0, pad_len), (0, 0)), mode='constant')
            target = np.pad(target, ((0, pad_len), (0, 0)), mode='constant')

        # Contact (可选)
        if self.has_contact:
            contact = np.load(contact_path).astype(np.float32)  # (T, 2)
            if T > self.max_len:
                contact = contact[:self.max_len]
            elif T < self.max_len:
                contact = np.pad(contact, ((0, self.max_len - T), (0, 0)), mode='constant')
        else:
            contact = np.zeros((self.max_len, 2), dtype=np.float32)

        return (
            torch.FloatTensor(distorted),
            torch.FloatTensor(target),
            torch.FloatTensor(contact),
        )


if __name__ == "__main__":
    # 测试 V2 格式
    dataset = MotionFixDataset("../training_data_v2")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for distorted, target, contact in loader:
        print(f"Batch distorted: {distorted.shape}")
        print(f"Batch target:    {target.shape}")
        print(f"Batch contact:   {contact.shape} (all zeros)")
        break

    print("Dataset test passed.")
