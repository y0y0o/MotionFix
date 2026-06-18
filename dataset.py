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

    def __len__(self):
        return len(self.distorted_files)

    def __getitem__(self, idx):
        distorted_path = self.distorted_files[idx]
        target_path = distorted_path.replace('distorted_', 'target_')

        distorted = np.load(distorted_path)  # (T, 22, 3)
        target = np.load(target_path)        # (T, 22, 3)

        # 展平: (T, 22, 3) -> (T, 66)
        distorted = distorted.reshape(distorted.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        # 统一长度: 截断或补零到max_len
        T = distorted.shape[0]

        if T > self.max_len:
            distorted = distorted[:self.max_len]
            target = target[:self.max_len]
        elif T < self.max_len:
            pad_len = self.max_len - T
            distorted = np.pad(distorted, ((0, pad_len), (0, 0)), mode='constant')
            target = np.pad(target, ((0, pad_len), (0, 0)), mode='constant')

        return (
            torch.FloatTensor(distorted),
            torch.FloatTensor(target)
        )


if __name__ == "__main__":
    dataset = MotionFixDataset("training_data")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for distorted, target in loader:
        print(f"Batch distorted: {distorted.shape}")
        print(f"Batch target:    {target.shape}")
        break

    print("Dataset test passed.")