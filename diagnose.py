# diagnose.py

import torch
import numpy as np
from motionfix_model import MotionFixNetwork

device = 'cuda'
model = MotionFixNetwork().to(device)
checkpoint = torch.load("checkpoints/best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载一个MoMask输出
momask_data = np.load("./momask_results/momask_a_person_spins_around_multiple_times_while_staying_in_place_no_ik.npy")
if len(momask_data.shape) == 4:
    motion = momask_data[0]  # (T, 22, 3)
else:
    motion = momask_data

print(f"MoMask动作形状: {motion.shape}")
print(f"MoMask Y范围: [{motion[:,:,1].min():.4f}, {motion[:,:,1].max():.4f}]")
print(f"MoMask 数值范围: [{motion.min():.4f}, {motion.max():.4f}]")

# 加载一个训练数据看看
train_target = np.load("training_data/target_000000.npy")
print(f"\n训练数据形状: {train_target.shape}")
print(f"训练数据Y范围: [{train_target[:,:,1].min():.4f}, {train_target[:,:,1].max():.4f}]")
print(f"训练数据 数值范围: [{train_target.min():.4f}, {train_target.max():.4f}]")

# 检查模型输出的修正量
T = motion.shape[0]
motion_flat = motion.reshape(T, -1)  # (T, 66)
motion_tensor = torch.FloatTensor(motion_flat).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(motion_tensor)

correction = (output - motion_tensor).squeeze(0).cpu().numpy()

print(f"\n修正量统计:")
print(f"  均值: {correction.mean():.6f}")
print(f"  标准差: {correction.std():.6f}")
print(f"  最大值: {correction.max():.6f}")
print(f"  最小值: {correction.min():.6f}")
print(f"  绝对值均值: {np.abs(correction).mean():.6f}")

# 对比输入和输出的差异
print(f"\n输入和输出是否相同: {np.allclose(motion_flat, output.squeeze(0).cpu().numpy(), atol=1e-4)}")

# 检查尺度差异
print(f"\n尺度对比:")
print(f"  MoMask数据尺度: {np.abs(motion_flat).mean():.4f}")
print(f"  训练数据尺度:   {np.abs(train_target.reshape(train_target.shape[0],-1)).mean():.4f}")
print(f"  修正量尺度:     {np.abs(correction).mean():.6f}")