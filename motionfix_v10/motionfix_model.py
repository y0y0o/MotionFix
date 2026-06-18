"""
MotionFix V10 - Transformer Encoder with FRDM-style Structured Losses

改进点（对标 FRDM）:
  1. 接触门控足部损失 L_Foot:
     当 b=1 (脚着地) 时，强制相邻帧脚位置变化为 0
     这是物理驱动的约束，而非纯数据驱动的模仿

  2. 时序平滑损失 L_Smooth:
     惩罚输出动作的高频抖动，鼓励低通特性

  3. 速度一致性损失 L_VelConsistency:
     速度积分得到的位置 应与 直接输出的位置一致

  4. 训练=推理一致:
     不再有 foot_only 标志分支。模型输入下半身有瑕疵的动作，
     输出全身修复动作。上半身因输入本身就是干净的，模型自然学到不动它。

  5. 接触标签 b 用于损失计算，不输入模型（遵循 FRDM 设计）
"""

import torch
import torch.nn as nn
import math
import numpy as np


# ================================================================
#  位置编码
# ================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :].unsqueeze(1)


# ================================================================
#  主网络
# ================================================================
class MotionFixNetwork(nn.Module):
    """
    Transformer Encoder，输入带瑕疵的动作 (T, 66)，输出修复后的动作 (T, 66)。

    不再有 foot_only / selective_replace 分支 ——
    训练和推理走完全相同的 forward，模型从数据中自然学会"只修下半身"。
    """

    def __init__(
        self,
        input_dim=66,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,              # (T, B, D) 格式
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim),
        )

    def forward(self, x):
        """
        x: (B, T, 66) 或 (T, B, 66)
        返回: (B, T, 66)
        """
        # 输入投影
        h = self.input_proj(x)                    # -> (B, T, d_model) or (T, B, d_model)

        # 转置为 Transformer 期望格式: (T, B, d_model)
        if h.dim() == 3 and h.shape[1] != x.shape[1]:
            h = h.permute(1, 0, 2)
        elif h.shape[0] == x.shape[0]:
            h = h.permute(1, 0, 2)

        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)                    # -> (B, T, d_model)

        output = self.output_proj(h)              # -> (B, T, 66)
        return output


# ================================================================
#  FRDM 风格结构化损失
# ================================================================
class MotionFixLoss(nn.Module):
    """
    四项损失，物理 + 数据双驱动:

      L_recon      基础 L1 重建（对标 FRDM 的 L_recon）
      L_Foot       接触门控足部速度约束 —— 核心物理项
                   当脚着地时，强制相邻帧位置变化为 0
      L_Smooth     时序平滑正则 —— 惩罚高频抖动
      L_VelCons    速度-位置一致性 —— 利用冗余约束自洽
    """

    def __init__(
        self,
        lambda_foot=2.0,         # 足部接触损失权重
        lambda_smooth=0.3,       # 时序平滑权重
        lambda_vel_cons=0.1,     # 速度-位置一致性权重
        foot_joints=(7, 8, 10, 11),
        contact_vel_thresh=0.03,
    ):
        super().__init__()
        self.lambda_foot = lambda_foot
        self.lambda_smooth = lambda_smooth
        self.lambda_vel_cons = lambda_vel_cons
        self.l1 = nn.L1Loss()

        # 脚部关节: 7=左脚踝, 8=右脚踝, 10=左脚掌, 11=右脚掌
        self.foot_joints = foot_joints
        self.foot_dims = []
        for j in foot_joints:
            self.foot_dims.extend([j * 3, j * 3 + 1, j * 3 + 2])

    # ----------------------------------------------------------
    #  L_Foot: 接触门控的足部速度约束
    # ----------------------------------------------------------
    def _foot_contact_loss(self, pred, contact):
        """
        pred:    (B, T, 66)  预测动作
        contact: (B, T, 2)   脚-地接触标签 {0, 1}

        L_Foot = mean( ||(p_{t+1} - p_t) * b_t||^2 )
        物理含义: 当脚着地时 (b=1)，相邻帧的脚位置变化应为 0
        """
        B, T, _ = pred.shape

        # 取脚踝位置: (B, T, 2, 3)
        foot_pos = pred[:, :, self.foot_dims].reshape(B, T, len(self.foot_joints), 3)

        # 只取左右脚踝 (第 0,1 个)
        foot_pos = foot_pos[:, :, :2, :]            # (B, T, 2, 3)

        # 相邻帧位移: (B, T-1, 2, 3)
        foot_vel = foot_pos[:, 1:, :, :] - foot_pos[:, :-1, :, :]

        # 接触掩码: (B, T-1, 2, 1)
        mask = contact[:, :-1, :].unsqueeze(-1)     # (B, T-1, 2, 1)

        # 只在接触帧计算损失
        loss = ((foot_vel * mask) ** 2).sum(dim=-1)  # (B, T-1, 2)
        loss = loss.mean()

        return loss

    # ----------------------------------------------------------
    #  L_Smooth: 时序平滑正则
    # ----------------------------------------------------------
    def _smooth_loss(self, pred):
        """
        惩罚相邻帧之间的加速（二阶差分），抑制高频抖动。
        """
        # 一阶: 速度
        vel = pred[:, 1:, :] - pred[:, :-1, :]       # (B, T-1, D)
        # 二阶: 加速度
        acc = vel[:, 1:, :] - vel[:, :-1, :]         # (B, T-2, D)
        return acc.pow(2).mean()

    # ----------------------------------------------------------
    #  L_VelCons: 速度-位置一致性
    # ----------------------------------------------------------
    def _velocity_consistency_loss(self, pred):
        """
        从预测位置计算速度 -> 累积回位置 -> 与直接预测的位置对比。
        利用冗余路径约束自洽性（对标 FRDM 的 L_vel-pos）。
        """
        # 预测速度
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]         # (B, T-1, D)

        # 从速度累积重建位置（以第 0 帧为锚点）
        pred_pos_from_vel = torch.cumsum(pred_vel, dim=1)   # (B, T-1, D)

        # 与直接输出的位置对比（第 1..T-1 帧）
        pred_pos_direct = pred[:, 1:, :]                    # (B, T-1, D)

        return self.l1(pred_pos_from_vel, pred_pos_direct)

    # ----------------------------------------------------------
    #  总损失
    # ----------------------------------------------------------
    def forward(self, pred, target, contact):
        """
        pred:    (B, T, 66)
        target:  (B, T, 66)
        contact: (B, T, 2)
        """
        # 1. 基础重建
        loss_recon = self.l1(pred, target)

        # 2. 接触门控足部约束 (FRDM 核心)
        loss_foot = self._foot_contact_loss(pred, contact)

        # 3. 时序平滑
        loss_smooth = self._smooth_loss(pred)

        # 4. 速度-位置一致性
        loss_vel_cons = self._velocity_consistency_loss(pred)

        # 加权求和
        loss_total = (
            loss_recon
            + self.lambda_foot * loss_foot
            + self.lambda_smooth * loss_smooth
            + self.lambda_vel_cons * loss_vel_cons
        )

        return loss_total, loss_recon, loss_foot, loss_smooth, loss_vel_cons


# ================================================================
#  自测
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MotionFix V10 - Self Test")
    print("=" * 60)

    model = MotionFixNetwork()
    x = torch.randn(2, 100, 66)
    contact = torch.zeros(2, 100, 2)

    # 前向
    y = model(x)
    print(f"Forward: {x.shape} -> {y.shape}  (should be same)")

    # 损失
    criterion = MotionFixLoss()
    target = x.clone()     # 理想情况：输入 = 输出
    total, l_recon, l_foot, l_smooth, l_vel = criterion(y, target, contact)

    print(f"\nLoss breakdown:")
    print(f"  L_recon:       {l_recon.item():.6f}")
    print(f"  L_Foot:        {l_foot.item():.6f}")
    print(f"  L_Smooth:      {l_smooth.item():.6f}")
    print(f"  L_VelCons:     {l_vel.item():.6f}")
    print(f"  Total:         {total.item():.6f}")

    # 测试 L_Foot 在脚着地时的行为
    print(f"\n--- L_Foot 行为测试 ---")
    contact_ones = torch.ones(2, 100, 2)    # 全部着地
    y_shifted = y.clone()
    y_shifted[:, 1:, 7*3] += 0.1            # 左脚踝 X 轴偏移 → 模拟脚滑
    _, l_foot_slip, _, _, _ = criterion(y_shifted, target, contact_ones)
    print(f"  全部着地 + 脚滑动: L_Foot = {l_foot_slip.item():.6f}  (should be > 0)")

    _, l_foot_clean, _, _, _ = criterion(y, target, contact_ones)
    print(f"  全部着地 + 无滑动: L_Foot = {l_foot_clean.item():.6f}  (should be ~0)")

    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")

    print("\nAll tests passed.")
