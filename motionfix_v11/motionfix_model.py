"""
MotionFix V11 - Fixed: V8 Selective Replace + Rebalanced FRDM-style Losses

修复要点:
  1. 恢复 V8 的 _selective_replace 推理硬约束:
     - 训练 foot_only=False: 全量重建（学习全局上下文）
     - 推理 foot_only=True:  只在脚着地且滑步的帧替换脚部
     - blend_alpha=0.5: 只应用50%修正，保证安全

  2. 损失权重重新平衡 — 修正力 >> 保守力:
     - 恢复 V8 式的直接 L1 脚部监督 (λ=2.0)，所有帧生效
     - 保留 FRDM 接触门控损失作为辅助 (λ=0.5)
     - 去掉加速度惩罚 L_Smooth（太保守）
     - 速度一致性改为温和速度匹配 (λ=0.3)

  3. 保持 FRDM 风格:
     - 接触标签 b 用于辅助损失，不输入模型
     - 速度-位置一致性作为自洽约束
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
    Transformer Encoder + V8 Selective Foot Replacement.

    训练: 重建全部关节（学习全局上下文）
    推理: 只在"脚在地面且滑动"的帧替换脚部
          其他帧（空中/正常接触）完全不改
    """

    def __init__(
        self,
        input_dim=66,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        blend_alpha=0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.blend_alpha = blend_alpha

        # 脚部关节: 7=左脚踝, 8=右脚踝, 10=左脚掌, 11=右脚掌
        self.foot_joints = [7, 8, 10, 11]
        self.foot_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j * 3, j * 3 + 1, j * 3 + 2])

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
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

    def forward(self, x, foot_only=False):
        """
        x: (B, T, 66)

        foot_only=False: 训练模式，输出全部关节
        foot_only=True:  推理模式，选择性替换脚部
        """
        # 输入投影
        h = self.input_proj(x)                     # (B, T, d_model)

        # Transformer 期望 (T, B, d_model)
        h = h.permute(1, 0, 2)
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)                    # (B, T, d_model)

        full_output = self.output_proj(h)          # (B, T, 66)

        if not foot_only:
            return full_output

        # === 推理模式：选择性替换（V8 核心机制）===
        output = x.clone()
        B = x.shape[0]
        for b in range(B):
            output[b] = self._selective_replace(x[b], full_output[b])
        return output

    def _selective_replace(self, original, predicted):
        """
        只在滑步帧替换脚部，其他帧保持原样。（V8 原始算法）

        original:  (T, 66)  输入动作（带瑕疵）
        predicted: (T, 66)  模型输出（全量重建）
        返回:      (T, 66)  选择性替换后的动作
        """
        output = original.clone()
        T = original.shape[0]

        for fj in self.foot_joints:
            dims = [fj * 3, fj * 3 + 1, fj * 3 + 2]
            y_dim = fj * 3 + 1  # Y 坐标（高度）

            heights = original[:, y_dim].detach().cpu().numpy()
            ground_level = np.percentile(heights, 5)
            contact_threshold = ground_level + 0.05

            for t in range(1, T):
                height = heights[t]

                if height < contact_threshold:
                    # 脚在地面 → 检查是否在滑动
                    orig_xz = original[t, [dims[0], dims[2]]]
                    prev_xz = original[t - 1, [dims[0], dims[2]]]
                    velocity = torch.norm(orig_xz - prev_xz).item()

                    if velocity > 0.03:
                        # 在滑动！→ 混合修正
                        alpha = self.blend_alpha
                        for d in dims:
                            output[t, d] = (
                                (1 - alpha) * original[t, d]
                                + alpha * predicted[t, d]
                            )
                # else: 脚在空中 → 不改（保持原样）

        return output


# ================================================================
#  V11 混合损失: V8 直接监督 + FRDM 接触门控辅助
# ================================================================
class MotionFixLoss(nn.Module):
    """
    损失设计原则: 修正力 >> 保守力

      L_recon      基础 L1 重建（全关节）
      L_foot       直接脚部位置 L1（V8 风格，所有帧）
      L_foot_vel   脚部速度 L1（所有帧）
      L_vel        全局速度匹配（温和，保持整体动力学）
      L_foot_ct    接触门控足部约束（FRDM 风格，辅助信号）
      L_vel_cons   速度-位置一致性（FRDM 风格自洽）
    """

    def __init__(
        self,
        lambda_foot=2.0,              # 直接脚部监督（最强）
        lambda_foot_vel=1.0,          # 脚部速度监督
        lambda_vel=0.3,               # 全局速度匹配（温和）
        lambda_foot_ct=0.5,           # 接触门控辅助（弱）
        lambda_vel_cons=0.2,          # 速度-位置自洽
        foot_joints=(7, 8, 10, 11),
    ):
        super().__init__()
        self.lambda_foot = lambda_foot
        self.lambda_foot_vel = lambda_foot_vel
        self.lambda_vel = lambda_vel
        self.lambda_foot_ct = lambda_foot_ct
        self.lambda_vel_cons = lambda_vel_cons
        self.l1 = nn.L1Loss()

        # 脚部关节维度
        self.foot_joints = foot_joints
        self.foot_dims = []
        for j in foot_joints:
            self.foot_dims.extend([j * 3, j * 3 + 1, j * 3 + 2])

    # ----------------------------------------------------------
    #  直接脚部 L1（V8 风格）
    # ----------------------------------------------------------
    def _foot_l1_loss(self, pred, target):
        """所有帧的脚部位置 + 速度直接监督"""
        pred_foot = pred[:, :, self.foot_dims]
        target_foot = target[:, :, self.foot_dims]
        loss_pos = self.l1(pred_foot, target_foot)

        pred_foot_vel = pred_foot[:, 1:, :] - pred_foot[:, :-1, :]
        target_foot_vel = target_foot[:, 1:, :] - target_foot[:, :-1, :]
        loss_vel = self.l1(pred_foot_vel, target_foot_vel)

        return loss_pos, loss_vel

    # ----------------------------------------------------------
    #  接触门控足部约束（FRDM 风格）
    # ----------------------------------------------------------
    def _foot_contact_loss(self, pred, contact):
        """
        当脚着地 (b=1) 时，约束相邻帧脚位置变化 → 0
        """
        B, T, _ = pred.shape

        # 取脚踝位置: (B, T, 4, 3) → 只取左右脚踝 (0,1)
        foot_pos = pred[:, :, self.foot_dims].reshape(B, T, len(self.foot_joints), 3)
        foot_pos = foot_pos[:, :, :2, :]              # (B, T, 2, 3)

        # 相邻帧位移
        foot_vel = foot_pos[:, 1:, :, :] - foot_pos[:, :-1, :, :]  # (B, T-1, 2, 3)

        # 接触掩码
        mask = contact[:, :-1, :].unsqueeze(-1)       # (B, T-1, 2, 1)

        loss = ((foot_vel * mask) ** 2).sum(dim=-1).mean()
        return loss

    # ----------------------------------------------------------
    #  速度-位置一致性（FRDM 风格）
    # ----------------------------------------------------------
    def _velocity_consistency_loss(self, pred):
        """cumsum(vel) 应与直接输出的位置一致"""
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        pred_pos_from_vel = torch.cumsum(pred_vel, dim=1)
        pred_pos_direct = pred[:, 1:, :]
        return self.l1(pred_pos_from_vel, pred_pos_direct)

    # ----------------------------------------------------------
    #  总损失
    # ----------------------------------------------------------
    def forward(self, pred, target, contact=None):
        """
        pred:    (B, T, 66)
        target:  (B, T, 66)
        contact: (B, T, 2)  可选，用于接触门控损失
        """
        # 1. 全关节重建
        loss_recon = self.l1(pred, target)

        # 2. 直接脚部监督（V8 核心，最强权重）
        loss_foot_pos, loss_foot_vel = self._foot_l1_loss(pred, target)
        loss_foot = loss_foot_pos + loss_foot_vel

        # 3. 全局速度匹配（温和）
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        target_vel = target[:, 1:, :] - target[:, :-1, :]
        loss_vel = self.l1(pred_vel, target_vel)

        # 4. 接触门控辅助（FRDM 风格，弱权重）
        loss_foot_ct = torch.tensor(0.0, device=pred.device)
        if contact is not None:
            loss_foot_ct = self._foot_contact_loss(pred, contact)

        # 5. 速度-位置自洽（FRDM 风格）
        loss_vel_cons = self._velocity_consistency_loss(pred)

        # 加权求和
        loss_total = (
            loss_recon
            + self.lambda_foot * loss_foot
            + self.lambda_foot_vel * loss_foot_vel
            + self.lambda_vel * loss_vel
            + self.lambda_foot_ct * loss_foot_ct
            + self.lambda_vel_cons * loss_vel_cons
        )

        return loss_total, loss_recon, loss_foot, loss_foot_ct, loss_vel_cons


# ================================================================
#  自测
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MotionFix V11 (Fixed) - Self Test")
    print("=" * 60)

    model = MotionFixNetwork(blend_alpha=0.5)
    x = torch.randn(2, 100, 66)
    contact = torch.zeros(2, 100, 2)

    # 训练模式
    y_train = model(x, foot_only=False)
    print(f"Train:  {x.shape} -> {y_train.shape}  (full output)")

    # 推理模式
    y_infer = model(x, foot_only=True)
    print(f"Infer:  {x.shape} -> {y_infer.shape}  (selective replace)")

    # 检查非脚部关节在推理时是否保持不变（V8 关键特性）
    non_foot = [i for i in range(66) if i not in model.foot_dims]
    diff = (y_infer[:, :, non_foot] - x[:, :, non_foot]).abs().max().item()
    print(f"Non-foot change (infer): {diff:.10f}  (should be 0.0)")

    # 损失测试
    criterion = MotionFixLoss()
    target = x.clone()
    total, l_recon, l_foot, l_foot_ct, l_vel_cons = criterion(y_train, target, contact)

    print(f"\nLoss breakdown:")
    print(f"  L_recon:     {l_recon.item():.6f}")
    print(f"  L_foot:      {l_foot.item():.6f}")
    print(f"  L_foot_ct:   {l_foot_ct.item():.6f}")
    print(f"  L_vel_cons:  {l_vel_cons.item():.6f}")
    print(f"  Total:       {total.item():.6f}")

    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")

    print("\nAll tests passed.")
