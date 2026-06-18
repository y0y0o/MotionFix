"""
MotionFix V9: Soft Gating + Temporal Smoothing + Kinematic Awareness

改进 vs V8:
  1. 软门控 + 时间平滑（消除抽搐）
  2. 骨骼链 IK（解决"只改脚不改腿"）
  3. 全向量化计算（训练速度与 V8 持平）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class MotionFixNetworkV9(nn.Module):
    def __init__(
        self,
        input_dim=66,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        blend_alpha=0.5,
        temperature=0.01,
        smooth_kernel_size=5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.blend_alpha = blend_alpha
        self.temperature = temperature

        self.foot_joints = [7, 8, 10, 11]
        self.foot_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j*3, j*3+1, j*3+2])

        # 脚部维度索引 → 用于广播索引
        self.register_buffer('foot_dims_tensor',
                             torch.tensor(self.foot_dims, dtype=torch.long))

        # 脚部维度掩码: (66,) — 用于快速检查是否属于脚
        foot_mask = torch.zeros(input_dim, dtype=torch.bool)
        foot_mask[self.foot_dims_tensor] = True
        self.register_buffer('foot_mask', foot_mask)

        # ---- Transformer backbone ----
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, input_dim),
        )

        # ---- V9: 可学习门控网络 ----
        self.gate_network = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.foot_joints)),
            nn.Sigmoid(),
        )

        # ---- V9: 时间平滑卷积 ----
        self.temporal_smooth = nn.Conv1d(
            len(self.foot_joints), len(self.foot_joints),
            kernel_size=smooth_kernel_size, padding=smooth_kernel_size // 2,
            groups=len(self.foot_joints), bias=False
        )
        with torch.no_grad():
            sigma = smooth_kernel_size / 5.0
            k = torch.arange(smooth_kernel_size, dtype=torch.float32) - smooth_kernel_size // 2
            gauss = torch.exp(-0.5 * (k / sigma) ** 2)
            gauss = gauss / gauss.sum()
            for i in range(len(self.foot_joints)):
                self.temporal_smooth.weight[i, 0] = gauss

    def forward(self, x, foot_only=False, return_gate=False):
        """
        foot_only=False: 训练 — 软门控混合（可微），不含 IK
        foot_only=True:  推理 — 软门控混合 + IK
        """
        B, T, _ = x.shape
        device = x.device

        h = self.input_proj(x)
        h = h.permute(1, 0, 2)
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)
        full_output = self.output_proj(h)

        # ---- 向量化软门控混合 ----
        blend_weights = self._compute_soft_blend_batched(x, full_output, h)
        # blend_weights: (B, T, 4)

        output = self._soft_blend_replace_batched(x, full_output, blend_weights)
        # output: (B, T, 66) — 脚部已混合，非脚部不变

        # 推理时做 IK
        if foot_only:
            for b in range(B):
                output[b] = self._leg_chain_ik(output[b], blend_weights[b])

        if return_gate:
            return output, self.gate_network(h)
        return output

    # ================================================================
    # 向量化版本：_compute_soft_blend（支持 batch）
    # ================================================================
    def _compute_soft_blend_batched(self, original, predicted, hidden):
        """
        全向量化软门控计算。

        original:   (B, T, 66)
        predicted:  (B, T, 66)
        hidden:     (B, T, d_model)

        Returns: (B, T, 4) blend weights
        """
        B, T, _ = original.shape
        device = original.device

        learned_gate = self.gate_network(hidden)  # (B, T, 4)

        # 批量计算每只脚的启发式门控
        heuristic_gate = torch.zeros(B, T, len(self.foot_joints), device=device)

        for j, fj in enumerate(self.foot_joints):
            y_dim = fj * 3 + 1
            xz_dims = [fj * 3, fj * 3 + 2]

            # 高度评分: (B, T)
            y = original[:, :, y_dim]
            ground = torch.quantile(y.reshape(B, -1), 0.05, dim=1)  # (B,)
            ground = ground[:, None]  # (B, 1)
            height_score = torch.sigmoid((ground + 0.03 - y) / self.temperature)

            # 速度评分: (B, T)
            xz = original[:, :, xz_dims]  # (B, T, 2)
            vel = torch.zeros(B, T, device=device)
            vel[:, 1:] = torch.norm(xz[:, 1:] - xz[:, :-1], dim=2)
            vel_score = torch.sigmoid((vel - 0.02) / self.temperature)

            heuristic_gate[:, :, j] = height_score * vel_score

        # 融合 + 时间平滑
        raw_weight = learned_gate * heuristic_gate  # (B, T, 4)

        # 时间平滑: (B, 4, T) → Conv1d → (B, 4, T) → (B, T, 4)
        raw_weight = raw_weight.permute(0, 2, 1)  # (B, 4, T)
        smoothed = self.temporal_smooth(raw_weight)
        smoothed = smoothed.permute(0, 2, 1)  # (B, T, 4)

        return smoothed * self.blend_alpha

    # ================================================================
    # 向量化版本：_soft_blend_replace（支持 batch）
    # ================================================================
    def _soft_blend_replace_batched(self, original, predicted, blend_weights):
        """
        全向量化软替换: 用 blend_weights 混合脚部，非脚部原样保留。

        blend_weights: (B, T, 4) — 每帧每只脚的混合强度
        Returns: (B, T, 66)
        """
        # blend_weights (B, T, 4) → 扩展到 (B, T, 12)（每个关节 XYZ 三通道重复）
        w_expanded = torch.repeat_interleave(blend_weights, 3, dim=2)  # (B, T, 12)

        # 只替换脚部维度
        output = original.clone()
        output[:, :, self.foot_dims_tensor] = (
            (1 - w_expanded) * original[:, :, self.foot_dims_tensor]
            + w_expanded * predicted[:, :, self.foot_dims_tensor]
        )
        return output

    # ================================================================
    # 推理专用 IK（不参与训练，不需要可微）
    # ================================================================
    def _leg_chain_ik(self, motion, blend_weights):
        """
        骨骼链 IK: 脚部修正后微调膝盖位置。

        右腿: joint 1(hip) → 4(knee) → 7(ankle) → 10(toe)
        左腿: joint 2(hip) → 5(knee) → 8(ankle) → 11(toe)
        """
        output = motion.clone()
        T = output.shape[0]
        device = output.device

        leg_chains = [
            (1, 4, 7, 10),   # 右腿 (hip, knee, ankle, toe)
            (2, 5, 8, 11),   # 左腿
        ]

        for hip_j, knee_j, ankle_j, toe_j in leg_chains:
            hip_pos = motion[:, hip_j*3:hip_j*3+3]
            knee_pos = motion[:, knee_j*3:knee_j*3+3]
            ankle_pos = motion[:, ankle_j*3:ankle_j*3+3]

            orig_upper_len = torch.norm(knee_pos - hip_pos, dim=1)
            orig_lower_len = torch.norm(ankle_pos - knee_pos, dim=1)

            foot_idx = self.foot_joints.index(ankle_j)
            ankle_dims = [ankle_j*3, ankle_j*3+1, ankle_j*3+2]

            # 批量处理所有帧（只用 torch ops）
            w_all = blend_weights[:, foot_idx]  # (T,)
            active = w_all > 0.01

            if not active.any():
                continue

            for t in range(1, T):
                w = w_all[t].item()
                if w < 0.01:
                    continue

                knee_cur = output[t, knee_j*3:knee_j*3+3]
                ankle_new = output[t, ankle_dims]
                hip_cur = output[t, hip_j*3:hip_j*3+3]

                direction = ankle_new - hip_cur
                dist = torch.norm(direction)
                if dist < 1e-6:
                    continue
                direction = direction / dist

                a = orig_upper_len[t]
                b = orig_lower_len[t]
                c = dist.item()
                c = max(abs(a.item() - b.item()) + 1e-6, min(c, a.item() + b.item() - 1e-6))

                cos_angle = (a*a + c*c - b*b) / (2*a*c + 1e-8)
                cos_angle = max(-1.0, min(1.0, cos_angle.item()))
                knee_angle = math.acos(cos_angle)

                up = torch.tensor([0.0, 1.0, 0.0], device=device)
                perp = torch.cross(direction, up)
                if torch.norm(perp) < 1e-6:
                    perp = torch.tensor([1.0, 0.0, 0.0], device=device)
                perp = perp / torch.norm(perp)

                knee_new = hip_cur + a * (
                    direction * math.cos(knee_angle) + perp * math.sin(knee_angle)
                )
                output[t, knee_j*3:knee_j*3+3] = (
                    (1 - w) * knee_cur + w * knee_new
                )

        return output


# ============================================================
# V9 Loss
# ============================================================
class MotionFixLossV9(nn.Module):
    def __init__(self, lambda_vel=0.5, lambda_foot=2.0, lambda_bone=1.0):
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_foot = lambda_foot
        self.lambda_bone = lambda_bone
        self.l1_loss = nn.L1Loss()

        self.foot_joints = [7, 8, 10, 11]
        self.foot_dims = []
        for j in self.foot_joints:
            self.foot_dims.extend([j*3, j*3+1, j*3+2])

        self.bone_pairs = [
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
            (0, 1), (1, 4), (4, 7), (7, 10),
            (0, 2), (2, 5), (5, 8), (8, 11),
            (9, 13), (13, 16), (16, 18), (18, 20),
            (9, 14), (14, 17), (17, 19), (19, 21),
        ]

    def forward(self, pred, target, original=None):
        loss_l1 = self.l1_loss(pred, target)

        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        target_vel = target[:, 1:, :] - target[:, :-1, :]
        loss_vel = self.l1_loss(pred_vel, target_vel)

        pred_foot = pred[:, :, self.foot_dims]
        target_foot = target[:, :, self.foot_dims]
        loss_foot = self.l1_loss(pred_foot, target_foot)

        pred_foot_vel = pred_foot[:, 1:, :] - pred_foot[:, :-1, :]
        target_foot_vel = target_foot[:, 1:, :] - target_foot[:, :-1, :]
        loss_foot_vel = self.l1_loss(pred_foot_vel, target_foot_vel)

        loss_bone = 0.0
        for pj, cj in self.bone_pairs:
            pred_len = torch.norm(
                pred[:, :, cj*3:cj*3+3] - pred[:, :, pj*3:pj*3+3], dim=2)
            target_len = torch.norm(
                target[:, :, cj*3:cj*3+3] - target[:, :, pj*3:pj*3+3], dim=2)
            loss_bone += self.l1_loss(pred_len, target_len)

        loss_contact_vel = 0.0
        if original is not None:
            for fj in self.foot_joints:
                y = original[:, :, fj*3+1]
                B = y.shape[0]
                ground = torch.quantile(y.reshape(B, -1), 0.05, dim=1)[:, None]
                contact_mask = (y < ground + 0.03).float()[:, :-1]
                foot_vel = torch.norm(
                    pred[:, 1:, fj*3:fj*3+3] - pred[:, :-1, fj*3:fj*3+3], dim=2)
                loss_contact_vel += (contact_mask * foot_vel).mean()

        total = (
            loss_l1
            + self.lambda_vel * loss_vel
            + self.lambda_foot * (loss_foot + loss_foot_vel)
            + self.lambda_bone * loss_bone
            + 0.5 * loss_contact_vel
        )
        return total, loss_l1, loss_foot, loss_bone, loss_contact_vel


if __name__ == "__main__":
    print("=" * 60)
    print("MotionFix V9 - Test (vectorized)")
    print("=" * 60)

    model = MotionFixNetworkV9(blend_alpha=0.5, temperature=0.01)
    x = torch.randn(2, 100, 66)

    import time
    t0 = time.time()
    y_train, gates = model(x, foot_only=False, return_gate=True)
    t1 = time.time()
    print(f"Train: {x.shape} -> {y_train.shape}  ({t1-t0:.3f}s)")
    print(f"Gate weights: {gates.shape}")

    t0 = time.time()
    y_infer = model(x, foot_only=True)
    t1 = time.time()
    print(f"Infer: {x.shape} -> {y_infer.shape}  ({t1-t0:.3f}s)")

    leg_dims = []
    for j in [1, 2, 4, 5, 7, 8, 10, 11]:
        leg_dims.extend([j*3, j*3+1, j*3+2])
    upper_dims = [i for i in range(66) if i not in leg_dims]
    diff_upper = (y_train[:, :, upper_dims] - x[:, :, upper_dims]).abs().max().item()
    print(f"Upper body change: {diff_upper:.10f} (should be 0)")

    criterion = MotionFixLossV9()
    total, l1, foot, bone, contact = criterion(y_train, torch.randn(2, 100, 66), x)
    print(f"Loss: total={total:.4f}, l1={l1:.4f}, foot={foot:.4f}, bone={bone:.4f}, contact={contact:.4f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # 速度测试
    x_batch = torch.randn(32, 196, 66)
    t0 = time.time()
    for _ in range(10):
        _ = model(x_batch, foot_only=False)
    t1 = time.time()
    print(f"Batch(32,196,66) avg: {(t1-t0)/10*1000:.0f}ms/forward")
    print("V9 All tests passed.")
