import math
from typing import Optional
import torch.nn as nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F
from einops import rearrange


class MHANet(nn.Module):
    def __init__(self, args):
        super(MHANet, self).__init__()
        in_channel = args.eeg_channel
        seq_len = args.window_length

        # EEGSingleStreamBlock 现在集成了 MTA
        self.EEGSingleStreamBlock = EEGSingleStreamBlock(
            in_channels=64,
            hidden_dim=128,
            num_heads=8,
            mlp_ratio=2.0,
            cond_dim=32,
            args=args,
        )

        self.Spatiotemporal_Convolution = Spatiotemporal_Convolution(
            in_channel, seq_len
        )

        # ⚠️ 注意: Multiscale_Temporal_Attention 已移除
        self.Multiscale_Global_Attention = Multiscale_Global_Attention(
            in_channels=64
        )  # 修复 MGA

        final_hidden = 4
        self.final_cond_proj = nn.Linear(
            32, final_hidden
        )  # project cond to final size for AdaLN
        self.final_layer = FinalLayer(
            hidden_size=final_hidden, patch_size=1, out_channels=2, act_layer=nn.SiLU
        )

    def forward(self, x, cond_vec: Optional[torch.Tensor] = None):
        B = x.shape[0]
        # ensure cond_vec
        if cond_vec is None:
            cond_vec = x.new_zeros((B, 32))

        # standardize shape to [B, C, H, W] for EEGSingleStreamBlock
        # if input is [B, 1, C, T], permute to [B, C, 1, T]
        if x.shape[1] == 1:
            x_proc = x.permute(0, 2, 1, 3)  # [B, C, 1, T]
        else:
            x_proc = x

        x_proc = self.EEGSingleStreamBlock(x_proc, cond_vec)  # [B, C, 1, T] -> same

        # ⚠️ 注意: Multiscale_Temporal_Attention 调用已移除

        x_proc = self.Multiscale_Global_Attention(x_proc)  # [B, 64, 1, T]

        # Spatiotemporal conv expects [B, 1, C, T] - ensure permutation
        # x_proc [B, 64, 1, T], Spatio Conv input [B, 1, C, T]
        x_spatio = x_proc.permute(0, 2, 1, 3)  # [B, 1, 64, T]
        x_spatio = self.Spatiotemporal_Convolution(x_spatio)  # [B, 4, 1, 1]

        # flatten -> [B, 4]
        x_flat = x_spatio.view(B, -1)  # [B, 4]

        # project cond to final size and call FinalLayer
        c_final = self.final_cond_proj(cond_vec)  # [B, final_hidden=4]

        out = self.final_layer(x_flat, c_final)  # [B, 2]
        return out


def modulate(
    x: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
):
    """
    Flexible FiLM-like modulation.
    - x can be shape [B, D] (vector per sample) or [B, C, H, W] (channel-first).
    - shift/scale expected shape [B, D] where D matches channel dim for x:
        - if x.dim()==2: D == x.size(1)
        - if x.dim()==4: D == x.size(1) (channels)
    """
    if shift is None and scale is None:
        return x

    if x.dim() == 2:
        # x: [B, D]
        if scale is None:
            return x + shift
        if shift is None:
            return x * (1 + scale)
        return x * (1 + scale) + shift
    elif x.dim() == 4:
        # x: [B, C, H, W], shift/scale: [B, C]
        B, C, H, W = x.shape
        if scale is None:
            shift_ = shift.view(B, C, 1, 1)
            return x + shift_
        if shift is None:
            scale_ = scale.view(B, C, 1, 1)
            return x * (1 + scale_)
        shift_ = shift.view(B, C, 1, 1)
        scale_ = scale.view(B, C, 1, 1)
        return x * (1 + scale_) + shift_
    else:
        # Fallback: try to broadcast last dim
        return x


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        return self.norm(x)


class EEGSingleStreamBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        cond_dim: int = 32,
        dropout: float = 0.1,
        args=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.in_channels = in_channels

        # condition embedder
        self.condition_encoder = MLPEmbedder(cond_dim, hidden_dim)

        # modulation generator: shift, scale, gate
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, hidden_dim * 3)
        )

        # channel projection
        self.channel_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # qkv + mlp projection
        # ⚠️ QKV 投影尺寸可能需要根据 hidden_dim 进行调整，但这里假设 hidden_dim=128
        self.qkv_proj = nn.Conv2d(
            hidden_dim, hidden_dim * 3 + self.mlp_hidden_dim, kernel_size=1
        )
        self.out_proj = nn.Conv2d(
            hidden_dim + self.mlp_hidden_dim, hidden_dim, kernel_size=1
        )

        # 1. 🌟 NEW: 集成 Multiscale_Temporal_Attention (MTA)
        # MTA 在原模型中用于调制 V，这里我们用它来处理 V。
        # 注意：MTA 的输入输出都是 [B, C, H, W] 形状，其中 H=1
        self.Multiscale_Temporal_Attention = Multiscale_Temporal_Attention(
            args, in_channels=hidden_dim
        )

        # normalization
        self.q_norm = QKNorm(self.head_dim)
        self.k_norm = QKNorm(self.head_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # activation
        self.act = nn.GELU()

        # final output projection
        self.final = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.modulate_fn = modulate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond_vec: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape
        if cond_vec is None:
            cond_vec = x.new_zeros((B, 32))

        # 1️⃣ Condition embedding + modulation params
        cond_feat = self.condition_encoder(cond_vec)  # [B, hidden_dim]
        shift, scale, gate = self.modulation(cond_feat).chunk(3, dim=-1)

        # 2️⃣ Project channels
        x_h = self.channel_proj(x)  # [B, hidden_dim, H, W]
        x_flat = rearrange(x_h, "b c h w -> b (h w) c")
        x_norm = self.norm(x_flat)
        x_norm = rearrange(x_norm, "b (h w) c -> b c h w", h=H, w=W)

        x_mod = self.modulate_fn(x_norm, shift=shift, scale=scale)

        # 3️⃣ QKV projection
        qkv_mlp = self.qkv_proj(x_mod)
        qkv, mlp_in = torch.split(
            qkv_mlp, [3 * self.hidden_dim, self.mlp_hidden_dim], dim=1
        )
        q, k, v = torch.chunk(qkv, 3, dim=1)  # q, k, v are [B, hidden_dim, H, W]

        # 2. 🌟 NEW: MTA 处理 V
        v = self.Multiscale_Temporal_Attention(v)  # [B, hidden_dim, H, W]

        # reshape for multi-head attention
        q = rearrange(q, "b (h d) H W -> b h (H W) d", h=self.num_heads)
        k = rearrange(k, "b (h d) H W -> b h (H W) d", h=self.num_heads)
        v = rearrange(v, "b (h d) H W -> b h (H W) d", h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4️⃣ Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = attn.softmax(dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = rearrange(attn_out, "b h (H W) d -> b (h d) H W", H=H, W=W)

        # 5️⃣ Parallel MLP
        mlp_out = self.act(mlp_in)
        out = torch.cat([attn_out, mlp_out], dim=1)
        out = self.out_proj(out)

        # 6️⃣ Residual + Gate
        gate_act = torch.sigmoid(gate).view(B, self.hidden_dim, 1, 1)
        x_after = x_h + self.dropout(gate_act * out)
        x_out = self.final(x_after)

        return x_out


class Spatiotemporal_Convolution(nn.Module):
    def __init__(self, in_channel, seq_len):
        super().__init__()
        kernel_t = min(8, seq_len)
        padding_t = kernel_t // 2
        self.Temporal_Convolution = nn.Sequential(
            nn.Conv2d(1, 4, (1, kernel_t), stride=1, padding=(0, padding_t)),
            nn.BatchNorm2d(4),
            nn.ELU(),
        )
        self.Spatio_Convolution = nn.Sequential(
            nn.Conv2d(4, 4, (in_channel, 1), stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.Temporal_Convolution(x)
        x = self.Spatio_Convolution(x)
        x = self.pool(x)
        return x


class Multiscale_Temporal_Layer(nn.Module):
    def __init__(self, seq_len, kernel_size):
        super(Multiscale_Temporal_Layer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.multiscaleConv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding
        )
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(seq_len)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.multiscaleConv(x)
        current_len = x.shape[-1]
        norm = nn.LayerNorm(current_len, elementwise_affine=False).to(x.device)
        x = norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class Multiscale_Temporal_Attention(nn.Module):
    # 🌟 接受 in_channels 并调整尺寸
    def __init__(
        self, args, in_channels=128
    ):  # 默认128，因为 ESSB 的 V 是 128 (hidden_dim)
        super().__init__()
        seq_len = args.window_length
        hidden_channels = 3  # 保持原模型的3个多尺度分支

        # 1. 🌟 移除 spatioConv: V 已经是 [B, C, 1, T] 形状，不需要再压通道

        # 2. 🌟 修改 upChannelConv: 从 C 调整到 3 个分支
        self.project_in = nn.Conv2d(in_channels, hidden_channels, 1)

        # 3. 🌟 修改 project_out: 从 3 个分支调整回 In_channels
        self.project_out = nn.Conv2d(hidden_channels, in_channels, 1)

        self.multiTemporal_K_2 = Multiscale_Temporal_Layer(seq_len, 2)
        self.multiTemporal_K_4 = Multiscale_Temporal_Layer(seq_len, 4)
        self.multiTemporal_K_6 = Multiscale_Temporal_Layer(seq_len, 6)

    def forward(self, x):
        # x 形状: [B, C, H, W] -> [B, 128, 1, T]
        B, C, H, W = x.shape
        shortcut = x.clone()  # 用于残差连接

        # 1. 通道降维，拆分分支
        x_proj = self.project_in(x)  # [B, 3, 1, T]

        # 转换为 [B, 3, T] 以适应 Conv1d
        x_proc = x_proj.squeeze(2)  # [B, 3, T]
        x_2, x_4, x_6 = x_proc.chunk(3, dim=1)  # 每个 [B, 1, T]

        # 2. 计算注意力权重 (Temporal Attn)
        # 注意力的输出是 [B, 1, 1] 形状 (AdaptiveAvgPool1d(1))
        x_attn_2 = self.multiTemporal_K_2(x_2).view(B, 1, 1, 1)
        x_attn_4 = self.multiTemporal_K_4(x_4).view(B, 1, 1, 1)
        x_attn_6 = self.multiTemporal_K_6(x_6).view(B, 1, 1, 1)

        # 3. 🌟 恢复乘法门控 (Multiplicative Gating)
        # 原始的分支特征 x_proj 是 [B, 3, 1, T]
        x_proj_2, x_proj_4, x_proj_6 = x_proj.chunk(3, dim=1)

        # 权重乘分支 (Multiplication)
        out_2 = x_attn_2 * x_proj_2  # [B, 1, 1, T]
        out_4 = x_attn_4 * x_proj_4  # [B, 1, 1, T]
        out_6 = x_attn_6 * x_proj_6  # [B, 1, 1, T]

        # 4. 融合并上采样
        out = torch.cat([out_2, out_4, out_6], dim=1)  # [B, 3, 1, T]
        out = self.project_out(out)  # [B, C, 1, T]

        # 5. 残差连接 (可选，但推荐)
        out = out + shortcut

        return out


class Multiscale_Global_Attention(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.in_channels = in_channels

        # 1. 🌟 调整 downchannel 以适应 in_channels
        self.downchannel = nn.Conv2d(in_channels, 3, 1)  # 降到 3 通道以匹配 3 个分支

        self.norm = nn.BatchNorm2d(3)  # BN3

        self.dilation_rate = 3

        # 2. 🌟 调整 conv 以适应 in_channels/3 的特征
        # 卷积核仍然是 1 通道对 1 通道，但输入是 3 通道，需要拆分。
        # 恢复原模型的 3 个分支卷积
        self.conv_0 = nn.Conv2d(
            1, 1, 3, padding=1, dilation=1
        )  # padding=1 for k=3, d=1
        self.conv_1 = nn.Conv2d(
            1, 1, 5, padding=4, dilation=2
        )  # padding=4 for k=5, d=2
        self.conv_2 = nn.Conv2d(
            1, 1, 7, padding=9, dilation=self.dilation_rate
        )  # padding=9 for k=7, d=3

        # 3. 🌟 恢复 upChannel 以适应 in_channels
        self.upChannel = nn.Conv2d(3, in_channels, 1)

    def forward(self, x):
        shortcut = x.clone()  # [B, 64, 1, T]

        # 1. 降维并归一化
        x_proj = self.downchannel(x)  # [B, 3, 1, T]
        x_norm = self.norm(x_proj)  # [B, 3, 1, T]

        # 2. 拆分并进行乘法门控
        y1, y2, y3 = torch.chunk(x_norm, 3, dim=1)  # 每个 [B, 1, 1, T]

        attn_0 = self.conv_0(y1) * y1
        attn_1 = self.conv_1(y2) * y2
        attn_2 = self.conv_2(y3) * y3

        attn = torch.cat([attn_0, attn_1, attn_2], dim=1)  # [B, 3, 1, T]

        # 3. 维度提升
        out = self.upChannel(attn)  # [B, 64, 1, T]

        # 4. 残差连接
        out = out + shortcut
        return out


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, act_layer):
        super().__init__()

        # Just use LayerNorm for the final layer
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if isinstance(patch_size, int):
            self.linear = nn.Linear(
                hidden_size, patch_size * patch_size * out_channels, bias=True
            )
        else:
            self.linear = nn.Linear(
                hidden_size,
                patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
                bias=True,
            )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Here we don't distinguish between the modulate types. Just use the simple one.
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x
