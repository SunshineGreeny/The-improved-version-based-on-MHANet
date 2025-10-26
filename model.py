import math
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

        self.out = nn.Linear(5, 2)

        self.channelAttention = ChannelAttention(args, dim=64, num_heads=16, bias=False)

        self.flatten = nn.Flatten()

        self.Spatiotemporal_Convolution = Spatiotemporal_Convolution(in_channel, seq_len)

        self.ECAGlobalBlock = ECAGlobalBlock(in_channel)

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)

        x = self.channelAttention(x)

        x = x.permute(0, 2, 1, 3)

        x = self.ECAGlobalBlock(x)

        x = self.Spatiotemporal_Convolution(x)

        x = self.flatten(x)

        x = self.out(x)

        return x


class Spatiotemporal_Convolution(nn.Module):
    def __init__(self, in_channel, seq_len):
        super(Spatiotemporal_Convolution, self).__init__()
        self.Temporal_Convolution = nn.Sequential(
            nn.Conv2d(1, 5, (1, 2), stride=1),

            nn.BatchNorm2d(5),

            nn.ELU()
        )

        self.Spatio_Convolution = nn.Sequential(
            nn.Conv2d(5, 5, (in_channel, 1), stride=1),

            nn.BatchNorm2d(5),

            nn.ELU()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.Temporal_Convolution(x)

        x = self.Spatio_Convolution(x)

        x = self.pool(x)

        return x


class ChannelAttention(nn.Module):

    def __init__(self, args, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()

        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dropout_attn = nn.Dropout(0.1) 
        self.dropout_proj = nn.Dropout(0.2) 

        self.Multiscale_Temporal_Attention = Multiscale_Temporal_Attention(args)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)

        v = self.Multiscale_Temporal_Attention(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.dropout_attn(attn) 

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.dropout_proj(out)

        return out


class Multiscale_Temporal_Layer(nn.Module):
    def __init__(self, seq_len, kernel_size):
        super(Multiscale_Temporal_Layer, self).__init__()

        self.multiscaleConv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding='same')

        self.act = nn.ELU()

        self.norm = nn.LayerNorm(seq_len)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.multiscaleConv(x)

        x = self.norm(x)

        x = self.act(x)

        x = self.pool(x)

        return x


class Multiscale_Temporal_Attention(nn.Module):
    def __init__(self, args):
        super(Multiscale_Temporal_Attention, self).__init__()

        seq_len = args.window_length

        self.spatioConv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(64, 1))

        self.upChannelConv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)

        self.project_out = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1)

        self.multiTemporal_K_2 = Multiscale_Temporal_Layer(seq_len, kernel_size=2)

        self.multiTemporal_K_4 = Multiscale_Temporal_Layer(seq_len, kernel_size=4)

        self.multiTemporal_K_6 = Multiscale_Temporal_Layer(seq_len, kernel_size=6)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)

        x = self.spatioConv(x)

        x = self.upChannelConv(x.squeeze(2))

        x, y, z = x.chunk(3, dim=1)

        x_attn = self.multiTemporal_K_2(x)

        y_attn = self.multiTemporal_K_4(y)

        z_attn = self.multiTemporal_K_6(z)

        out = x_attn * x + y_attn * y + z_attn * z

        out = out.view(x.shape[0], 1, 1, -1)

        out = self.project_out(out)

        return out


class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 形状: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)

        y = y.squeeze(-1).transpose(-1, -2) # (B, 1, C)

        y = self.conv(y)

        y = y.transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECAGlobalBlock(nn.Module):
    """
    用 ECA 替换 Multiscale_Global_Attention。
    它接收 MHANet 的 permute 后的输入 (B, T, C, 1)。
    """

    def __init__(self, channels, k_size=3):
        super().__init__()
        # channels 对应 C (特征维度，例如 64)
        self.norm = nn.BatchNorm2d(channels)
        self.eca = ECALayer(channels, k_size=k_size)

    def forward(self, x):
        # x: (B, T, C, 1)
        shortcut = x.clone()

        # 1. 适配 ECA/BatchNorm2d 的标准形状: (B, C, H, W)
        x_in = x.permute(0, 2, 1, 3)  # (B, C, T, 1)

        # 2. 归一化 (作用于通道 C)
        x_norm = self.norm(x_in)

        # 3. ECA 注意力
        x_attn = self.eca(x_norm)

        # 4. 残差连接
        out = x_attn + x_in

        # 5. 恢复 MHANet 后续模块所需的形状: (B, T, C, 1)
        out = out.permute(0, 2, 1, 3)

        return out
