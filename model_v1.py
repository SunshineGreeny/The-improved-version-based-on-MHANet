import math
import torch.nn as nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath


class MHANet(nn.Module):
    def __init__(self, args):
        super(MHANet, self).__init__()
        in_channel = args.csp_comp
        seq_len = args.window_length

        # self.BandSelectBlock = BandSelectBlock(
        #     feature_dimension=in_channel, features_num=3, drop_rate=0.1
        # )

        self.out = nn.Linear(5, 2)

        self.channelAttention = ChannelAttention(
            args, dim=in_channel, num_heads=16, bias=False
        )

        self.flatten = nn.Flatten()

        self.Spatiotemporal_Convolution = Spatiotemporal_Convolution(
            in_channel, seq_len
        )

        self.Multiscale_Global_Attention = Multiscale_Global_Attention()

        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (B, C, T, 1)
        # x = self.BandSelectBlock(x)
        x = self.channelAttention(x)  # 输入 (B, C, T, 1), 输出 (B, C, T, 1)
        x = x.permute(0, 3, 1, 2)  # 转换为 (B, 1, C, T) 以适应 Spatiotemporal_Convolution

        x = self.Multiscale_Global_Attention(x)
        x = self.Spatiotemporal_Convolution(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


class CovBlock(nn.Module):
    def __init__(self, feature_dimension, hidden_dim=None, dropout=0.1):
        super().__init__()
        D = feature_dimension
        if hidden_dim is None:
            hidden_dim = max(4, int(D * 0.6))
        self.cov_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.Dropout(dropout, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(D, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, D),
        )

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        cov = x.transpose(-2, -1) @ x
        cov_norm = torch.norm(x, p=2, dim=-2, keepdim=True)
        cov_norm = cov_norm.transpose(-2, -1) @ cov_norm
        cov = cov / (cov_norm + 1e-6)
        cov_stat = cov.mean(dim=-1)
        weight = self.cov_mlp(cov_stat)
        return weight.unsqueeze(-1)


class BandSelectBlock(nn.Module):
    def __init__(self, feature_dimension, features_num=3, drop_rate=0.1):
        super().__init__()
        base = feature_dimension // features_num
        extra = feature_dimension % features_num
        self.split_sizes = [
            (base + (1 if i < extra else 0)) for i in range(features_num)
        ]
        self.CovBlockList = nn.ModuleList(
            [
                CovBlock(D, hidden_dim=max(4, int(D * 0.6)), dropout=drop_rate)
                for D in self.split_sizes
            ]
        )
        self.channel_proj = nn.Conv2d(feature_dimension, feature_dimension, 1)

    def forward(self, x):
        B, D, H, W = x.shape
        x_list = torch.split(x, self.split_sizes, dim=1)
        feature_maps_weighted = []
        for feature_map, cov_block in zip(x_list, self.CovBlockList):
            D_i, N = feature_map.shape[1], H * W
            inp = rearrange(feature_map, "B C H W -> B (H W) C")
            if N > 1:
                inp = inp / (N - 1)
            c_weight = cov_block(inp)
            weight_map = c_weight.unsqueeze(-1)
            feature_maps_weighted.append(weight_map * feature_map)
        output_cat = torch.cat(feature_maps_weighted, dim=1)
        output_cat = self.channel_proj(output_cat)
        return output_cat


class Spatiotemporal_Convolution(nn.Module):
    def __init__(self, in_channel, seq_len):
        super(Spatiotemporal_Convolution, self).__init__()
        # 这里的输入是 (B, 1, C, T)
        self.Temporal_Convolution = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(1, 2), stride=1),  # 沿着时间维度卷积
            nn.BatchNorm2d(5),
            nn.ELU(),
        )

        self.Spatio_Convolution = nn.Sequential(
            nn.Conv2d(
                5, 5, kernel_size=(in_channel, 1), stride=1
            ),  # 沿着空间/通道维度卷积
            nn.BatchNorm2d(5),
            nn.ELU(),
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
        # 修正 START: 移除硬编码的 in_channel，使用传入的 dim
        # in_channel = args.csp_comp
        # 修正 END
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )

        self.BandSelectBlock = BandSelectBlock(
            feature_dimension=dim,
            features_num=3,
            drop_rate=0.1,
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dropout_attn = nn.Dropout(0.2)
        self.dropout_proj = nn.Dropout(0.4)

        self.Multiscale_Temporal_Attention = Multiscale_Temporal_Attention(
            args, channel_dim=dim
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)

        # q = self.BandSelectBlock(q)
        # k   = self.BandSelectBlock(k)
        v = self.BandSelectBlock(v)

        v = self.Multiscale_Temporal_Attention(v)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)

        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        attn = self.dropout_attn(attn)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)

        out = self.dropout_proj(out)

        return out


class Multiscale_Temporal_Layer(nn.Module):
    def __init__(self, seq_len, kernel_size):
        super(Multiscale_Temporal_Layer, self).__init__()

        self.multiscaleConv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same"
        )

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
    # 修正 START: 接收 channel_dim 参数
    def __init__(self, args, channel_dim):
        super(Multiscale_Temporal_Attention, self).__init__()

        seq_len = args.window_length

        self.spatioConv = nn.Conv2d(
            in_channels=channel_dim, out_channels=1, kernel_size=(1, 1)
        )  # 从(B,C,T,1)到(B,1,T,1)

        self.upChannelConv = nn.Conv1d(
            in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0
        )

        self.project_out = nn.Conv2d(
            in_channels=1, out_channels=channel_dim, kernel_size=1, stride=1
        )

        self.multiTemporal_K_2 = Multiscale_Temporal_Layer(seq_len, kernel_size=2)
        self.multiTemporal_K_4 = Multiscale_Temporal_Layer(seq_len, kernel_size=4)
        self.multiTemporal_K_6 = Multiscale_Temporal_Layer(seq_len, kernel_size=6)

    def forward(self, x):
        # 输入 x 的 shape: (B, C, T, 1)

        # x = x.permute(0, 2, 1, 3) # (B, T, C, 1)
        x_spatio = self.spatioConv(x)  # (B, 1, T, 1)

        x_up = self.upChannelConv(x_spatio.squeeze(-1))  # 输入 (B,1,T), 输出 (B,3,T)

        x1, y, z = x_up.chunk(3, dim=1)  # (B,1,T) each

        x_attn = self.multiTemporal_K_2(x1)
        y_attn = self.multiTemporal_K_4(y)
        z_attn = self.multiTemporal_K_6(z)

        out = x_attn * x1 + y_attn * y + z_attn * z  # (B, 1, T)

        out = out.unsqueeze(-1)  # (B, 1, T, 1)

        out = self.project_out(out)  # 恢复通道维度 (B, C, T, 1)

        return out * x  # 返回加权后的原始输入


class Multiscale_Global_Attention(nn.Module):
    def __init__(self):
        super(Multiscale_Global_Attention, self).__init__()

        self.downchannel = nn.Conv2d(3, 1, 1, 1, 0)

        self.norm = nn.BatchNorm2d(1)

        self.dilation_rate = 3

        self.conv_0 = nn.Conv2d(1, 1, 3, padding="same", dilation=1)

        self.conv_1 = nn.Conv2d(1, 1, 5, padding="same", dilation=2)

        self.conv_2 = nn.Conv2d(1, 1, 7, padding="same", dilation=self.dilation_rate)

        self.upChannel = nn.Sequential(nn.Conv2d(1, 3, 1, 1, 0))

    def forward(self, x):
        # 输入 x: (B, 1, C, T)
        shortcut = x.clone()

        x = self.norm(x)

        x = self.upChannel(x)  # (B, 3, C, T)

        y = x.clone()

        y1, y2, y3 = torch.chunk(y, 3, dim=1)  # (B, 1, C, T) each

        attn_0 = self.conv_0(y1) * y1
        attn_1 = self.conv_1(y2) * y2
        attn_2 = self.conv_2(y3) * y3

        attn = torch.cat([attn_0, attn_1, attn_2], dim=1)  # (B, 3, C, T)

        out = x * attn  # (B, 3, C, T)

        out = self.downchannel(out) + shortcut  # (B, 1, C, T)

        return out
