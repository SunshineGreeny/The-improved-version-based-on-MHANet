import math
import torch.nn as nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F
from einops import rearrange

class MHANet(nn.Module):
    def __init__(self, args):
        super(MHANet, self).__init__()
        model_channels = args.csp_comp
        seq_len = args.window_length
        self.d_model = model_channels
        self.ssm_dropout = 0.1

        self.Spatiotemporal_SSMBlock = SSMSpatiotemporal(
            model_channels, seq_len, d_model=64
        )
        self.out = nn.Linear(5, 2)
        self.channelAttention = ChannelAttention(
            args, dim=model_channels, num_heads=16, bias=False
        )
        self.flatten = nn.Flatten()
        
        self.ECAGlobalBlock = ECAGlobalBlock(model_channels) 

        # self.ffn = FeedForward(d_model=model_channels, dropout=self.ssm_dropout)
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        # 输入 x 形状: (B, 1, C, T)

        # 1. 通道注意力 (CA)
        x_ca = x.permute(0, 2, 1, 3) # (B, 1, C, T) -> (B, C, 1, T)
        x_ca = self.channelAttention(x_ca) # (B, C, 1, T)
        x_ca = x_ca.permute(0, 2, 1, 3) # (B, C, 1, T) -> (B, 1, C, T)

        # 2. ECA 全局块 
        x_eca = self.ECAGlobalBlock(x_ca) 

        # 3. FFN 特征增强
        # (B, 1, C, T) -> (B, C, T) -> (B, T, C)
        x_seq = x_eca.squeeze(1).transpose(1, 2) 
        # 恢复形状 (B, T, C) -> (B, C, T) -> (B, 1, C, T)
        x_seq = x_seq.transpose(1, 2).unsqueeze(1)

        # 4. 时空 SSM 块
        x_ssm = self.Spatiotemporal_SSMBlock(x_seq)

        # 5. 输出
        x_flat = self.flatten(x_ssm)
        x_out = self.out(x_flat)

        return x_out


class SSMSpatiotemporal(nn.Module):
    def __init__(self, in_channel, seq_len, d_model=64, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.in_channel = in_channel
        
        # === 维度修正和加速核心：将 SSM 特征维度从 C*d_model 降到 d_model ===
        d_ssm = d_model # SSM 块的最终特征维度 (例如 64)
        
        # 1. 输入投影 (替代原来的 in_proj): 
        # (B, 1, C, T) -> (B, d_ssm, 1, T)
        # 使用 in_channel x 1 的卷积核，将 C 维度压缩掉 (空间维度 H=C)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(1, d_ssm, (in_channel, 1)), # 使用 C x 1 卷积，将 C 维度混合并映射到 d_ssm
            nn.ReLU()
        )
        # SSM 块的输入特征维度现在是 d_ssm，而不是 C * d_model
        
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(
                d_model=d_ssm, # 使用 d_ssm 作为特征维度 (例如 64)，计算量大大降低
                bidirectional=True,  
                dropout=0.1,
                seq_len=seq_len
            ) for _ in range(n_layers)
        ])

        # 输出投影 (B, d_ssm, 1, T) -> (B, 5, 1, 1) -> (B, 5)
        self.out_proj = nn.Sequential(
            nn.Conv2d(d_ssm, 5, (1, 1)),  # 通道压缩到5
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # x: (B, 1, C, T)
        B, _, C, T = x.shape

        # 1. 空间投影 (B, 1, C, T) -> (B, d_ssm, 1, T)
        x = self.spatial_proj(x)  # **修正了 AttributeError**

        # 2. 转换为序列格式 (B, T, d_ssm)
        x = x.squeeze(2).transpose(1, 2) # (B, d_ssm, 1, T) -> (B, d_ssm, T) -> (B, T, d_ssm)

        # 3. SSM处理
        all_skip = 0
        for ssm_block in self.ssm_blocks:
            x, skip = ssm_block(x)
            all_skip += skip

        # 4. 恢复形状并输出
        # (B, T, d_ssm) -> (B, d_ssm, T) -> (B, d_ssm, 1, T)
        x = all_skip.transpose(1, 2).unsqueeze(2) 
        x = self.out_proj(x)
        
        return x


class ChannelAttention(nn.Module):
    def __init__(self, args, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
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
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dropout_attn = nn.Dropout(0.2)
        self.dropout_proj = nn.Dropout(0.3)
        
        self.Multiscale_Temporal_Attention = Multiscale_Temporal_Attention(args)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        v = self.Multiscale_Temporal_Attention(v) # (b, c, 1, w)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads) # V 现在是原始 V

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
        # 使用 dilation 或更复杂的结构可能更好，但保持原 multiscaleConv
        self.multiscaleConv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same"
        )
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(seq_len)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, 1, T)
        x = self.multiscaleConv(x)
        # LayerNorm 是作用于最后一个维度的，这里是 T
        x = self.norm(x) 
        x = self.act(x)
        x = self.pool(x) # (B, 1, 1)
        return x


class Multiscale_Temporal_Attention(nn.Module):
    def __init__(self, args):
        super(Multiscale_Temporal_Attention, self).__init__()
        model_channels = args.csp_comp
        seq_len = args.window_length

        # (B, 1, C, T) -> (B, 1, 1, T)，压缩通道维度 C
        self.spatioConv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(model_channels, 1)
        )

        # (B, 1, T) -> (B, 3, T)，将单个特征流扩展为 3 个用于多尺度处理
        self.upChannelConv = nn.Conv1d(
            in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0
        )

        # (B, 1, 1, T) -> (B, C, 1, T)，恢复通道数 C
        self.project_out = nn.Conv2d(
            in_channels=1, out_channels=model_channels, kernel_size=1, stride=1
        )

        self.multiTemporal_K_2 = Multiscale_Temporal_Layer(seq_len, kernel_size=2)
        self.multiTemporal_K_4 = Multiscale_Temporal_Layer(seq_len, kernel_size=4)
        self.multiTemporal_K_6 = Multiscale_Temporal_Layer(seq_len, kernel_size=6)

    def forward(self, x):
        # 输入 x 形状: (B, C, 1, T) - 这是来自 ChannelAttention 的 V
        
        # 1. 空间压缩
        x_in = x.permute(0, 2, 1, 3) # (B, C, 1, T) -> (B, 1, C, T)
        x_spatio_compressed = self.spatioConv(x_in) # (B, 1, 1, T)

        # 2. 扩展通道并分块
        x_seq = x_spatio_compressed.squeeze(2) # (B, 1, T)
        x_expanded = self.upChannelConv(x_seq) # (B, 3, T)
        x, y, z = x_expanded.chunk(3, dim=1) # (B, 1, T) * 3

        # 3. 多尺度时序注意力计算
        # attn_weights: (B, 1, 1) * 3
        x_attn = self.multiTemporal_K_2(x) 
        y_attn = self.multiTemporal_K_4(y)
        z_attn = self.multiTemporal_K_6(z)
        
        # 4. 注意力加权与聚合
        # 注意力权重广播到 (B, 1, T)
        out = x_attn * x + y_attn * y + z_attn * z 

        # 5. 恢复形状并投影通道
        out = out.view(x.shape[0], 1, 1, -1) # (B, 1, 1, T)
        out = self.project_out(out) # (B, C, 1, T)

        return out


class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # (B, C, H, W) -> (B, C, 1, 1)
        # 动态计算 kernel_size k
        # k = int(abs((math.log(channel, 2) + 1) / 2.)) 
        # k = k if k % 2 else k + 1
        
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, 1, T)
        y = self.avg_pool(x) # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2) # (B, 1, C)
        y = self.conv(y) # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class ECAGlobalBlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.eca = ECALayer(channels, k_size=k_size)

    def forward(self, x):
        # 输入 x 形状: (B, 1, C, T)
        
        # 1. 适配 ECA/BatchNorm2d 的标准形状: (B, C, H, W)
        x_in = x.permute(0, 2, 1, 3) # (B, C, 1, T)
        shortcut = x_in.clone()

        # 2. 归一化
        x_norm = self.norm(x_in)

        # 3. ECA 注意力
        x_attn = self.eca(x_norm)

        # **改进：新增残差连接**
        out = x_attn + shortcut 

        # 5. 恢复后续模块所需的形状: (B, 1, C, T)
        out = out.permute(0, 2, 1, 3) 

        return out


class GatedTemporalConv(nn.Module):
    # 改进：重命名以反映其门控时序卷积的性质，并使 kernel_size 可配置
    def __init__(
            self,
            d_model,
            bidirectional=False,
            dropout=0.1,
            kernel_size=5 # 改进：新增 kernel_size 参数
    ):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.kernel_size = kernel_size

        # 简化的SSM参数 (用于残差/跳跃连接)
        self.D = nn.Parameter(torch.randn(d_model))

        # 简化的卷积核 - 使用可学习参数
        # 改进：卷积核大小使用配置的 kernel_size
        self.conv_kernel = nn.Parameter(torch.randn(d_model, 1, kernel_size)) 
        self.conv_norm = nn.LayerNorm(d_model)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, D) 输入序列
        返回: (B, L, D) 输出序列
        """
        B, L, D = x.shape
        residual = x
        
        # 计算填充
        padding = (self.kernel_size - 1) // 2

        # 转置为卷积格式 (B, D, L)
        x_t = x.transpose(1, 2)  # (B, D, L)
        
        # 卷积操作 - 使用分组卷积 (groups=D)，实现深度可分离卷积的效果
        # kernel 形状: (D, 1, K)
        kernel = self.conv_kernel.squeeze(1) # (D, K)
        kernel = kernel.unsqueeze(1) # (D, 1, K)

        if self.bidirectional:
            # 双向处理：正向 + 反向
            x_forward = F.conv1d(x_t, kernel, padding=padding, groups=D)
            # 反向卷积：翻转序列，进行卷积，再翻转回来
            x_reverse = F.conv1d(x_t.flip(-1), kernel, padding=padding, groups=D).flip(-1)
            x_conv = (x_forward + x_reverse) / 2
        else:
            # 单向处理
            x_conv = F.conv1d(x_t, kernel, padding=padding, groups=D)

        # 恢复形状并添加跳跃连接
        x_conv = x_conv.transpose(1, 2)  # (B, L, D)
        x_conv = self.conv_norm(x_conv)

        # 跳跃连接 (类似 Mamba 中的 D 矩阵) + 激活
        x_out = x_conv + residual * self.D.unsqueeze(0).unsqueeze(0)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)

        return x_out, residual # 返回 residual 作为 skip，保持 SSMBlock 接口一致


class SSMBlock(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0.1, seq_len=None):
        super().__init__()
        self.d_model = d_model
        
        k_size = 5 # 默认值
        self.prenorm = nn.LayerNorm(d_model)
        self.pre_layer = nn.Linear(d_model, d_model * 2)

        # 使用更灵活的 GatedTemporalConv
        self.layer = GatedTemporalConv(
            d_model=d_model * 2,
            bidirectional=bidirectional,
            dropout=dropout,
            kernel_size=k_size
        )

        # === 修正部分开始 ===
        # 修正 linear_next 的输入维度。它作用于 inter (维度为 d_model)。
        self.linear_next = nn.Linear(d_model, d_model) # 修正: (in=D, out=D)
        
        # linear_skip 作用于 x_conv (维度为 d_model * 2)，用于跳跃连接
        self.linear_skip = nn.Linear(d_model * 2, d_model) # 保持: (in=2*D, out=D)
        # === 修正部分结束 ===

    def forward(self, x):
        skip = x

        # 归一化和投影
        x = self.prenorm(x) # (B, L, D)
        x = self.pre_layer(x) # (B, L, 2*D)

        # SSM/Gated Conv 处理
        x_conv, _ = self.layer(x) # (B, L, 2*D)

        # 门控机制 (GLU-like)
        gate, filter = x_conv.chunk(2, dim=-1) # D, D
        gate = torch.sigmoid(gate)
        filter = torch.tanh(filter)
        inter = gate * filter # (B, L, D)

        # 输出
        # out_skip 用于累加到下一层，使用 x_conv (2*D)
        out_skip = self.linear_skip(x_conv) # (B, L, 2*D) -> (B, L, D)
        
        # out_next 用于残差连接，使用 inter (D) -> 此时维度匹配
        out_next = self.linear_next(inter) + skip # (B, L, D) -> (B, L, D) + (B, L, D)

        return out_next, out_skip
