# rl_algorithms/dqn/attention_network.py
# Attention-DQN 网络：多头注意力 + Q 值估计（毕设创新点）

import math

import torch
import torch.nn as nn

from config.dqn_config import (
    STATE_SIZE, ACTION_SIZE, HIDDEN_SIZES,
    ATTENTION_HEAD_DIM, NUM_ATTENTION_HEADS, ATTENTION_DROPOUT,
)


class MultiHeadAttention(nn.Module):
    """标准多头注意力模块"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, N, embed_dim)
        返回: out (B, N, embed_dim), attn_weights (B, H, N, N)
        """
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                 # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)       # (B, N, D)
        out = self.out_proj(out)
        return out, attn


class AttentionQNetwork(nn.Module):
    """
    带多头注意力的 Q 网络（Attention-DQN，毕设创新）。

    流程:
        输入 state → 投影到 embed_dim → 复制为 N 个 token (加可学习位置编码)
        → 多头注意力编码 token 间关系 → 残差 + LayerNorm
        → 全局平均池化 → MLP 头 → Q 值
    """

    def __init__(self, state_size=None, action_size=None,
                 hidden_sizes=None, num_heads=None, head_dim=None, dropout=None):
        super().__init__()
        self.state_size = state_size or STATE_SIZE
        self.action_size = action_size or ACTION_SIZE
        self.num_heads = num_heads or NUM_ATTENTION_HEADS
        self.head_dim = head_dim or ATTENTION_HEAD_DIM
        self.dropout = dropout if dropout is not None else ATTENTION_DROPOUT
        self.embed_dim = self.num_heads * self.head_dim  # e.g. 4 × 64 = 256

        # 状态投影
        self.state_proj = nn.Linear(self.state_size, self.embed_dim)

        # 可学习的 token 位置编码：N 个 token，每个 embed_dim 维
        self.num_tokens = self.num_heads  # token 数量 = 头数
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # 多头注意力
        self.attention = MultiHeadAttention(self.embed_dim, self.num_heads, self.dropout)
        self.attn_norm = nn.LayerNorm(self.embed_dim)

        # Q 值输出头: embed_dim → hidden → action_size
        hs = hidden_sizes or HIDDEN_SIZES
        prev = self.embed_dim
        mlp_layers = []
        for h in hs:
            mlp_layers.append(nn.Linear(prev, h))
            mlp_layers.append(nn.ReLU())
            prev = h
        mlp_layers.append(nn.Linear(prev, self.action_size))
        self.q_head = nn.Sequential(*mlp_layers)

    def forward(self, x):
        B = x.size(0)

        # 投影到嵌入空间
        embed = self.state_proj(x)                      # (B, embed_dim)

        # 复制为 N 个 token，加上可学习位置编码
        tokens = embed.unsqueeze(1).expand(-1, self.num_tokens, -1)  # (B, N, embed_dim)
        tokens = tokens + self.pos_embed                             # (B, N, embed_dim)

        # 多头注意力 + 残差
        attn_out, attn_weights = self.attention(tokens)   # (B, N, embed_dim)
        attn_out = self.attn_norm(attn_out + tokens)

        # 全局平均池化 → Q 值
        pooled = attn_out.mean(dim=1)                     # (B, embed_dim)
        q_values = self.q_head(pooled)                    # (B, action_size)

        return q_values, attn_weights
