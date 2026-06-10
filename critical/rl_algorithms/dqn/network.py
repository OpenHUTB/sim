# rl_algorithms/dqn/network.py
# 标准 DQN Q 网络结构

import torch
import torch.nn as nn

from config.dqn_config import STATE_SIZE, ACTION_SIZE, HIDDEN_SIZES, ACTIVATION


def _build_mlp(in_dim, out_dim, hidden_sizes, activation="relu"):
    """构建多层感知机"""
    layers = []
    prev = in_dim
    act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn)
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """标准 DQN Q 网络"""

    def __init__(self, state_size=None, action_size=None, hidden_sizes=None):
        super().__init__()
        self.state_size = state_size or STATE_SIZE
        self.action_size = action_size or ACTION_SIZE
        hs = hidden_sizes or HIDDEN_SIZES
        self.net = _build_mlp(self.state_size, self.action_size, hs, ACTIVATION)

    def forward(self, x):
        return self.net(x)
