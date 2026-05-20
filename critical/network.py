# rl_algorithms/ppo/network.py
# PPO 标准 Actor-Critic 网络

import torch
import torch.nn as nn

from config.ppo_config import (
    STATE_SIZE, ACTION_SIZE,
    HIDDEN_SIZES, ACTOR_HIDDEN, CRITIC_HIDDEN, ACTIVATION,
)


def _get_activation(name):
    return {"relu": nn.ReLU(), "tanh": nn.Tanh()}.get(name, nn.Tanh())


class Actor(nn.Module):
    """PPO Actor: 输出离散动作概率分布"""

    def __init__(self, state_size=None, action_size=None,
                 hidden_sizes=None, head_hidden=None):
        super().__init__()
        self.state_size = state_size or STATE_SIZE
        self.action_size = action_size or ACTION_SIZE
        hs = hidden_sizes or HIDDEN_SIZES
        hh = head_hidden or ACTOR_HIDDEN
        act = _get_activation(ACTIVATION)

        # 共享特征提取
        shared = []
        prev = self.state_size
        for h in hs:
            shared.append(nn.Linear(prev, h))
            shared.append(act)
            prev = h
        self.shared = nn.Sequential(*shared)

        # 策略头
        head = []
        prev = hs[-1] if hs else self.state_size
        for h in hh:
            head.append(nn.Linear(prev, h))
            head.append(act)
            prev = h
        head.append(nn.Linear(prev, self.action_size))
        head.append(nn.Softmax(dim=-1))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        features = self.shared(x)
        return self.head(features)


class Critic(nn.Module):
    """PPO Critic: 输出状态价值 V(s)"""

    def __init__(self, state_size=None, hidden_sizes=None, head_hidden=None):
        super().__init__()
        self.state_size = state_size or STATE_SIZE
        hs = hidden_sizes or HIDDEN_SIZES
        hh = head_hidden or CRITIC_HIDDEN
        act = _get_activation(ACTIVATION)

        shared = []
        prev = self.state_size
        for h in hs:
            shared.append(nn.Linear(prev, h))
            shared.append(act)
            prev = h
        self.shared = nn.Sequential(*shared)

        head = []
        prev = hs[-1] if hs else self.state_size
        for h in hh:
            head.append(nn.Linear(prev, h))
            head.append(act)
            prev = h
        head.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        features = self.shared(x)
        return self.head(features)
