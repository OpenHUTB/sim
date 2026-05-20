# rl_algorithms/ppo/smooth_network.py
# Smooth-PPO 网络：带 LayerNorm 的平滑策略 Actor-Critic（毕设创新）

import torch
import torch.nn as nn

from config.ppo_config import (
    STATE_SIZE, ACTION_SIZE,
    HIDDEN_SIZES, ACTOR_HIDDEN, CRITIC_HIDDEN, ACTIVATION,
)


def _get_activation(name):
    return {"relu": nn.ReLU(), "tanh": nn.Tanh()}.get(name, nn.Tanh())


class SmoothActor(nn.Module):
    """
    Smooth-PPO Actor 网络（创新点）。

    与标准 Actor 的区别：在每一层后加入 LayerNorm，
    使特征分布更加平滑稳定，减少梯度震荡，提升行为自然度。
    """

    def __init__(self, state_size=None, action_size=None,
                 hidden_sizes=None, head_hidden=None):
        super().__init__()
        self.state_size = state_size or STATE_SIZE
        self.action_size = action_size or ACTION_SIZE
        hs = hidden_sizes or HIDDEN_SIZES
        hh = head_hidden or ACTOR_HIDDEN
        act = _get_activation(ACTIVATION)

        shared = []
        prev = self.state_size
        for h in hs:
            shared.append(nn.Linear(prev, h))
            shared.append(nn.LayerNorm(h))
            shared.append(act)
            prev = h
        self.shared = nn.Sequential(*shared)

        head = []
        prev = hs[-1] if hs else self.state_size
        for h in hh:
            head.append(nn.Linear(prev, h))
            head.append(nn.LayerNorm(h))
            head.append(act)
            prev = h
        head.append(nn.Linear(prev, self.action_size))
        head.append(nn.Softmax(dim=-1))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(self.shared(x))


class SmoothCritic(nn.Module):
    """
    Smooth-PPO Critic 网络。

    同样加入 LayerNorm 以平滑价值估计。
    """

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
            shared.append(nn.LayerNorm(h))
            shared.append(act)
            prev = h
        self.shared = nn.Sequential(*shared)

        head = []
        prev = hs[-1] if hs else self.state_size
        for h in hh:
            head.append(nn.Linear(prev, h))
            head.append(nn.LayerNorm(h))
            head.append(act)
            prev = h
        head.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(self.shared(x))
