# rl_algorithms/dqn/agent.py
# 标准 DQN 智能体：ε-greedy 探索 + 经验回放 + 目标网络

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.base_agent import BaseAgent
from rl_algorithms.dqn.network import QNetwork
from rl_algorithms.dqn.replay_buffer import ReplayBuffer
from config.dqn_config import (
    STATE_SIZE, ACTION_SIZE, HIDDEN_SIZES,
    LEARNING_RATE, GAMMA, TAU, TARGET_UPDATE_FREQ,
    MEMORY_SIZE, BATCH_SIZE, MIN_REPLAY_SIZE,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    TRAIN_EVERY_N_STEPS,
)


class DQNAgent(BaseAgent):
    """
    标准 DQN 智能体。

    使用双网络（Q-Network + Target-Network）+ ε-greedy 探索。
    """

    def __init__(self, state_size=None, action_size=None):
        state_size = state_size or STATE_SIZE
        action_size = action_size or ACTION_SIZE
        super().__init__(state_size, action_size, name="DQN")

        # 网络
        self.q_net = QNetwork(state_size, action_size, HIDDEN_SIZES).to(self.device)
        self.target_net = QNetwork(state_size, action_size, HIDDEN_SIZES).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        # 回放池
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.batch_size = BATCH_SIZE

        # 超参数
        self.gamma = GAMMA
        self.tau = TAU
        self.target_update_freq = TARGET_UPDATE_FREQ
        self.train_every = TRAIN_EVERY_N_STEPS

        # 探索
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        # 上次训练的 loss
        self.last_loss = None

    # ================================================================
    # 核心接口
    # ================================================================

    def act(self, state, evaluate=False):
        """
        选择动作。

        evaluate=True 时关闭探索（用于评估/测试）。
        """
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        with torch.no_grad():
            state_t = self.to_tensor(state).unsqueeze(0)
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=-1).item()

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        """执行一步学习更新"""
        if len(self.memory) < MIN_REPLAY_SIZE:
            return None
        if self.total_steps % self.train_every != 0:
            return None

        self.train_steps += 1

        # 采样
        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        s = self.to_tensor(states)
        a = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = self.to_tensor(next_states)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 当前 Q 值
        q = self.q_net(s).gather(1, a)

        # 目标 Q 值（使用 target 网络）
        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1, keepdim=True)[0]
            target_q = r + self.gamma * next_q * (1 - d)

        loss = self.loss_fn(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # 软更新目标网络
        if self.train_steps % self.target_update_freq == 0:
            self._soft_update()

        self.last_loss = loss.item()
        self.update_epsilon()
        return {"loss": self.last_loss, "epsilon": self.epsilon}

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ================================================================
    # 持久化
    # ================================================================

    def _save_checkpoint(self, checkpoint, path):
        checkpoint.update({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        })
        torch.save(checkpoint, path)

    def _load_checkpoint(self, checkpoint):
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)

    # ================================================================
    # 内部
    # ================================================================

    def _soft_update(self):
        """目标网络软更新：θ_target = τ·θ + (1-τ)·θ_target"""
        for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
