# rl_algorithms/base_agent.py
# 所有 RL 智能体的基类：设备管理、模型保存/加载、训练状态追踪

import os
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseAgent(ABC):
    """
    RL 智能体抽象基类。

    子类必须实现:
        act(state) -> action
        train() -> loss_info
        save(path) / load(path)
    """

    def __init__(self, state_size, action_size, name="BaseAgent"):
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 训练状态
        self.episode = 0
        self.total_steps = 0
        self.train_steps = 0

        # 日志
        self.loss_history = []
        self.reward_history = []

    @abstractmethod
    def act(self, state):
        """根据当前状态选择动作"""

    @abstractmethod
    def train(self):
        """执行一步学习更新，返回 loss 信息 dict 或 None"""

    def store(self, *args):
        """
        存储一条 transition。默认空操作，子类按需覆盖。
        DQN 子类存入 replay buffer，PPO 子类存入 rollout storage。
        """

    def update_epsilon(self):
        """探索率衰减（DQN 系用），子类覆盖"""

    def start_episode(self):
        """新 episode 开始时调用"""
        self.episode += 1

    def end_episode(self, total_reward):
        """episode 结束时记录奖励"""
        self.reward_history.append(total_reward)

    # ================================================================
    # 模型持久化
    # ================================================================

    def save(self, path):
        """保存模型检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
        }
        self._save_checkpoint(checkpoint, path)

    def load(self, path):
        """加载模型检查点"""
        if not os.path.exists(path):
            raise FileNotFoundError("检查点不存在: %s" % path)
        checkpoint = torch.load(path, map_location=self.device)
        self.episode = checkpoint.get("episode", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        self.train_steps = checkpoint.get("train_steps", 0)
        self._load_checkpoint(checkpoint)

    def _save_checkpoint(self, checkpoint, path):
        """子类覆盖以添加自有参数"""
        torch.save(checkpoint, path)

    def _load_checkpoint(self, checkpoint):
        """子类覆盖以恢复自有参数"""

    # ================================================================
    # 工具
    # ================================================================

    def to_tensor(self, x, dtype=torch.float32):
        """便捷地将 numpy 转为 tensor 并放到正确设备"""
        return torch.tensor(np.asarray(x), dtype=dtype, device=self.device)
