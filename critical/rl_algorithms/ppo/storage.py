# rl_algorithms/ppo/storage.py
# 轨迹数据存储（Rollout Buffer），用于 GAE 优势计算

import numpy as np


class RolloutStorage:
    """
    PPO 轨迹存储。

    在 rollout 阶段收集 (s, a, log_prob, r, s', done) 序列，
    在 update 阶段批量取出用于 GAE + 多轮策略更新。
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def push(self, state, action, log_prob, reward, next_state, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(action)
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.dones.append(float(done))

    def get_all(self):
        """返回全部轨迹数据的 numpy 数组"""
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.next_states, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)

    @property
    def total_reward(self):
        return sum(self.rewards)
