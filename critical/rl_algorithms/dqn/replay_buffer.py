# rl_algorithms/dqn/replay_buffer.py
# 经验回放池：优先从容量缓冲区中均匀采样

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """固定容量的经验回放池 (FIFO)"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """均匀随机采样，返回 numpy 数组"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, min_size):
        """回放池是否已达到最低采样量"""
        return len(self.buffer) >= min_size
