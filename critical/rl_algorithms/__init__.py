# rl_algorithms/__init__.py
# 强化学习算法模块统一入口

from .base_agent import BaseAgent

from .dqn.dqn_agent import DQNAgent
from .dqn.attention_agent import AttentionDQNAgent

from .ppo.ppo_agent import PPOAgent
from .ppo.smooth_agent import SmoothPPOAgent

# 算法注册表
ALGORITHM_REGISTRY = {
    "dqn": DQNAgent,
    "attention_dqn": AttentionDQNAgent,
    "ppo": PPOAgent,
    "smooth_ppo": SmoothPPOAgent,
}


def create_agent(algo_name, state_size=None, action_size=None):
    """根据名称创建智能体实例"""
    cls = ALGORITHM_REGISTRY.get(algo_name.lower())
    if cls is None:
        raise KeyError("未知算法: %s，可用: %s"
                       % (algo_name, list(ALGORITHM_REGISTRY.keys())))
    return cls(state_size=state_size, action_size=action_size)
