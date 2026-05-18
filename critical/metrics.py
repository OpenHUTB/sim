# utils/metrics.py
# 评估指标计算：TTC、THW、碰撞概率、危险等级、成功率
# 供 experiments/evaluate.py 与 visualization/ 调用

import math
from collections import deque

import numpy as np


# ================================================================
# 核心指标
# ================================================================

def compute_ttc(distance, rel_speed):
    """
    Time-To-Collision (秒)。

    TTC = distance / rel_speed，仅当 ego 速度 > 前车速度时有效。
    若 rel_speed <= 0（自车慢于前车），返回 inf。
    """
    if rel_speed <= 0.0:
        return float("inf")
    return distance / rel_speed


def compute_thw(distance, ego_speed):
    """
    Time-Headway (秒)。

    THW = distance / ego_speed，表示以当前速度行驶到达前车位置的时间。
    ego_speed <= 0 时返回 inf。
    """
    if ego_speed <= 0.0:
        return float("inf")
    return distance / ego_speed


def compute_collision_probability(ttc_values, threshold=2.0):
    """
    碰撞概率：TTC 低于阈值的帧数占比。

    ttc_values: list[float] 各帧 TTC
    threshold: TTC 危险阈值 (秒)，默认 2s
    """
    if not ttc_values:
        return 0.0
    dangerous = sum(1 for t in ttc_values if t < threshold)
    return dangerous / len(ttc_values)


# ================================================================
# 危险等级
# ================================================================

def danger_level(ttc, thw):
    """
    综合危险等级 (0-3)。

    0 = 安全
    1 = 低危
    2 = 中危
    3 = 高危

    判定规则:
        TTC < 2s  → +2
        TTC < 5s  → +1
        THW < 1s  → +1
    """
    level = 0
    if ttc < 5.0:
        level += 1
    if ttc < 2.0:
        level += 1
    if thw < 1.0:
        level += 1
    return min(level, 3)


def classify_danger(distribution):
    """
    根据 TTC/THW 分布统计危险等级频率。

    distribution: list[tuple(ttc, thw)]
    返回: dict {0: count, 1: count, 2: count, 3: count}
    """
    levels = {0: 0, 1: 0, 2: 0, 3: 0}
    for ttc, thw in distribution:
        lv = danger_level(ttc, thw)
        levels[lv] += 1
    return levels


# ================================================================
# Episode 统计
# ================================================================

class EpisodeStats:
    """单 episode 内指标累积计算器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_steps = 0
        self.collision_occurred = False
        self.reward_sum = 0.0
        self.rewards = []
        self.ttc_values = []
        self.thw_values = []
        self.distances = []
        self.speeds = []

    def update(self, distance, ego_speed, rel_speed, reward, done, collided):
        self.total_steps += 1
        self.reward_sum += reward
        self.rewards.append(reward)
        self.distances.append(distance)
        self.speeds.append(ego_speed)

        ttc = compute_ttc(distance, rel_speed)
        thw = compute_thw(distance, ego_speed)
        self.ttc_values.append(ttc)
        self.thw_values.append(thw)

        if collided:
            self.collision_occurred = True

    def summary(self):
        """返回该 episode 的统计摘要"""
        n = max(self.total_steps, 1)
        finite_ttc = [t for t in self.ttc_values if math.isfinite(t)]
        finite_thw = [t for t in self.thw_values if math.isfinite(t)]

        return {
            "total_steps": self.total_steps,
            "collision": self.collision_occurred,
            "total_reward": self.reward_sum,
            "mean_reward": self.reward_sum / n,
            "min_distance": min(self.distances) if self.distances else float("inf"),
            "mean_distance": np.mean(self.distances) if self.distances else 0.0,
            "min_ttc": min(finite_ttc) if finite_ttc else float("inf"),
            "mean_ttc": np.mean(finite_ttc) if finite_ttc else float("inf"),
            "min_thw": min(finite_thw) if finite_thw else float("inf"),
            "mean_thw": np.mean(finite_thw) if finite_thw else float("inf"),
            "collision_probability": compute_collision_probability(self.ttc_values),
            "max_danger_level": max(
                danger_level(t, h) for t, h in zip(self.ttc_values, self.thw_values)
            ) if self.ttc_values else 0,
        }


# ================================================================
# 多 Episode 汇总
# ================================================================

def aggregate_episodes(episode_summaries):
    """
    汇总多个 episode 的评估结果。

    episode_summaries: list[dict]，来自 EpisodeStats.summary()
    返回: dict
    """
    if not episode_summaries:
        return {}

    total = len(episode_summaries)
    collisions = sum(1 for s in episode_summaries if s["collision"])
    success = total - collisions

    rewards = [s["total_reward"] for s in episode_summaries]
    min_ttcs = [s["min_ttc"] for s in episode_summaries if math.isfinite(s["min_ttc"])]
    mean_ttcs = [s["mean_ttc"] for s in episode_summaries if math.isfinite(s["mean_ttc"])]
    min_distances = [s["min_distance"] for s in episode_summaries]
    danger_maxs = [s["max_danger_level"] for s in episode_summaries]

    return {
        "total_episodes": total,
        "collisions": collisions,
        "success_count": success,
        "collision_rate": collisions / total,
        "success_rate": success / total,
        "mean_total_reward": np.mean(rewards),
        "std_total_reward": np.std(rewards),
        "mean_min_ttc": np.mean(min_ttcs) if min_ttcs else float("inf"),
        "mean_min_distance": np.mean(min_distances),
        "mean_max_danger_level": np.mean(danger_maxs),
        "danger_level_distribution": {
            lv: danger_maxs.count(lv) / total
            for lv in range(4)
        },
    }


# ================================================================
# 实时平滑
# ================================================================

class MovingAverage:
    """滑动窗口平均，用于平滑训练曲线"""

    def __init__(self, window_size=100):
        self.window = deque(maxlen=window_size)

    def update(self, value):
        self.window.append(value)
        return self.mean()

    def mean(self):
        if not self.window:
            return 0.0
        return np.mean(self.window)

    def std(self):
        if len(self.window) < 2:
            return 0.0
        return np.std(self.window)
