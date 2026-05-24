# env/reward.py
# 通用奖励函数：仅包含全局统一的基础奖惩规则
# 按 env/CLAUDE.md 规范 —— 不绑定任何具体危险场景

import numpy as np

from config.carla_config import MIN_SAFE_DISTANCE, COLLISION_PENALTY, SUCCESS_REWARD
from utils.metrics import compute_ttc, compute_thw


class RewardCalculator:
    """
    通用奖励计算器。

    基础奖励分量（所有场景、所有算法共用）:
      safety_distance  — 安全距离奖励
      ttc             — TTC 碰撞时间预警
      collision       — 碰撞大额惩罚
      lane_deviation  — 偏离车道惩罚
      speed_limit     — 超速惩罚
      goal            — 安全完成正向激励
    """

    def __init__(self, weights=None):
        self.prev_distance = None
        self.cumulative_reward = 0.0
        self.step_rewards = []

        # 默认权重（场景可覆盖）
        self.weights = weights or {
            "safety_distance": 0.30,
            "ttc": 0.25,
            "collision": 1.0,
            "lane_deviation": 0.10,
            "speed_limit": 0.05,
            "goal": 1.0,
        }

    # ================================================================
    # Public API
    # ================================================================

    def reset(self):
        self.prev_distance = None
        self.cumulative_reward = 0.0
        self.step_rewards.clear()

    def compute(self, env):
        """计算当前步即时奖励"""
        w = self.weights

        ego_speed = env._get_ego_speed()
        distance = env._get_distance()
        rel_speed = env._get_rel_speed()
        collided = (env.collision_sensor.collided
                    if env.collision_sensor else False)
        done = env._is_done()

        r_safety = self._safety_distance_reward(distance)
        r_ttc = self._ttc_reward(distance, abs(rel_speed))
        r_collision = COLLISION_PENALTY if collided else 0.0
        r_lane = self._lane_deviation_reward(env)
        r_speed_limit = self._speed_limit_reward(ego_speed)
        r_goal = SUCCESS_REWARD if (done and not collided
                                     and env._step_count >= env._max_steps) else 0.0

        reward = (
            w["safety_distance"] * r_safety
            + w["ttc"] * r_ttc
            + w["collision"] * r_collision
            + w["lane_deviation"] * r_lane
            + w["speed_limit"] * r_speed_limit
            + w["goal"] * r_goal
        )

        self.prev_distance = distance
        self.cumulative_reward += reward
        self.step_rewards.append(reward)
        return reward

    # ================================================================
    # 通用奖励分量
    # ================================================================

    @staticmethod
    def _safety_distance_reward(distance):
        """安全距离: 10~30m 最佳，<3m 重罚"""
        d_min = MIN_SAFE_DISTANCE
        d_opt_low, d_opt_high = 10.0, 30.0

        if distance < d_min:
            return -2.0 * (d_min - distance) / d_min
        if distance <= d_opt_low:
            return 0.2 + 0.3 * (distance - d_min) / (d_opt_low - d_min)
        if distance <= d_opt_high:
            return 0.5
        return 0.5 * np.exp(-0.02 * (distance - d_opt_high))

    @staticmethod
    def _ttc_reward(distance, rel_speed):
        """TTC 预警: <2s 重罚, 2~5s 轻罚, >5s 安全"""
        ttc = compute_ttc(distance, rel_speed)
        if not np.isfinite(ttc):
            return 0.3
        if ttc < 2.0:
            return -3.0 * (2.0 - ttc) / 2.0
        if ttc < 5.0:
            return -0.5 * (5.0 - ttc) / 3.0
        if ttc <= 10.0:
            return 0.3 * (ttc - 5.0) / 5.0
        return 0.3

    @staticmethod
    def _lane_deviation_reward(env):
        """偏离车道惩罚（通用安全约束）"""
        ego = env.ego_vehicle
        if ego is None:
            return 0.0
        try:
            from utils.geometry_utils import get_lane_offset
            lateral = abs(get_lane_offset(ego, env.world))
            if lateral > 2.0:
                return -0.1 * lateral
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _speed_limit_reward(ego_speed):
        """超速惩罚（全局速度上限）"""
        from config.carla_config import VEHICLE_MAX_SPEED
        limit_ms = VEHICLE_MAX_SPEED / 3.6
        if ego_speed > limit_ms:
            return -0.2 * (ego_speed - limit_ms) / limit_ms
        return 0.0

    # ================================================================
    # 扩展接口：上层场景可注入自定义权重
    # ================================================================

    def set_weights(self, weights):
        """场景层可调用此方法调整奖励权重"""
        self.weights.update(weights)
