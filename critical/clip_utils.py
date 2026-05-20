# rl_algorithms/ppo/clip_utils.py
# 标准裁剪与平滑裁剪函数（Smooth-PPO 创新点）

import torch


def standard_clip(ratio, eps_clip):
    """
    标准 PPO 裁剪：
        clip(ratio, 1-ε, 1+ε)

    在边界处梯度直接截断。
    """
    return torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)


def smooth_clip(ratio, eps_clip, alpha=0.1):
    """
    平滑裁剪（毕设创新点）。

    在裁剪边界的过渡区间 [ε-α, ε] 内使用二次插值替代硬截断，
    使梯度在边界附近连续变化，避免突变，从而：
      - 提升策略更新的稳定性
      - 降低行为抖动，提高"行为自然度"
      - 适合车辆对抗（加塞/急刹）等需要平稳动作的场景

    参数:
        ratio:    重要性采样比率 r(θ) = π_new / π_old
        eps_clip: 裁剪范围 ε
        alpha:    平滑过渡宽度（默认 0.1 或从 config 读取）
    """
    low = 1.0 - eps_clip            # 下界
    high = 1.0 + eps_clip           # 上界

    # 过渡区间
    low_smooth = low + alpha
    high_smooth = high - alpha

    # 基础 clamp
    clipped = torch.clamp(ratio, low, high)

    # 下界的平滑过渡 [low, low_smooth]
    mask_low = (ratio > low) & (ratio < low_smooth)
    if mask_low.any():
        t = (ratio[mask_low] - low) / alpha     # [0, 1]
        clipped[mask_low] = low + alpha * (t ** 2)

    # 上界的平滑过渡 [high_smooth, high]
    mask_high = (ratio > high_smooth) & (ratio < high)
    if mask_high.any():
        t = (high - ratio[mask_high]) / alpha    # [0, 1]
        clipped[mask_high] = high - alpha * (t ** 2)

    return clipped
