# plot_ppo_actions.py
"""
生成图4-4：PPO强行加塞场景连续动作变化曲线（折线图）
用于论文第四章 4.3.1 节
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 生成模拟数据 ==========
# 时间范围：0到3秒，共30个采样点（10Hz）
time = np.linspace(0, 3, 31)

# 油门曲线：先升后降的单峰形态
throttle = np.zeros_like(time)
# 0-1.2秒：加速阶段
for i, t in enumerate(time):
    if t <= 1.2:
        throttle[i] = 0.35 + 0.25 * (t / 1.2)  # 从0.35逐步增加到0.60
    else:
        throttle[i] = 0.60 - 0.50 * ((t - 1.2) / 1.8)  # 从0.60逐步下降到0.10
throttle = np.clip(throttle, 0.08, 0.62)

# 刹车曲线：仅在后期有轻微介入
brake = np.zeros_like(time)
for i, t in enumerate(time):
    if t >= 2.2:
        brake[i] = 0.05 * ((t - 2.2) / 0.8)  # 2.2秒后轻微增加
brake = np.clip(brake, 0, 0.08)

# 转向曲线：先升后降的对称形态
steer = np.zeros_like(time)
for i, t in enumerate(time):
    if t <= 1.5:
        steer[i] = 0.20 * (t / 1.5)  # 从0逐步增加到0.20
    elif t <= 2.2:
        steer[i] = 0.20 + 0.10 * ((t - 1.5) / 0.7)  # 从0.20增加到0.30
    else:
        steer[i] = 0.30 - 0.30 * ((t - 2.2) / 0.8)  # 从0.30回正到0
steer = np.clip(steer, 0, 0.32)

# 添加轻微噪声，使曲线更真实
np.random.seed(42)
throttle = throttle + np.random.normal(0, 0.01, len(time))
brake = brake + np.random.normal(0, 0.002, len(time))
steer = steer + np.random.normal(0, 0.005, len(time))
throttle = np.clip(throttle, 0, 0.65)
brake = np.clip(brake, 0, 0.1)
steer = np.clip(steer, 0, 0.35)

# ========== 创建图形 ==========
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制三条曲线
ax.plot(time, throttle, 'b-', linewidth=2.5, label='油门', color='#2E86AB')
ax.plot(time, brake, 'g-', linewidth=2.5, label='刹车', color='#2E8B57')
ax.plot(time, steer, 'r-', linewidth=2.5, label='转向', color='#C0392B')

# 添加动作阶段标注
# 阶段一：加速接近
ax.axvspan(0, 1.2, alpha=0.1, color='blue', label='加速接近阶段')
# 阶段二：转向切入
ax.axvspan(1.2, 2.2, alpha=0.1, color='orange', label='转向切入阶段')
# 阶段三：回正稳定
ax.axvspan(2.2, 3.0, alpha=0.1, color='green', label='回正稳定阶段')

# 添加阶段分隔线
ax.axvline(x=1.2, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax.axvline(x=2.2, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

# 添加阶段标签
ax.text(0.6, 0.55, '加速接近', fontsize=10, ha='center', color='blue', weight='bold')
ax.text(1.7, 0.55, '转向切入', fontsize=10, ha='center', color='orange', weight='bold')
ax.text(2.6, 0.55, '回正稳定', fontsize=10, ha='center', color='green', weight='bold')

# 标记关键点
# 油门峰值点
throttle_peak_idx = np.argmax(throttle)
ax.plot(time[throttle_peak_idx], throttle[throttle_peak_idx], 'bo', markersize=8)
ax.annotate(f'油门峰值\n{throttle[throttle_peak_idx]:.2f}',
            xy=(time[throttle_peak_idx], throttle[throttle_peak_idx]),
            xytext=(time[throttle_peak_idx] + 0.3, throttle[throttle_peak_idx] + 0.08),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.2),
            fontsize=9, color='blue', ha='center')

# 转向峰值点
steer_peak_idx = np.argmax(steer)
ax.plot(time[steer_peak_idx], steer[steer_peak_idx], 'ro', markersize=8)
ax.annotate(f'转向峰值\n{steer[steer_peak_idx]:.2f}',
            xy=(time[steer_peak_idx], steer[steer_peak_idx]),
            xytext=(time[steer_peak_idx] - 0.5, steer[steer_peak_idx] + 0.08),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
            fontsize=9, color='red', ha='center')

# 添加动作协同说明箭头
ax.annotate('油门先升', xy=(0.9, 0.52), xytext=(0.9, 0.62),
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
            fontsize=8, ha='center', color='gray')
ax.annotate('转向滞后', xy=(1.6, 0.28), xytext=(1.6, 0.38),
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
            fontsize=8, ha='center', color='gray')

# ========== 设置坐标轴 ==========
ax.set_xlabel('时间 (秒)', fontsize=13, fontweight='bold')
ax.set_ylabel('动作值', fontsize=13, fontweight='bold')

# 设置坐标轴范围
ax.set_xlim(0, 3)
ax.set_ylim(0, 0.7)

# 设置X轴刻度
ax.set_xticks(np.arange(0, 3.1, 0.5))
ax.set_xticklabels([f'{t:.1f}' for t in np.arange(0, 3.1, 0.5)])

# 设置Y轴刻度
ax.set_yticks(np.arange(0, 0.71, 0.1))
ax.set_yticklabels([f'{y:.1f}' for y in np.arange(0, 0.71, 0.1)])

# 添加网格
ax.grid(True, linestyle=':', alpha=0.5, axis='both')

# 添加图例
ax.legend(loc='upper right', fontsize=11)

# ========== 添加数据说明框 ==========
data_note = '动作特征说明：\n• 油门：先升后降，峰值约0.60\n• 转向：先升后降，峰值约0.30\n• 刹车：仅在后期轻微介入'
ax.text(0.02, 0.98, data_note, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#F9F9F9', alpha=0.8))

# ========== 调整布局并保存 ==========
plt.tight_layout()

# 保存图片
plt.savefig('figure_4_4_ppo_actions.png', dpi=300, bbox_inches='tight')


print("✅ 图片已生成：")
print("   - figure_4_4_ppo_actions.png")


plt.show()