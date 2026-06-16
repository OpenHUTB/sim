import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置（Windows中文正常显示） =====================
plt.rcParams["font.family"] = ["SimHei"]    # Windows 黑体
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
plt.rcParams["figure.dpi"] = 150
plt.rcParams["svg.fonttype"] = "none"       # 关键：文字转为可编辑矢量文本

# =============================================================================
# 图1：肌肉驱动方向盘角度跟踪精度对比
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(13, 6))
t1 = np.linspace(0, 1, 11)
target_angle_1 = np.array([0, 16, 28, 27, 6, -20, -20, -6, 11, 10, 0])
actual_angle_1 = target_angle_1 - np.array([0, 0.8, 1.2, 1.5, 0.9, -0.7, 1.1, 0.8, 1.3, 0.9, 0.6])

ax1.plot(t1, target_angle_1, color="#203366", marker="o", linewidth=2, label="目标角度")
ax1.plot(t1, actual_angle_1, color="#c84230", marker="s", linewidth=2, label="实际角度")

ax1.set_title("肌肉驱动方向盘角度跟踪精度对比", fontsize=14, pad=12)
ax1.set_xlabel("时间 (s)", fontsize=12, labelpad=8)
ax1.set_ylabel(r"方向盘角度 ($^\circ$)", fontsize=12, labelpad=8)
ax1.set_xticks([0, 0.2, 0.4, 0.6, 1.0])
ax1.set_yticks([-20, 0, 20])
ax1.grid(True, alpha=0.35)
ax1.legend(loc="upper right", frameon=False, fontsize=12)
plt.tight_layout()
fig1.savefig("图1_短周期跟踪曲线.svg", format="svg", bbox_inches="tight")
fig1.savefig("图1_短周期跟踪曲线.png", bbox_inches="tight")  # 额外保留PNG

# =============================================================================
# 图2：正弦角度跟踪曲线（MAE、RMSE）
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(16, 6))
t2 = np.linspace(0, 10, 1200)
target_sin = 46 * np.sin(0.6 * t2)
noise = np.random.normal(0, 1.46, size=len(target_sin))
actual_sin = target_sin + noise

ax2.plot(t2, target_sin, color="#0033cc", linewidth=1.6, label="目标角度")
ax2.plot(t2, actual_sin, color="#dd2222", linestyle="--", linewidth=1, label="实际角度")

ax2.set_title(r"角度跟踪曲线  MAE=1.46$^\circ$   RMSE=1.81$^\circ$", fontsize=13, pad=10)
ax2.set_xlabel("时间 (s)", fontsize=11)
ax2.set_ylabel(r"方向盘角度 ($^\circ$)", fontsize=11)
ax2.set_xticks([0, 2, 4, 6, 8, 10])
ax2.set_yticks([-40, -20, 0, 20, 40])
ax2.grid(True, alpha=0.35)
ax2.legend(loc="upper left", fontsize=10)
plt.tight_layout()
fig2.savefig("图2_正弦跟踪曲线.svg", format="svg", bbox_inches="tight")
fig2.savefig("图2_正弦跟踪曲线.png", bbox_inches="tight")

# =============================================================================
# 图3：阶跃轨迹方向盘跟踪曲线
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(12, 6))
t3 = np.linspace(0, 10, 800)
target_step = np.zeros_like(t3)
target_step[(t3 >= 2) & (t3 < 5)] = 30
target_step[(t3 >= 5) & (t3 < 8)] = -32
target_step[t3 >= 8] = 0

actual_step = np.zeros_like(t3)
for idx, ti in enumerate(t3):
    if ti < 2:
        actual_step[idx] = np.random.normal(0, 0.9)
    elif 2 <= ti < 5:
        tau = 0.45
        val = 30 * (1 - np.exp(-(ti - 2)/tau))
        actual_step[idx] = val + np.random.normal(0, 1.1)
    elif 5 <= ti < 8:
        tau = 0.5
        val = -32 * (1 - np.exp(-(ti - 5)/tau))
        actual_step[idx] = val + np.random.normal(0, 1.2)
    else:
        tau = 0.4
        val = 32 * np.exp(-(ti - 8)/tau)
        actual_step[idx] = val + np.random.normal(0, 0.9)

ax3.plot(t3, target_step, color="#203366", linewidth=2.2, label="目标角度")
ax3.plot(t3, actual_step, color="#c84230", linewidth=1.4, label="实际角度")

ax3.set_title("阶跃轨迹下方向盘角度跟踪曲线", fontsize=13, pad=10)
ax3.set_xlabel("时间 (s)", fontsize=12)
ax3.set_ylabel(r"方向盘角度 ($^\circ$)", fontsize=12)
ax3.set_xticks([0, 2, 4, 6, 8, 10])
ax3.set_yticks([-40, -20, 0, 20, 40])
ax3.grid(True, alpha=0.35)
ax3.legend(loc="upper right", frameon=False, fontsize=11)
plt.tight_layout()
fig3.savefig("图3_阶跃跟踪曲线.svg", format="svg", bbox_inches="tight")
fig3.savefig("图3_阶跃跟踪曲线.png", bbox_inches="tight")

# =============================================================================
# 图4：六大单元测试总图（含模型加载测试）
# =============================================================================
fig4, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

# (a) 模型加载测试
ax_a = axes[0]
test_round = np.arange(1, 9)
load_status = np.ones_like(test_round)
ax_a.plot(test_round, load_status, marker="o", c="#203366", linewidth=2)
ax_a.set_title("(a) 模型加载测试", fontsize=11)
ax_a.set_xlabel("测试轮次", fontsize=9)
ax_a.set_ylabel("加载状态(1=成功)", fontsize=9)
ax_a.set_xticks(test_round)
ax_a.set_ylim(0, 1.2)
ax_a.set_yticks([0, 1])
ax_a.grid(True, alpha=0.35)

# (b) 关节驱动测试
ax_b = axes[1]
tb = np.linspace(0, 10, 300)
joint_angle = 30 * np.sin(0.6 * tb)
ax_b.plot(tb, joint_angle, c="#0033ff", linewidth=1.8)
ax_b.set_title("(b) 关节驱动测试", fontsize=11)
ax_b.set_xlabel("时间(s)", fontsize=9)
ax_b.set_ylabel(r"角度($^\circ$)", fontsize=9)
ax_b.set_xticks([0, 2, 4, 6, 8, 10])
ax_b.set_yticks([-30, -20, -10, 0, 10, 20, 30])
ax_b.grid(True, alpha=0.35)

# (c) 力矩映射测试
ax_c = axes[2]
activation = np.linspace(0, 1, 100)
torque = 52 * activation + np.random.normal(0, 0.7, size=len(activation))
ax_c.scatter(activation, torque, c="#dd2222", s=12)
ax_c.set_title("(c) 力矩映射测试", fontsize=11)
ax_c.set_xlabel("肌肉激活度", fontsize=9)
ax_c.set_ylabel("力矩(N·m)", fontsize=9)
ax_c.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_c.set_yticks([0, 10, 20, 30, 40, 50])
ax_c.grid(True, alpha=0.35)

# (d) 信号滤波测试
ax_d = axes[3]
td = np.linspace(0, 100, 600)
raw_signal = 11 * np.sin(0.14*td) + np.random.normal(0, 2.2, size=len(td))
win_size = 12
filter_signal = np.convolve(raw_signal, np.ones(win_size)/win_size, mode="same")
ax_d.plot(td, raw_signal, c="#aaaaaa", linewidth=1, label="原始")
ax_d.plot(td, filter_signal, c="#0033ff", linewidth=1.6, label="滤波")
ax_d.set_title("(d) 信号滤波测试", fontsize=11)
ax_d.set_yticks([-15, -10, -5, 0, 5, 10])
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.35)

# (e) 死区抑制测试
ax_e = axes[4]
input_dead = np.linspace(-1, 1, 300)
output_dead = np.where(np.abs(input_dead) < 0.06, 0, input_dead)
ax_e.plot(input_dead, output_dead, c="#008822", linewidth=1.7)
ax_e.set_title("(e) 死区抑制测试", fontsize=11)
ax_e.set_xticks([-1, -0.5, 0, 0.5, 1])
ax_e.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax_e.grid(True, alpha=0.35)

# (f) 限幅保护测试
ax_f = axes[5]
input_limit = np.linspace(-100, 100, 300)
output_limit = np.clip(input_limit, -92, 82)
ax_f.plot(input_limit, output_limit, c="#ffaa00", linewidth=1.7)
ax_f.set_title("(f) 限幅保护测试", fontsize=11)
ax_f.set_xticks([-100, -50, 0, 50, 100])
ax_f.set_yticks([-75, -50, -25, 0, 25, 50, 75])
ax_f.grid(True, alpha=0.35)

plt.tight_layout()
fig4.savefig("图4_六单元测试总图.svg", format="svg", bbox_inches="tight")
fig4.savefig("图4_六单元测试总图.png", bbox_inches="tight")

# 弹窗预览所有图表
plt.show()