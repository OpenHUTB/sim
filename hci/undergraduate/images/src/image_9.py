import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 全局字体配置
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.titlesize"] = 12

def save_fig(fig, name):
    fig.tight_layout(pad=2.0)
    fig.savefig(f"{name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{name}.png", bbox_inches="tight")

fig7, ax7 = plt.subplots(figsize=(16, 6))
ax7.set_xlim(0, 100)
ax7.set_ylim(0, 100)
ax7.axis("off")
ax7.set_title("控制信号处理流水线")
# 六段处理框
pipe_data = [
    (6, 30, 12, 40, "#e0ecf8", r"原始激活信号\n$u(k) \in [0, 1]$\n来自键盘 / 仿真\n含基线漂移、抖动", "输入"),
    (22, 30, 12, 40, "#fff6cc", r"EMA 低通滤波\n$f = \alpha\cdot u + (1-\alpha)\cdot u_f^{pre}$\n$\alpha \in (0, 1]$\n抑制高频抖动", "第 1 级"),
    (38, 30, 12, 40, "#ffe6e6", r"死区过滤\n$u_d = \mathrm{sgn}(u_f)\cdot$\n$\max(|u_f|-u_{th}, 0)$\n消除零附近噪声", "第 2 级"),
    (54, 30, 12, 40, "#e8f4e0", r"非线性增益映射\n$y = K_p \cdot g(u_d)$\n小信号高灵敏度\n大信号渐进饱和", "第 3 级"),
    (70, 30, 12, 40, "#f0e6f8", r"饱和限幅\n$\theta_{cmd} = \mathrm{sat}(y, \pm\theta_{max})$\n防超调与机械碰撞\n硬性安全边界", "第 4 级"),
    (86, 30, 12, 40, "#d9f2f2", "下发至仿真\n驱动肌肉作动器\n$\theta_{cmd} \to$ 关节力矩\n$\to$ 方向盘转动", "输出")
]
for x,y,w,h,color,txt,label in pipe_data:
    ax7.add_patch(mpatches.Rectangle((x,y),w,h, ec="black", fc=color, lw=1.1))
    ax7.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=11)
    ax7.text(x+w/2, 22, label, ha="center", va="top", fontsize=12)
# 箭头连接
for i in range(5):
    x_start = pipe_data[i][0] + pipe_data[i][2]
    x_end = pipe_data[i+1][0]
    ax7.annotate("", xy=(x_end, 50), xytext=(x_start,50), arrowprops=dict(arrowstyle="->", lw=1.6))
# 底部注释
ax7.text(50, 10, "每一级处理对应公式 (2-12) ~ (2-15)，可独立调参，便于实验对比与故障定位。", ha="center", fontsize=11)
save_fig(fig7, "图7_控制信号流水线")
plt.show()