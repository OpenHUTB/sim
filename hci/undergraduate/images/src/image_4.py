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

fig4, ax4 = plt.subplots(figsize=(18, 6))
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 100)
ax4.axis("off")
ax4.set_title("视觉引导下肌肉驱动方向盘操控的闭环反馈结构")
# 所有模块框
block_info = [
    (8, 45, 12, 20, r"目标轨迹\n$\theta^*(t)$"),
    (28, 42, 18, 26, r"操作者神经控制\n肌肉激活 $u(t)$"),
    (58, 42, 18, 26, "信号处理与映射\n滤波/死区/增益"),
    (82, 42, 14, 26, "肌肉骨骼—方向盘\nMuJoCo 动力学")
]
for x, y, w, h, txt in block_info:
    ax4.add_patch(mpatches.Rectangle((x, y), w, h, ec="black", fc="#f8f8f8"))
    ax4.text(x+w/2, y+h/2, txt, ha="center", va="center")
# 求和圆圈
circ_sum = mpatches.Circle((22, 55), 6, ec="black", fc="white")
ax4.add_patch(circ_sum)
ax4.text(22, 59, r"$+$", fontsize=14)
ax4.text(22, 49, r"$-$", fontsize=14)
ax4.text(16, 55, r"$e(t)$", va="center")
# 主信号线
ax4.annotate("", xy=(22,55), xytext=(20,55), arrowprops=dict(arrowstyle="->", lw=1.4))
ax4.annotate("", xy=(28,55), xytext=(34,55), arrowprops=dict(arrowstyle="->", lw=1.4))
ax4.text(46, 57, r"$u(t)$")
ax4.annotate("", xy=(58,55), xytext=(64,55), arrowprops=dict(arrowstyle="->", lw=1.4))
ax4.text(76, 57, r"$\theta_{cmd}$")
ax4.annotate("", xy=(82,55), xytext=(96,55), arrowprops=dict(arrowstyle="->", lw=1.4))
ax4.text(97, 57, r"$\theta(t)$")
# 反馈线：角度反馈
ax4.plot([96,96], [55,18], c="black")
ax4.plot([96,22], [18,18], c="black")
ax4.annotate("", xy=(22,18), xytext=(22,50), arrowprops=dict(arrowstyle="->", lw=1.4))
ax4.text(50, 12, r"实时角度反馈 $\theta(t)$", ha="center")
# 视觉显示分支
ax4.plot([37,37], [55,28], c="black")
ax4.annotate("", xy=(37,28), xytext=(48,28), arrowprops=dict(arrowstyle="->", lw=1.4))
ax4.text(52, 28, "视觉显示 (目标 / 当前 / 误差)", va="center")
save_fig(fig4, "图4_闭环反馈框图")
plt.show()