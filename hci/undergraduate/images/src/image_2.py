import matplotlib.pyplot as plt
import numpy as np
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

fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.axis("off")
ax2.set_title("Hill 三元元素肌肉模型 (CE+PE+SE)")
# 左右骨骼竖杆
ax2.plot([5,5], [20,80], lw=4, c="#666666")
ax2.plot([95,95], [20,80], lw=4, c="#666666")
ax2.text(5, 85, "骨骼端", ha="center")
ax2.text(95, 85, "肌腱端", ha="center")
# CE收缩圆
circle_ce = mpatches.Circle((30, 60), 8, ec="#0044bb", fc="#e6f0ff", lw=1.5)
ax2.add_patch(circle_ce)
ax2.text(30, 60, "CE", fontsize=14, weight="bold", c="#0044bb", ha="center")
ax2.annotate("激活信号 a(t)", xy=(30, 68), xytext=(30, 78), arrowprops=dict(arrowstyle="->", lw=1.2), ha="center")
ax2.text(30, 48, r"收缩元 $a\cdot fL(\tilde{l})\cdot fV(\tilde{v})$", ha="center")
# SE串联弹性（肌腱锯齿）
se_x = np.linspace(42, 72, 20)
se_y = np.array([60 + 6*((-1)**i) for i in range(20)])
ax2.plot(se_x, se_y, c="#bb5533", lw=2)
ax2.text(57, 72, "SE 串联弹性元 (肌腱)", ha="center")
# PE并联弹性
pe_x = np.linspace(12, 42, 20)
pe_y = np.array([40 + 6*((-1)**i) for i in range(20)])
ax2.plot(pe_x, pe_y, c="#339933", lw=2)
ax2.text(27, 28, r"PE 并联弹性元 $fP(\tilde{l})$", ha="center")
# 阻尼方块
rect_damp = mpatches.Rectangle((48, 34), 8, 6, ec="#666", fc="#eeeeee")
ax2.add_patch(rect_damp)
ax2.text(52, 43, r"阻尼 $\beta\cdot\tilde{v}$", ha="center")
# 连接线
ax2.plot([5,30], [60,60], c="black")
ax2.plot([38,42], [60,60], c="black")
ax2.plot([72,95], [60,60], c="black")
ax2.plot([5,12], [40,40], c="black")
ax2.plot([42,48], [40,40], c="black")
ax2.plot([56,95], [40,40], c="black")
# 力箭头
ax2.annotate("", xy=(3,60), xytext=(0,60), arrowprops=dict(arrowstyle="->", lw=1.5))
ax2.text(2,63, r"$F$", va="bottom")
ax2.annotate("", xy=(97,60), xytext=(100,60), arrowprops=dict(arrowstyle="->", lw=1.5))
ax2.text(98,63, r"$F$", va="bottom")
# 底部公式框
rect_formula = mpatches.Rectangle((15, 5), 70, 12, ec="#dd7744", fc="#fff8e6", lw=1.2)
ax2.add_patch(rect_formula)
ax2.text(50, 11, r"$F = F_0 \cdot \left[\ a \cdot fL(\tilde{l}) \cdot fV(\tilde{v}) + fP(\tilde{l}) + \beta \cdot \tilde{v}\ \right]$", fontsize=13, ha="center")
ax2.text(50, 7, "主动收缩力 + 被动弹性力 + 阻尼力", ha="center")
save_fig(fig2, "图2_Hill肌肉模型")
plt.show()