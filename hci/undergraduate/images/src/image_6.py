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

fig6, ax6 = plt.subplots(figsize=(7, 10))
ax6.set_xlim(0, 100)
ax6.set_ylim(0, 100)
ax6.axis("off")
ax6.set_title("系统工作流程")
# 流程元素
flow_items = [
    ("ellipse", 50, 92, 22, 6, "系统启动"),
    ("rect", 50, 80, 38, 7, "加载配置文件\n模型路径 / 参数 / 步长 / 引导轨迹"),
    ("rect", 50, 70, 38, 7, "初始化仿真环境\n载入肌肉骨骼 + 方向盘模型"),
    ("diamond", 50, 58, 36, 10, "物理参数\n自检通过？"),
    ("rect", 50, 46, 38, 7, "获取肌肉激活信号\n键盘指令 / 仿真生成 u(t)"),
    ("rect", 50, 36, 38, 7, "控制算法处理\n滤波 / 死区 / 映射 / 限幅"),
    ("rect", 50, 26, 38, 7, "下发指令 → 仿真推进\n驱动方向盘转动"),
    ("rect", 50, 16, 38, 7, "更新视觉引导界面\n显示目标 / 实际 / 误差"),
    ("diamond", 50, 4, 36, 10, "达到停止条件？"),
    ("ellipse", 10, 4, 18, 6, "输出评估结果")
]
for shape, x, y, w, h, txt in flow_items:
    if shape == "ellipse":
        patch = mpatches.Ellipse((x,y+h/2), w, h, ec="#335599", fc="#e0ecff")
    elif shape == "rect":
        patch = mpatches.Rectangle((x-w/2, y), w, h, ec="black", fc="#fff8cc")
    elif shape == "diamond":
        verts = [[x-w/2,y+h/2],[x,y+h],[x+w/2,y+h/2],[x,y]]
        patch = mpatches.Polygon(verts, ec="#bb4444", fc="#ffe6e6")
    ax6.add_patch(patch)
    ax6.text(x, y+h/2, txt, ha="center", va="center")
# 流程连线
conn_lines = [
    (50,92,50,87), (50,80,50,77), (50,70,50,68),
    (32,58,10,58), (10,58,10,92), (10,92,38,92),
    (50,58,50,53), (50,46,50,41), (50,36,50,31), (50,26,50,21), (50,16,50,14),
    (50,4, 88,4), (88,4,88,46), (88,46,68,46),
    (32,4, 19,4)
]
for x1,y1,x2,y2 in conn_lines:
    ax6.plot([x1,x2], [y1,y2], c="black", lw=1.2)
# 箭头标注
ax6.annotate("", xy=(50,87), xytext=(50,92), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(50,77), xytext=(50,80), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(50,68), xytext=(50,70), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(50,53), xytext=(50,58), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(50,41), xytext=(50,46), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(50,31), xytext=(50,36), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(50,21), xytext=(50,26), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(50,14), xytext=(50,16), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(32,4), xytext=(50,4), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(10,92), xytext=(10,58), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.annotate("", xy=(68,46), xytext=(88,46), arrowprops=dict(arrowstyle="->", lw=1.2))
ax6.text(15, 70, "否")
ax6.text(42, 55, "是")
ax6.text(80, 8, "否")
ax6.text(15, 8, "是")
ax6.text(50, 51, "— 进入主循环 —", ha="center")
save_fig(fig6, "图6_系统工作流程")
plt.show()