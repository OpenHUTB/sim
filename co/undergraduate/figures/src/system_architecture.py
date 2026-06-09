# -*- coding: utf-8 -*-
"""
使用 Python + Matplotlib 生成论文流程图。
依赖：
    pip install matplotlib
运行：
    python 当前脚本.py
输出：
    PNG、PDF、SVG 三种格式图片。

说明：
    为保证中文稳定显示，脚本优先调用 Noto Sans CJK SC 字体；
    Windows 环境可改为 Microsoft YaHei 或 SimHei。
"""
import matplotlib as mpl
from matplotlib import font_manager

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if __import__("os").path.exists(p)), None)
if FONT_PATH:
    zh_font = font_manager.FontProperties(fname=FONT_PATH)
else:
    zh_font = font_manager.FontProperties(family="sans-serif")

mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def t(ax, x, y, s, **kwargs):
    kwargs.setdefault("fontproperties", zh_font)
    ax.text(x, y, s, **kwargs)

def rounded_box(ax, x, y, w, h, title, subtitle, fc, ec):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.09",
        linewidth=1.35,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--"
    )
    ax.add_patch(box)
    t(ax, x + w/2, y + h*0.62, title, ha="center", va="center",
      fontsize=17, fontweight="bold")
    t(ax, x + w/2, y + h*0.31, subtitle, ha="center", va="center",
      fontsize=10.8)

def arrow(ax, start, end, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color="black", shrinkA=0, shrinkB=0
    ))

fig, ax = plt.subplots(figsize=(7.2, 10.2), dpi=220)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

x, w, h = 0.17, 0.72, 0.082
ys = [0.875, 0.745, 0.615, 0.485, 0.355, 0.225, 0.095]
items = [
    ("飞机辅助感知层", "（身份确认、目标位姿、最近障碍距离）", "#eaf3ff", "#5c8fd9"),
    ("任务与模式管理层", "（搜索 / 接近 / 对准 / 微动 / 停车 / 中止）", "#edf8e8", "#74ad5d"),
    ("局部轨迹生成层", "（预对接点、对接点、候选轨迹、可行性筛选）", "#fff4cc", "#f4ad28"),
    ("误差反馈控制层", r"（$e_x,\ e_y,\ e_\theta \rightarrow v_d,\ \omega_d$）", "#f0eafd", "#8062c7"),
    ("4WS映射与约束处理层", r"（$v_d,\ \omega_d \rightarrow \delta_f,\ \delta_r$，轮速）", "#eaf3ff", "#5c8fd9"),
    ("Gazebo执行层", "（转向关节、车轮速度控制器）", "#ffe9db", "#ec7e3b"),
    ("状态反馈 / 感知更新", "", "#f2f2f2", "#8a8a8a"),
]

for y, item in zip(ys, items):
    rounded_box(ax, x, y, w, h, *item)

for i in range(len(ys)-1):
    arrow(ax, (0.53, ys[i]), (0.53, ys[i+1]+h))

arrow(ax, (x, ys[-1] + h*0.45), (0.08, ys[-1] + h*0.45), lw=1.45)
ax.plot([0.08, 0.08, x], [ys[-1] + h*0.45, ys[0] + h*0.55, ys[0] + h*0.55],
        color="black", linewidth=1.45)
arrow(ax, (0.08, ys[0] + h*0.55), (x, ys[0] + h*0.55), lw=1.45)

plt.tight_layout()
for ext in ["png", "pdf", "svg"]:
    plt.savefig(f"figures/system_architecture.{ext}", bbox_inches="tight")
