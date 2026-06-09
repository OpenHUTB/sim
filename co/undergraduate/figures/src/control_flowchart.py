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
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, FancyArrowPatch

def t(ax, x, y, s, **kwargs):
    kwargs.setdefault("fontproperties", zh_font)
    ax.text(x, y, s, **kwargs)

def process(ax, cx, cy, w, h, text, fc="#eef5ff", ec="#2d65b8", fs=10):
    ax.add_patch(Rectangle((cx-w/2, cy-h/2), w, h, linewidth=1.05, edgecolor=ec, facecolor=fc))
    t(ax, cx, cy, text, ha="center", va="center", fontsize=fs)

def rounded(ax, cx, cy, w, h, text, fc="#fff6df", ec="#222", fs=10):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.05, edgecolor=ec, facecolor=fc))
    t(ax, cx, cy, text, ha="center", va="center", fontsize=fs)

def decision(ax, cx, cy, w, h, text, fc="#f1fbef", ec="#19753b", fs=9.5):
    pts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.05, edgecolor=ec, facecolor=fc))
    t(ax, cx, cy, text, ha="center", va="center", fontsize=fs)

def arrow(ax, start, end, lw=1.2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color="black", shrinkA=0, shrinkB=0))

fig, ax = plt.subplots(figsize=(7.4, 11.6), dpi=220)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

cx = 0.50
w_proc = 0.40
h_proc = 0.041
ys = {
    "start": 0.965,
    "sense": 0.895,
    "valid": 0.815,
    "points": 0.745,
    "mode": 0.685,
    "traj": 0.625,
    "feasible": 0.555,
    "ref": 0.485,
    "err": 0.435,
    "speed": 0.385,
    "vdwd": 0.335,
    "map": 0.285,
    "limit": 0.235,
    "publish": 0.185,
    "stop_dec": 0.115,
    "loop": 0.045,
}

rounded(ax, cx, ys["start"], 0.16, 0.036, "开始")
process(ax, cx, ys["sense"], w_proc, 0.052, "读取感知信息\n（身份、位姿、障碍距离）")
decision(ax, cx, ys["valid"], 0.29, 0.075, "目标身份\n是否有效？")
process(ax, cx, ys["points"], w_proc, h_proc, "生成预对接点与对接点")
process(ax, cx, ys["mode"], w_proc, 0.052, "当前模式选择目标点\n（接近 / 对准 / 微动）")
process(ax, cx, ys["traj"], w_proc, h_proc, "生成候选局部轨迹")
decision(ax, cx, ys["feasible"], 0.33, 0.07, "可行性筛选\n（曲率 / 安全距离）", fc="#fffaf0", ec="#e28a00")
process(ax, cx, ys["ref"], w_proc, h_proc, "提取参考状态与参考曲率", fc="#f4efff", ec="#7957c1")
process(ax, cx, ys["err"], w_proc, h_proc, "计算误差（$e_x, e_y, e_θ$）", fc="#f4efff", ec="#7957c1")
process(ax, cx, ys["speed"], w_proc, h_proc, "速度调度", fc="#f4efff", ec="#7957c1")
process(ax, cx, ys["vdwd"], w_proc, h_proc, "计算 $v_d$ 与 $ω_d$", fc="#f4efff", ec="#7957c1")
process(ax, cx, ys["map"], w_proc, h_proc, "映射为 $δ_f, δ_r$", fc="#f4efff", ec="#7957c1")
process(ax, cx, ys["limit"], w_proc, h_proc, "限幅与斜率限制", fc="#f4efff", ec="#7957c1")
process(ax, cx, ys["publish"], w_proc, h_proc, "发布控制命令", fc="#f4efff", ec="#7957c1")
decision(ax, cx, ys["stop_dec"], 0.31, 0.065, "是否进入\n停车集合 $Ω_t$？")
rounded(ax, cx, ys["loop"], 0.28, 0.040, "循环执行", fc="#fff8e9", ec="#bd7b00")

process(ax, 0.78, ys["valid"], 0.18, 0.046, "保持等待", fc="#fff1f1", ec="#d93333")
process(ax, 0.78, ys["feasible"], 0.18, 0.046, "中止 / 停车", fc="#fff1f1", ec="#d93333")
process(ax, 0.80, ys["stop_dec"], 0.22, 0.060, "零速停车\n（$v=0, δ_f=0, δ_r=0$）", fc="#effaf0", ec="#3b8b45")

sequence = ["start","sense","valid","points","mode","traj","feasible","ref","err","speed","vdwd","map","limit","publish","stop_dec","loop"]
for a, b in zip(sequence[:-1], sequence[1:]):
    if a in ["valid", "feasible", "stop_dec"]:
        arrow(ax, (cx, ys[a]-0.037), (cx, ys[b]+(0.026 if b in ["points","ref"] else 0.021)))
    else:
        arrow(ax, (cx, ys[a]-0.026), (cx, ys[b]+0.026))

t(ax, 0.462, 0.776, "是", fontsize=10, color="green")
t(ax, 0.608, ys["valid"]+0.020, "否", fontsize=10, color="red")
arrow(ax, (cx+0.145, ys["valid"]), (0.69, ys["valid"]))

t(ax, 0.462, 0.513, "有", fontsize=10, color="green")
t(ax, 0.608, ys["feasible"]+0.020, "无", fontsize=10, color="red")
arrow(ax, (cx+0.165, ys["feasible"]), (0.69, ys["feasible"]))

t(ax, 0.608, ys["stop_dec"]+0.016, "是", fontsize=10, color="green")
t(ax, 0.462, ys["stop_dec"]-0.053, "否", fontsize=10, color="red")
arrow(ax, (cx+0.155, ys["stop_dec"]), (0.69, ys["stop_dec"]))

ax.plot([0.87, 0.95, 0.95, 0.70], [ys["valid"], ys["valid"], ys["sense"], ys["sense"]],
        color="black", linewidth=1.1)
arrow(ax, (0.70, ys["sense"]), (cx+w_proc/2, ys["sense"]))

ax.plot([0.87, 0.95, 0.95, 0.70], [ys["feasible"], ys["feasible"], ys["sense"], ys["sense"]],
        color="black", linewidth=1.1)

ax.plot([cx, 0.06, 0.06, 0.30], [ys["loop"], ys["loop"], ys["sense"], ys["sense"]],
        color="black", linewidth=1.1)
arrow(ax, (0.30, ys["sense"]), (cx-w_proc/2, ys["sense"]))

plt.tight_layout()
for ext in ["png", "pdf", "svg"]:
    plt.savefig(f"figures/control_flowchart.{ext}", bbox_inches="tight")
