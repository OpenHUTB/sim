import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

# ========= 基础设置 =========
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

FIG_W, FIG_H = 18, 11
BLUE = "#2f66b3"
LIGHT_BLUE = "#eef5ff"
MID_BLUE = "#dceaff"
BORDER = "#7aa2d8"
DASH = "#9bb8df"
ARROW = "#8c8c8c"
TEXT = "#222222"

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


# ========= 工具函数 =========
def rounded_box(x, y, w, h, fc="white", ec=BORDER, lw=1.4, radius=0.08, linestyle="-", z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius * min(w, h)}",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=linestyle, zorder=z
    )
    ax.add_patch(patch)
    return patch


def text_center(x, y, s, size=14, weight="normal", color=TEXT, z=5):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, weight=weight, color=color, zorder=z)


def draw_arrow(x1, y1, x2, y2, ms=18, lw=1.5, color=ARROW):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='-|>', mutation_scale=ms,
                          linewidth=lw, color=color, zorder=3)
    ax.add_patch(arr)


def draw_number_circle(cx, cy, num, r=1.25):
    circ = Circle((cx, cy), r, facecolor=BLUE, edgecolor=BLUE, linewidth=1.2, zorder=4)
    ax.add_patch(circ)
    ax.text(cx, cy, str(num), ha="center", va="center", fontsize=11, color="white", weight="bold", zorder=5)


def draw_content_box(x, y, w, h, num, title_lines, bullet_lines):
    rounded_box(x, y, w, h, fc="white", ec=BORDER, lw=1.3, radius=0.08, z=2)

    # 编号圆
    draw_number_circle(x + 2.0, y + h - 3.4, num)

    # 标题
    title_y = y + h - 3.8
    ax.text(x + w / 2 + 0.8, title_y,
            "\n".join(title_lines),
            ha="center", va="center", fontsize=12, weight="bold", color=BLUE, zorder=5)

    # 分隔线
    ax.plot([x + 0.8, x + w - 0.8], [y + h - 7.2, y + h - 7.2], color=BORDER, linewidth=1.0, zorder=4)

    # 项目符号
    start_y = y + h - 10.5
    dy = 3.1
    for i, line in enumerate(bullet_lines):
        ax.text(x + 1.6, start_y - i * dy, f"• {line}", ha="left", va="center",
                fontsize=10.6, color=TEXT, zorder=5)


def draw_small_step_box(x, y, w, h, num, label):
    rounded_box(x, y, w, h, fc="white", ec=BORDER, lw=1.2, radius=0.10, z=2)
    draw_number_circle(x + w / 2, y + h - 3.0, num, r=1.3)
    ax.text(x + w / 2, y + 3.6, label, ha="center", va="center",
            fontsize=11.5, color=TEXT, weight="bold", zorder=5)


def draw_section_container(x, y, w, h, title):
    rounded_box(x, y, w, h, fc="none", ec=DASH, lw=1.2, radius=0.05, linestyle="--", z=1)
    title_w, title_h = 12, 4.2
    rounded_box(x + (w - title_w) / 2, y + h - 2.4, title_w, title_h,
                fc=BLUE, ec=BLUE, lw=1.0, radius=0.30, z=3)
    text_center(x + w / 2, y + h - 0.35, title, size=14, weight="bold", color="white", z=5)


# ========= 顶部课题目标 =========
rounded_box(31, 93, 38, 5, fc="white", ec=BLUE, lw=1.5, radius=0.15, z=2)
text_center(50, 95.5, "课题目标：基于高保真模拟器的车辆与飞行器协同控制", size=13, weight="bold")
draw_arrow(50, 92.8, 50, 88.5, ms=20)


# ========= 研究内容 =========
draw_section_container(0.5, 55, 99, 31, "研究内容")

content_boxes = [
    {
        "num": 1,
        "title": ["机场牵引作业流程", "建模与问题抽象"],
        "bullets": ["场景约束分析", "作业阶段划分", "状态变量与安全边界定义"]
    },
    {
        "num": 2,
        "title": ["面向自动对接的", "飞机辅助感知方法"],
        "bullets": ["目标身份确认", "相对位姿感知", "近距安全判断"]
    },
    {
        "num": 3,
        "title": ["多约束条件下的", "路径规划与协同控制"],
        "bullets": ["局部轨迹生成", "误差反馈控制", "四轮转向映射与", "约束处理"]
    },
    {
        "num": 4,
        "title": ["基于高保真模拟器的", "系统集成与验证"],
        "bullets": ["SolidWorks/URDF建模", "ROS-Gazebo联合仿真", "传感器挂载与控制器配置"]
    },
    {
        "num": 5,
        "title": ["牵引车辆结构与", "动力系统的工程化思考"],
        "bullets": ["双桥转向结构", "动力布置分析", "工程实现参考"]
    },
]

start_x = 2
gap = 3
box_w = 17.2
box_h = 22

for i, item in enumerate(content_boxes):
    x = start_x + i * (box_w + gap)
    draw_content_box(x, 57.2, box_w, box_h, item["num"], item["title"], item["bullets"])
    if i < len(content_boxes) - 1:
        draw_arrow(x + box_w + 0.6, 68.2, x + box_w + gap - 0.8, 68.2, ms=16)

draw_arrow(50, 54.8, 50, 49.5, ms=20)


# ========= 技术路线 =========
draw_section_container(0.5, 34, 99, 13.5, "技术路线")

steps = [
    "场景分析", "车辆建模", "感知设计", "路径规划",
    "协同控制", "仿真搭建", "全流程测试", "结果分析"
]

step_w = 10.6
step_h = 7.3
step_gap = 1.1
sx0 = 2.3

for i, label in enumerate(steps):
    x = sx0 + i * (step_w + step_gap)
    draw_small_step_box(x, 36.1, step_w, step_h, i + 1, label)
    if i < len(steps) - 1:
        draw_arrow(x + step_w + 0.2, 39.7, x + step_w + step_gap - 0.2, 39.7, ms=14)

draw_arrow(50, 33.8, 50, 28.5, ms=20)


# ========= 研究成果 =========
# 外框
rounded_box(31, 6, 38, 16, fc="white", ec=BLUE, lw=1.4, radius=0.08, z=2)
# 标题栏
header = Rectangle((31, 18.0), 38, 4.0, facecolor=MID_BLUE, edgecolor=BLUE, linewidth=1.2, zorder=3)
ax.add_patch(header)
text_center(50, 20.0, "研究成果", size=14, weight="bold", color=BLUE)

results = [
    "1. 双桥转向牵引车运动学模型",
    "2. 飞机辅助感知与自动对接控制方法",
    "3. ROS-Gazebo联合仿真平台",
    "4. 全流程测试验证结果",
]

for i, line in enumerate(results):
    ax.text(34, 15.7 - i * 2.9, line, ha="left", va="center", fontsize=11.5, color=TEXT, zorder=5)


# ========= 导出 =========
plt.tight_layout()
fig.savefig("research_framework.svg", bbox_inches="tight")
fig.savefig("research_framework.pdf", bbox_inches="tight")
fig.savefig("research_framework.png", dpi=300, bbox_inches="tight")
plt.show()