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

fig1, ax1 = plt.subplots(figsize=(16, 7))
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 30)
ax1.axis("off")
# 顶部总标题框
rect_top = mpatches.Rectangle((30, 26), 40, 3, ec="black", fc="white", lw=1.2)
ax1.add_patch(rect_top)
ax1.text(50, 27.5, "基于肌肉驱动的生物力学人机交互系统", ha="center", va="center", fontsize=16)
# 7个章节顶层框
chap_x = [10, 22, 36, 49, 62, 75, 88]
chap_text = [
    "第1章\n结论",
    "第2章\n理论基础\n与关键技术",
    "第3章\n系统总体设计",
    "第4章\n建模与实现",
    "第5章\n视觉引导\n与控制算法",
    "第6章\n测试与精度分析",
    "第7章\n总结与展望"
]
for x, txt in zip(chap_x, chap_text):
    ax1.add_patch(mpatches.Rectangle((x-5, 20), 10, 4, ec="black", fc="white"))
    ax1.text(x, 22, txt, ha="center", va="center")
# 各章节子条目
sub_data = [
    [(10, 16, "研究背景与意义"), (10, 13, "国内外研究现状"), (10, 10, "研究内容与组织")],
    [(22, 16, "上肢生物力学理论"), (22, 13, "MuJoCo 仿真引擎"), (22, 10, "User-in-the-Box 框架"), (22, 7, "视觉引导与控制基础"), (22, 4, "系统评估指标体系")],
    [(36, 16, "系统需求分析"), (36, 13, "四层架构设计"), (36, 10, "系统工作流程"), (36, 7, "开发环境与工具")],
    [(49, 16, "建模总体设计"), (49, 13, "上肢肌肉骨骼模型"), (49, 10, "方向盘物理模型"), (49, 7, "肌肉-方向盘耦合"), (49, 4, "模型合理性验证")],
    [(62, 16, "算法整体架构"), (62, 13, "数据处理与运动学"), (62, 10, "控制模块设计实现"), (62, 7, "视觉引导模块"), (62, 4, "系统集成与运行")],
    [(75, 16, "测试目的与方案"), (75, 13, "测试环境与参数"), (75, 10, "功能测试"), (75, 7, "精度与稳定性测试")],
    [(88, 16, "研究总结"), (88, 13, "不足与展望")]
]
for sublist in sub_data:
    for x, y, txt in sublist:
        ax1.add_patch(mpatches.Rectangle((x-4.5, y-1.2), 9, 2.2, ec="black", fc="white"))
        ax1.text(x, y, txt, ha="center", va="center")
save_fig(fig1, "图1_章节组织结构")
plt.show()