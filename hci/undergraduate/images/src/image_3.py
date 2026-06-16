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

fig3, ax3 = plt.subplots(figsize=(14, 6.5))
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 100)
ax3.axis("off")
# 顶层仿真器
rect_top = mpatches.Rectangle((40, 85), 20, 10, ec="#335599", fc="#e0ecff")
ax3.add_patch(rect_top)
ax3.text(50, 90, "仿真器", ha="center")
# 三大基类
base_x = [15, 50, 85]
base_txt = ["生物力学模型基类", "交互任务基类", "感知模块基类"]
for x, txt in zip(base_x, base_txt):
    ax3.add_patch(mpatches.Rectangle((x-12, 70), 24, 8, ec="#998844", fc="#fff8cc"))
    ax3.text(x, 74, txt, ha="center")
    ax3.plot([50, x], [85,78], c="black")
# 内置实现子类
impl_data = [
    (15, 60, "食指指点上肢模型"),
    (50, 60, "指点任务"),
    (50, 48, "跟踪任务"),
    (85, 60, "视觉感知—固定相机"),
    (85, 48, "本体感觉—含末端位置")
]
for x,y,txt in impl_data:
    ax3.add_patch(mpatches.Rectangle((x-11, y-3), 22, 6, ec="#bb9933", fc="#fff2b8"))
    ax3.text(x, y, txt, ha="center")
# 自定义派生类
derive_data = [
    (15, 48, "用户自定义派生类"),
    (50, 36, "用户自定义派生类"),
    (85, 36, "用户自定义感知通道")
]
for x,y,txt in derive_data:
    ax3.add_patch(mpatches.Rectangle((x-11, y-3), 22, 6, ec="#559955", fc="#d9f2d9"))
    ax3.text(x, y, txt, ha="center")
# 连线
ax3.plot([15,15], [70,63], c="black")
ax3.plot([15,15], [60,51], c="black")
ax3.plot([50,50], [70,63], c="black")
ax3.plot([50,50], [60,51], c="black")
ax3.plot([50,50], [48,39], c="black")
ax3.plot([85,85], [70,63], c="black")
ax3.plot([85,85], [60,51], c="black")
ax3.plot([85,85], [48,39], c="black")
# 底部图例说明
ax3.add_patch(mpatches.Rectangle((5, 8), 90, 14, ec="#999", fc="#f8f8f8"))
leg_txt = ["■ 仿真器顶层", "■ 三大子模块基类", "■ 框架内置实现", "■ 用户自定义派生"]
leg_x = [12, 30, 52, 76]
for i, x in enumerate(leg_x):
    c = ["#335599","#fff8cc","#fff2b8","#d9f2d9"][i]
    ax3.add_patch(mpatches.Rectangle((x-3, 16), 2, 2, ec="black", fc=c))
    ax3.text(x+3, 17, leg_txt[i], va="center")
ax3.text(50, 11, "说明: 每个子模块基类向下派生出具体实现; 本研究主要在生物力学模型与交互任务两个维度上做适配，感知模块沿用框架默认实现。", ha="center")
save_fig(fig3, "图3_框架类层次图")
plt.show()