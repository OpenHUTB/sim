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

fig8, ax8 = plt.subplots(figsize=(14, 7))
ax8.set_xlim(0, 100)
ax8.set_ylim(0, 100)
ax8.axis("off")
ax8.set_title("测试方案与指标体系")
# 顶层总框
ax8.add_patch(mpatches.Rectangle((40, 86), 20, 10, ec="black", fc="white"))
ax8.text(50, 91, "系统综合测试", ha="center")
# 三大测试分类
cat_x = [20, 50, 80]
cat_txt = ["功能测试", "精度测试", "稳定性 / 易用性测试"]
for x, txt in zip(cat_x, cat_txt):
    ax8.add_patch(mpatches.Rectangle((x-12, 72), 24, 8, ec="black", fc="white"))
    ax8.text(x, 76, txt, ha="center")
    ax8.plot([50, x], [86,80], c="black")
# 功能测试子项
func_sub = [
    (10,62,"遥控驾驶任务"),(22,62,"仿真环境搭建"),
    (10,54,"控制精度任务"),(22,54,"模型加载验证"),
    (10,46,"小物体触碰任务"),(22,46,"视觉引导显示")
]
# 精度测试子项
prec_sub = [
    (40,62,"阶跃跟踪"),(52,62,"不同幅值组合"),
    (40,54,"正弦跟踪"),(52,54,"不同频率组合"),
    (40,46,"定点保持"),(52,46,"重复试验取均值")
]
# 稳定性子项
stab_sub = [
    (70,62,"连续 1 h 运行"),(82,62,"非专业用户上手"),
    (70,54,"角度漂移监测"),(82,54,"学习曲线观察"),
    (70,46,"仿真帧率监测"),(82,46,"主观评分")
]
all_sub = func_sub + prec_sub + stab_sub
for x,y,txt in all_sub:
    ax8.add_patch(mpatches.Rectangle((x-8,y-2.5),16,5,ec="black",fc="white"))
    ax8.text(x,y,txt,ha="center")
# 连线
ax8.plot([20,20],[72,68],c="black")
ax8.plot([20,20],[62,58],c="black")
ax8.plot([20,20],[54,50],c="black")
ax8.plot([50,50],[72,68],c="black")
ax8.plot([50,50],[62,58],c="black")
ax8.plot([50,50],[54,50],c="black")
ax8.plot([80,80],[72,68],c="black")
ax8.plot([80,80],[62,58],c="black")
ax8.plot([80,80],[54,50],c="black")
# 量化指标分割线
ax8.plot([5,95], [38,38], c="black", lw=1.5)
ax8.text(50, 41, "↓ 量化指标输出 ↓", ha="center", fontsize=13)
# 指标框
metric_data = [
    (12,22,r"MAE\n平均绝对误差"),
    (30,22,r"RMSE\n均方根误差"),
    (48,22,r"$e_{max}$\n最大误差"),
    (66,22,r"响应延迟 $t_d$\n互相关 argmax"),
    (84,22,"连续运行稳定性\n漂移量 / 帧率")
]
for x,y,txt in metric_data:
    ax8.add_patch(mpatches.Rectangle((x-8,y-4),16,8,ec="black",fc="white"))
    ax8.text(x,y,txt,ha="center")
# 合格基线框
rect_base = mpatches.Rectangle((10, 5), 80, 14, ec="#dd6622", fc="#fff2e0", lw=1.5)
ax8.add_patch(rect_base)
ax8.text(50, 15, "指标合格基线", ha="center", fontsize=14, weight="bold", c="#bb3300")
ax8.text(50, 10, r"MAE $\leq 3.0^\circ \cdot$ RMSE $\leq 3.5^\circ \cdot$ 响应延迟 $\leq 10\ \mathrm{ms}$", ha="center")
ax8.text(50, 6, "连续 60 min 无崩溃 / 无明显漂移 / 无异常抖动", ha="center")
save_fig(fig8, "图8_测试指标体系")
plt.show()