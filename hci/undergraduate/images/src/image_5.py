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

fig5, ax5 = plt.subplots(figsize=(15, 7))
ax5.set_xlim(0, 100)
ax5.set_ylim(0, 100)
ax5.axis("off")
ax5.set_title("系统四层模块化架构")
# 四层底色
layer_y = [76, 51, 26, 1]
layer_h = [22, 23, 23, 23]
layer_color = ["#fff6cc", "#e0ecf8", "#e8f4e0", "#f8e6e6"]
layer_name = ["视觉引导层", "控制算法层", "仿真建模层", "数据采集与评估层"]
for i in range(4):
    y = layer_y[i]
    h = layer_h[i]
    ax5.add_patch(mpatches.Rectangle((2, y), 96, h, ec="black", fc=layer_color[i], lw=1.2))
    ax5.text(6, y+h/2, layer_name[i], fontsize=13, weight="bold", va="center")
# 每层内子框
vis_box = [(18,82,22,12,"目标轨迹生成\n阶跃 / 正弦"), (44,82,22,12,"实时引导界面\n目标 / 当前 / 误差显示"), (70,82,22,12,"误差可视化\n实时反馈条 + 数据曲线")]
ctrl_box = [(14,57,16,12,r"EMA 低通滤波\n$\alpha$ 参数可调"), (34,57,14,12,r"死区过滤\n$u_{th}$ 阈值"), (52,57,18,12,r"非线性映射 + 增益\n$K_p \cdot g(\cdot)$"), (74,57,16,12,r"饱和限幅\n$\pm\theta_{max}$")]
sim_box = [(10,32,18,12,"上肢肌肉骨骼模型\nmobl_arms_bimanual"), (32,32,16,12,"方向盘物理模型\nXML / MJCF 描述"), (52,32,20,12,"耦合驱动约束\n关节-方向盘刚性绑定"), (76,32,18,12,r"MuJoCo 求解器\n前向动力学 + 接触")]
data_box = [(18,7,22,12,"实验数据记录\n时间戳 / 角度 / 误差"), (44,7,22,12,r"精度指标计算\nMAE / RMSE / $e_{max}$"), (70,7,22,12,"数据保存与绘图\nCSV + Matplotlib")]
all_box = vis_box + ctrl_box + sim_box + data_box
for x,y,w,h,txt in all_box:
    ax5.add_patch(mpatches.Rectangle((x,y),w,h,ec="black",fc="white"))
    ax5.text(x+w/2, y+h/2, txt, ha="center", va="center")
# 左右箭头
ax5.annotate("", xy=(0,87), xytext=(0,12), arrowprops=dict(arrowstyle="->", lw=1.6))
ax5.text(-2, 50, "状态反馈", rotation=90, va="center", ha="center")
ax5.annotate("", xy=(100,12), xytext=(100,87), arrowprops=dict(arrowstyle="->", lw=1.6))
ax5.text(102, 50, "控制信号", rotation=270, va="center", ha="center")
save_fig(fig5, "图5_四层模块化架构")
plt.show()