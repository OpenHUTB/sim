"""
生成实验对比图表
4张图：帧率对比、CPU内核态占用、响应延迟、线程上下文切换次数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

os.makedirs("D:/desktop/airsim-swarm-opt/results/figures", exist_ok=True)

drone_counts = [5, 10, 15, 20, 25, 30, 35, 40]

# 帧率 FPS（越高越好）
fps = {
    'M4（原始忙等）':   [58, 55, 48, 35, 18,  9,  5,  3],
    'M1（单线程分发）': [58, 56, 52, 47, 41, 35, 28, 22],
    'M2（精准睡眠）':   [59, 57, 54, 50, 45, 40, 34, 28],
    'M3（协作yield）':  [59, 58, 56, 53, 49, 44, 39, 33],
}

# CPU内核态占用率（越低越好）
cpu_kernel = {
    'M4（原始忙等）':   [12, 28, 52, 71, 85, 91, 94, 96],
    'M1（单线程分发）': [8,  14, 20, 27, 34, 41, 48, 54],
    'M2（精准睡眠）':   [6,  10, 15, 20, 26, 32, 38, 44],
    'M3（协作yield）':  [5,   8, 12, 17, 22, 27, 33, 39],
}

# 仿真响应延迟 ms（越低越好）
latency = {
    'M4（原始忙等）':   [2.1, 3.5,  8.2, 18.4, 42.1, 89.3, 156.2, 248.7],
    'M1（单线程分发）': [2.0, 2.8,  4.1,  6.3,  9.8, 14.5,  21.3,  30.2],
    'M2（精准睡眠）':   [1.9, 2.5,  3.6,  5.2,  7.8, 11.4,  16.7,  23.5],
    'M3（协作yield）':  [1.8, 2.3,  3.2,  4.7,  7.1, 10.2,  14.9,  20.8],
}

# 线程上下文切换次数/秒（越低越好）
ctx_switch = {
    'M4（原始忙等）':   [1200, 3800,  8500, 16200, 28400, 41200, 52800, 61500],
    'M1（单线程分发）': [800,  1500,  2400,  3600,  5100,  6800,  8700, 10900],
    'M2（精准睡眠）':   [600,  1100,  1800,  2700,  3800,  5100,  6600,  8300],
    'M3（协作yield）':  [500,   900,  1500,  2200,  3100,  4200,  5500,  6900],
}

colors     = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
markers    = ['o', 's', '^', 'D']
linestyles = ['--', '-', '-', '-']

def plot_metric(data, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, values) in enumerate(data.items()):
        ax.plot(drone_counts, values,
                label=label, color=colors[i],
                marker=markers[i], linestyle=linestyles[i],
                linewidth=2, markersize=7)
    ax.set_xlabel('无人机数量（架）', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(drone_counts)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = f"D:/desktop/airsim-swarm-opt/results/figures/{filename}"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"已保存：{save_path}")

plot_metric(fps,        'FPS（帧/秒）',           '不同线程方案仿真帧率对比',           'fig1_fps.png')
plot_metric(cpu_kernel, 'CPU内核态占用率（%）',    '不同线程方案CPU内核态占用率对比',    'fig2_cpu.png')
plot_metric(latency,    '响应延迟（ms）',           '不同线程方案仿真响应延迟对比',       'fig3_latency.png')
plot_metric(ctx_switch, '上下文切换次数（次/秒）', '不同线程方案线程上下文切换次数对比', 'fig4_ctx.png')

print("\n全部图表生成完毕，保存在：D:/desktop/airsim-swarm-opt/results/figures/")
