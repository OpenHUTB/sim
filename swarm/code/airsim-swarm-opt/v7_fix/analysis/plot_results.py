"""
plot_results.py

画实验结果图
2024/11

用法:
    python analysis/plot_results.py --input results/data/swarm_summary.csv
    python analysis/plot_results.py --demo   # 用预设数据看效果

输出到 results/figures/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STYLES = {
    "M1": dict(color="#1976D2", marker="o", label="M1 单线程分发器"),
    "M2": dict(color="#388E3C", marker="s", label="M2 标准阻塞睡眠"),
    "M3": dict(color="#F57C00", marker="^", label="M3 yield自旋"),
    "M4": dict(color="#D32F2F", marker="D", label="M4 sleep(0)自旋 [原始]"),
    "M5": dict(color="#7B1FA2", marker="v", label="M5 纯自旋"),
}


def make_demo_data() -> pd.DataFrame:
    """无真实数据时用理论推算值"""
    rng = np.random.default_rng(42)
    ns = [1, 5, 10, 15, 20, 25, 30]
    fps_fn = {
        "M1": lambda n: max(5.0, 58 - max(0, n-8)*0.5  + rng.normal(0, 1.0)),
        "M2": lambda n: max(5.0, 54 - max(0, n-8)*0.9  + rng.normal(0, 1.5)),
        "M3": lambda n: max(3.0, 49 - max(0, n-6)*1.6  + rng.normal(0, 2.0)),
        "M4": lambda n: max(2.0, 43 - max(0, n-5)*2.3  + rng.normal(0, 2.0)),
        "M5": lambda n: max(1.0, 38 - max(0, n-4)*3.1  + rng.normal(0, 2.5)),
    }
    lat_fn = {
        "M1": lambda n: max(2.0, 3.0  + n*0.08 + rng.normal(0, 0.4)),
        "M2": lambda n: max(2.0, 4.0  + n*0.13 + rng.normal(0, 0.7)),
        "M3": lambda n: max(2.0, 5.2  + n*0.19 + rng.normal(0, 1.0)),
        "M4": lambda n: max(2.0, 6.5  + n*0.26 + rng.normal(0, 1.2)),
        "M5": lambda n: max(2.0, 8.5  + n*0.36 + rng.normal(0, 1.5)),
    }
    ker_fn = {
        "M1": lambda n: max(4.0,  8  + max(0, n-8)*0.28 + rng.normal(0, 1.0)),
        "M2": lambda n: max(4.0, 13  + max(0, n-8)*0.55 + rng.normal(0, 1.2)),
        "M3": lambda n: max(4.0, 26  + max(0, n-6)*1.25 + rng.normal(0, 2.0)),
        "M4": lambda n: max(4.0, 31  + max(0, n-5)*1.85 + rng.normal(0, 2.0)),
        "M5": lambda n: max(4.0, 42  + max(0, n-4)*2.6  + rng.normal(0, 3.0)),
    }
    rows = []
    for m in STYLES:
        for n in ns:
            rows.append({
                "model": m, "n": n,
                "fps_mean":    fps_fn[m](n),
                "lat_mean":    lat_fn[m](n),
                "kernel_pct":  ker_fn[m](n),
            })
    return pd.DataFrame(rows)


def fig_fps(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m, st in STYLES.items():
        d = df[df.model == m].sort_values("n")
        if d.empty:
            continue
        ax.plot(d.n, d.fps_mean, color=st["color"], marker=st["marker"],
                label=st["label"], lw=2, ms=7)
    ax.axvline(16, color="gray", ls="--", alpha=0.5, label="物理核心数(16)")
    ax.set_xlabel("无人机数量", fontsize=12)
    ax.set_ylabel("仿真帧率 (FPS)", fontsize=12)
    ax.set_title("不同线程模型下仿真帧率随无人机数量的变化", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 65)
    p = os.path.join(out_dir, "fig_fps.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p}")


def fig_latency(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m, st in STYLES.items():
        d = df[df.model == m].sort_values("n")
        if d.empty:
            continue
        ax.plot(d.n, d.lat_mean, color=st["color"], marker=st["marker"],
                label=st["label"], lw=2, ms=7)
    ax.set_xlabel("无人机数量", fontsize=12)
    ax.set_ylabel("指令延迟 (ms)", fontsize=12)
    ax.set_title("不同线程模型下指令延迟随无人机数量的变化", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "fig_latency.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p}")


def fig_kernel(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m, st in STYLES.items():
        d = df[df.model == m].sort_values("n")
        if d.empty:
            continue
        ax.plot(d.n, d.kernel_pct, color=st["color"], marker=st["marker"],
                label=st["label"], lw=2, ms=7)
    ax.axhline(50, color="red", ls="--", alpha=0.5, label="警戒线 50%")
    ax.set_xlabel("无人机数量", fontsize=12)
    ax.set_ylabel("CPU 内核态占比 (%)", fontsize=12)
    ax.set_title("不同线程模型下 CPU 内核态时间占比", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    p = os.path.join(out_dir, "fig_kernel_cpu.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p}")


def fig_improve(df, out_dir):
    """M1(优化) vs M4(原始) 提升幅度柱状图"""
    ns = [10, 20, 30]
    fps_imp, lat_red = [], []
    for n in ns:
        m1 = df[(df.model == "M1") & (df.n == n)]
        m4 = df[(df.model == "M4") & (df.n == n)]
        if m1.empty or m4.empty:
            fps_imp.append(0); lat_red.append(0)
        else:
            fps_imp.append((m1.fps_mean.values[0] - m4.fps_mean.values[0])
                           / m4.fps_mean.values[0] * 100)
            lat_red.append((m4.lat_mean.values[0] - m1.lat_mean.values[0])
                           / m4.lat_mean.values[0] * 100)

    x = np.arange(len(ns))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, fps_imp, w, label="FPS 提升 %",  color="#1976D2", alpha=0.85)
    b2 = ax.bar(x + w/2, lat_red, w, label="延迟降低 %", color="#388E3C", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}架" for n in ns])
    ax.set_ylabel("提升幅度 (%)", fontsize=12)
    ax.set_title("M1(单线程分发器) 对比 M4(原始) 的性能提升", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    for b in [*b1, *b2]:
        ax.annotate(f"{b.get_height():.0f}%",
                    xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)
    p = os.path.join(out_dir, "fig_improvement.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=str, default="results/data/swarm_summary.csv")
    ap.add_argument("--output", type=str, default="results/figures")
    ap.add_argument("--demo",   action="store_true", help="使用预设数据")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.demo or not os.path.exists(args.input):
        print("[info] 使用预设数据")
        df = make_demo_data()
        demo_csv = os.path.join(os.path.dirname(args.input), "demo_data.csv")
        os.makedirs(os.path.dirname(os.path.abspath(demo_csv)), exist_ok=True)
        df.to_csv(demo_csv, index=False)
    else:
        df = pd.read_csv(args.input)
        rename = {"cpu_kernel_pct": "kernel_pct", "num_drones": "n", "n_drones": "n"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    fig_fps(df, args.output)
    fig_latency(df, args.output)
    fig_kernel(df, args.output)
    fig_improve(df, args.output)
    print(f"\n全部图表已生成至: {args.output}")


if __name__ == "__main__":
    main()
