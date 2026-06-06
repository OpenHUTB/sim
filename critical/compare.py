# experiments/compare.py
# 算法对比：DQN vs Attention-DQN、PPO vs Smooth-PPO
# 生成柱状图、折线图、雷达图、对比表格——供毕设可视化展示

import os
import sys
import argparse
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.data_saver import load_training_log, save_metrics, ensure_dir
from utils.metrics import MovingAverage


# ================================================================
# 数据加载
# ================================================================

def load_training_data(result_dirs):
    """
    加载多组训练日志。

    result_dirs: dict {label: csv_path}
    返回: dict {label: list[dict]}
    """
    data = {}
    for label, path in result_dirs.items():
        if os.path.exists(path):
            data[label] = load_training_log(path)
        else:
            print("警告: 日志文件不存在 %s" % path)
            data[label] = []
    return data


def load_evaluation_data(eval_dirs):
    """
    加载多组评估结果 JSON。

    返回: dict {label: dict}
    """
    data = {}
    for label, path in eval_dirs.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data[label] = json.load(f)
        else:
            print("警告: 评估文件不存在 %s" % path)
    return data


# ================================================================
# 指标计算
# ================================================================

def compute_comparison_table(eval_data):
    """
    从评估数据生成对比表格。

    返回: list[dict]，每行包含所有指标
    """
    rows = []
    for label, ed in eval_data.items():
        s = ed.get("summary", {})
        rows.append({
            "label": label,
            "algorithm": ed.get("algorithm", label),
            "scenario": ed.get("scenario", ""),
            "collision_rate": s.get("collision_rate", 0),
            "success_rate": s.get("success_rate", 0),
            "mean_reward": s.get("mean_total_reward", 0),
            "std_reward": s.get("std_total_reward", 0),
            "mean_min_ttc": s.get("mean_min_ttc", float("inf")),
            "mean_min_distance": s.get("mean_min_distance", float("inf")),
            "mean_max_danger": s.get("mean_max_danger_level", 0),
            "danger_dist": s.get("danger_level_distribution", {}),
        })
    return rows


def compare(algo_pair, scenario_name, log_paths, eval_paths,
            output_dir="results/comparison"):
    """
    对比一对算法在单个场景上的表现。

    参数:
        algo_pair:     ("DQN", "Attention-DQN") 或 ("PPO", "Smooth-PPO")
        scenario_name: 场景标识
        log_paths:     dict {algo_name: training_log_csv_path}
        eval_paths:    dict {algo_name: eval_json_path}
        output_dir:    图表/表格保存目录

    返回:
        dict: 对比结果
    """
    name_a, name_b = algo_pair
    ensure_dir(output_dir)

    print("=" * 65)
    print("  算法对比: %s vs %s  (%s)" % (name_a, name_b, scenario_name))
    print("=" * 65)

    # 加载训练日志
    train_data = load_training_data(log_paths)

    # 加载评估结果
    eval_data = load_evaluation_data(eval_paths)

    # 训练曲线统计
    train_summary = {}
    for label, logs in train_data.items():
        if not logs:
            train_summary[label] = {"episodes": 0}
            continue
        rewards = [float(r["reward"]) for r in logs if "reward" in r]
        ma = MovingAverage(window_size=100)
        smoothed = [ma.update(r) for r in rewards]
        train_summary[label] = {
            "episodes": len(logs),
            "final_reward": rewards[-1] if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "final_avg100": smoothed[-1] if smoothed else 0,
            "max_avg100": max(smoothed) if smoothed else 0,
        }

    # 对比表格
    table = compute_comparison_table(eval_data)

    # ---- 输出 ----
    print("\n--- 训练曲线统计 ---")
    for label, ts in train_summary.items():
        print("  %s: episodes=%d final_reward=%.1f "
              "max_avg100=%.1f"
              % (label, ts["episodes"], ts["final_reward"], ts["max_avg100"]))

    print("\n--- 评估指标对比 ---")
    header = "%-20s %10s %10s %10s %10s %10s"
    print(header % ("指标", name_a, name_b, "差值", "改善率", ""))
    print("-" * 65)

    metrics_to_compare = [
        ("collision_rate", "碰撞率", True),     # 越低越好
        ("success_rate", "安全完成率", False),   # 越高越好
        ("mean_reward", "平均奖励", False),
        ("mean_min_ttc", "平均最小TTC", False),
        ("mean_min_distance", "平均最小距离", False),
        ("mean_max_danger", "平均危险等级", True),
    ]

    comparison_results = {}
    for key, label, lower_better in metrics_to_compare:
        val_a = table[0].get(key, 0) if table else 0
        val_b = table[1].get(key, 0) if len(table) > 1 else 0
        diff = val_b - val_a
        if abs(val_a) > 1e-9:
            improvement = diff / abs(val_a) * 100
        else:
            improvement = 0.0

        direction = "↓更好" if lower_better else "↑更好"
        comparison_results[key] = {
            "label": label,
            name_a: round(val_a, 4),
            name_b: round(val_b, 4),
            "diff": round(diff, 4),
            "improvement_pct": round(improvement, 2),
            "lower_better": lower_better,
        }
        row_fmt = "%-20s %10.4f %10.4f %+10.4f %+9.1f%% %s"
        print(row_fmt % (label, val_a, val_b, diff, improvement, direction))

    # 危险等级分布对比
    print("\n--- 危险等级分布 ---")
    if table:
        for i, row in enumerate(table):
            dist = row.get("danger_dist", {})
            print("  %s: L0(safe)=%.1f%% L1(low)=%.1f%% "
                  "L2(medium)=%.1f%% L3(high)=%.1f%%"
                  % (row["label"],
                     dist.get("0", 0) * 100,
                     dist.get("1", 0) * 100,
                     dist.get("2", 0) * 100,
                     dist.get("3", 0) * 100))

    # 创新效果总结
    print("\n--- 创新效果总结 ---")
    improvements = []
    for key, cr in comparison_results.items():
        imp = cr["improvement_pct"]
        if (cr["lower_better"] and imp < 0) or (not cr["lower_better"] and imp > 0):
            improvements.append((cr["label"], abs(imp), "改善"))
        elif abs(imp) < 0.5:
            pass  # 基本持平
        else:
            improvements.append((cr["label"], abs(imp), "退化"))
    if improvements:
        for label, imp, direction in sorted(improvements, key=lambda x: -x[1]):
            print("  %s: %s %.1f%%" % (label, direction, imp))
    else:
        print("  两项算法表现基本持平")

    # 保存
    result = {
        "algo_pair": list(algo_pair),
        "scenario": scenario_name,
        "training_summary": train_summary,
        "metrics_comparison": comparison_results,
        "eval_table": table,
    }
    result_path = os.path.join(output_dir,
                               "%s_vs_%s_%s.json" % (name_a, name_b, scenario_name))
    save_metrics(result_path, result)
    print("\n对比结果已保存至: %s" % result_path)

    # 尝试生成 matplotlib 图表
    try:
        _plot_comparison(train_data, eval_data, algo_pair, scenario_name, output_dir)
    except ImportError:
        print("matplotlib 未安装，跳过图表生成")

    return result


def _plot_comparison(train_data, eval_data, algo_pair, scenario_name, output_dir):
    """生成对比图表（需 matplotlib）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    name_a, name_b = algo_pair

    # 左图：训练奖励曲线
    ax = axes[0]
    for label, logs in train_data.items():
        if not logs:
            continue
        rewards = [float(r["reward"]) for r in logs if "reward" in r]
        ma = MovingAverage(window_size=100)
        smoothed = [ma.update(r) for r in rewards]
        ax.plot(smoothed, label=label, linewidth=1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward (window=100)")
    ax.set_title("Training Curves: %s vs %s" % (name_a, name_b))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：评估指标柱状图
    ax = axes[1]
    if eval_data:
        labels = list(eval_data.keys())
        metrics_names = ["碰撞率", "安全完成率", "平均奖励(÷10)", "平均TTC(s)", "危险等级"]
        x = np.arange(len(metrics_names))
        width = 0.35

        for i, (lbl, ed) in enumerate(eval_data.items()):
            s = ed.get("summary", {})
            values = [
                s.get("collision_rate", 0) * 100,
                s.get("success_rate", 0) * 100,
                s.get("mean_total_reward", 0) / 10,
                min(s.get("mean_min_ttc", 5), 10),
                s.get("mean_max_danger_level", 0),
            ]
            bars = ax.bar(x + i * width - width / 2, values, width, label=lbl, alpha=0.85)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        "%.1f" % val, ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=9)
        ax.set_title("Evaluation Metrics: %s vs %s (%s)"
                     % (name_a, name_b, scenario_name))
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(output_dir,
                             "%s_vs_%s_%s.png" % (name_a, name_b, scenario_name))
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print("图表已保存至: %s" % plot_path)


# ================================================================
# 入口
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="算法对比")
    parser.add_argument("--pair", type=str, default="dqn",
                        choices=["dqn", "ppo"],
                        help="对比对: dqn=DQN vs AttnDQN, ppo=PPO vs SmoothPPO")
    parser.add_argument("--scenario", type=str, default="rain_storm",
                        help="场景名称")
    parser.add_argument("--log_a", type=str, default="",
                        help="算法A训练日志CSV路径")
    parser.add_argument("--log_b", type=str, default="",
                        help="算法B训练日志CSV路径")
    parser.add_argument("--eval_a", type=str, default="",
                        help="算法A评估JSON路径")
    parser.add_argument("--eval_b", type=str, default="",
                        help="算法B评估JSON路径")
    parser.add_argument("--output_dir", type=str, default="results/comparison")
    args = parser.parse_args()

    # 确定算法对
    if args.pair == "dqn":
        algo_pair = ("DQN", "Attention-DQN")
        default_log_a = "results/dqn/dqn_%s_*/logs/training_log.csv" % args.scenario
        default_log_b = "results/attention_dqn/attn_dqn_%s_*/logs/training_log.csv" % args.scenario
    else:
        algo_pair = ("PPO", "Smooth-PPO")
        default_log_a = "results/ppo/ppo_%s_*/logs/training_log.csv" % args.scenario
        default_log_b = "results/smooth_ppo/smooth_ppo_%s_*/logs/training_log.csv" % args.scenario

    # 构建路径
    log_paths = {
        algo_pair[0]: args.log_a or _find_latest(default_log_a),
        algo_pair[1]: args.log_b or _find_latest(default_log_b),
    }
    eval_paths = {
        algo_pair[0]: args.eval_a or "results/evaluation/%s_%s.json"
                      % (algo_pair[0].lower().replace("-", "_"), args.scenario),
        algo_pair[1]: args.eval_b or "results/evaluation/%s_%s.json"
                      % (algo_pair[1].lower().replace("-", "_"), args.scenario),
    }

    compare(algo_pair, args.scenario, log_paths, eval_paths, args.output_dir)


def _find_latest(pattern):
    """在模式中包含通配符时查找最新文件"""
    import glob
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else pattern
