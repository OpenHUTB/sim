"""Compare baseline and optimized reward objectives on real T2M motions.

This is an offline reward-ablation experiment. It does not simulate a policy and
does not generate fake gait or muscle data. It scores the real pre-generated T2M
motions with:

1. the reference baseline objective: body/velocity/upright/energy;
2. the optimized objective: baseline terms plus joint-angle, muscle-drive, and
   action-smoothness regularizers.

The runtime environment now computes the true muscle-drive and smoothness terms
from MuJoCo controls/actions. Offline T2M motions do not contain MuJoCo muscle
activations, so this script uses deterministic kinematic proxies for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_DIR = PROJECT_DIR / "data" / "t2m"
OUTPUT_DIR = PROJECT_DIR / "output" / "reward_optimization"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


BASELINE_WEIGHTS = {
    "tracking": 0.6,
    "velocity": 0.2,
    "upright": 0.1,
    "energy": 0.1,
}

OPTIMIZED_WEIGHTS = {
    "tracking": 0.55,
    "velocity": 0.2,
    "upright": 0.08,
    "energy": 0.07,
    "joint_angle": 0.05,
    "muscle_drive": 0.025,
    "smoothness": 0.025,
}


def normalize_pose_aa(pose_aa: np.ndarray) -> np.ndarray:
    pose_aa = np.asarray(pose_aa, dtype=np.float64)
    if pose_aa.ndim == 2 and pose_aa.shape[1] == 156:
        pose_aa = np.concatenate(
            [pose_aa[:, :66], np.zeros((pose_aa.shape[0], 6), dtype=np.float64)],
            axis=1,
        )
    if pose_aa.ndim == 2 and pose_aa.shape[1] == 72:
        pose_aa = pose_aa.reshape(-1, 24, 3)
    if pose_aa.ndim != 3 or pose_aa.shape[1:] != (24, 3):
        raise ValueError(f"不支持的 pose_aa 形状: {pose_aa.shape}")
    return pose_aa


def load_motion(path: Path) -> tuple[str, dict[str, Any]]:
    motion_dict = joblib.load(path)
    if not isinstance(motion_dict, dict) or not motion_dict:
        raise ValueError(f"动作文件不是非空 dict: {path}")
    name, motion = next(iter(motion_dict.items()))
    return str(name), motion


def motion_group(path: Path) -> str:
    match = re.match(r"mdm_(.+)_\d+\.pkl", path.name)
    return match.group(1) if match else path.stem


def exp_score(cost: float, scale: float) -> float:
    return float(np.exp(-scale * max(float(cost), 0.0)))


def score_motion(path: Path) -> dict[str, Any]:
    _, motion = load_motion(path)
    pose = normalize_pose_aa(motion["pose_aa"])
    fps = float(motion.get("fps", 30))
    dt = 1.0 / max(fps, 1.0)

    joint_pose = pose[:, 1:, :]
    joint_speed = np.diff(joint_pose, axis=0) / dt
    joint_acc = np.diff(joint_speed, axis=0) / dt if len(joint_speed) > 1 else np.zeros_like(joint_speed)

    trans = motion.get("root_trans_offset", motion.get("trans", motion.get("trans_orig")))
    trans = np.asarray(trans, dtype=np.float64)
    root_speed = np.diff(trans, axis=0) / dt if trans.ndim == 2 and trans.shape[1] == 3 else np.zeros((len(pose) - 1, 3))

    root_tilt_cost = np.mean(pose[:, 0, :2] ** 2)
    joint_angle_cost = np.mean(np.linalg.norm(joint_pose, axis=-1) ** 2)
    muscle_drive_proxy_cost = np.mean(joint_speed**2) if joint_speed.size else 0.0
    energy_cost = np.mean(joint_speed**2) + 0.05 * np.mean(root_speed**2) if joint_speed.size else 0.0
    smoothness_cost = np.mean(joint_acc**2) if joint_acc.size else 0.0

    tracking_score = 1.0
    velocity_score = 1.0
    upright_score = exp_score(root_tilt_cost, 3.0)
    energy_score = exp_score(energy_cost, 0.0008)
    joint_angle_score = exp_score(joint_angle_cost, 0.35)
    muscle_drive_score = exp_score(muscle_drive_proxy_cost, 0.0008)
    smoothness_score = exp_score(smoothness_cost, 0.00002)

    baseline_score = (
        BASELINE_WEIGHTS["tracking"] * tracking_score
        + BASELINE_WEIGHTS["velocity"] * velocity_score
        + BASELINE_WEIGHTS["upright"] * upright_score
        + BASELINE_WEIGHTS["energy"] * energy_score
    )
    optimized_score = (
        OPTIMIZED_WEIGHTS["tracking"] * tracking_score
        + OPTIMIZED_WEIGHTS["velocity"] * velocity_score
        + OPTIMIZED_WEIGHTS["upright"] * upright_score
        + OPTIMIZED_WEIGHTS["energy"] * energy_score
        + OPTIMIZED_WEIGHTS["joint_angle"] * joint_angle_score
        + OPTIMIZED_WEIGHTS["muscle_drive"] * muscle_drive_score
        + OPTIMIZED_WEIGHTS["smoothness"] * smoothness_score
    )

    return {
        "motion_file": path.name,
        "group": motion_group(path),
        "frames": int(pose.shape[0]),
        "baseline_score": float(baseline_score),
        "optimized_score": float(optimized_score),
        "tracking_score": tracking_score,
        "velocity_score": velocity_score,
        "upright_score": upright_score,
        "energy_score": energy_score,
        "joint_angle_score": joint_angle_score,
        "muscle_drive_proxy_score": muscle_drive_score,
        "smoothness_score": smoothness_score,
        "joint_angle_cost": float(joint_angle_cost),
        "muscle_drive_proxy_cost": float(muscle_drive_proxy_cost),
        "energy_cost": float(energy_cost),
        "smoothness_cost": float(smoothness_cost),
    }


def group_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({row["group"] for row in rows})
    result = []
    numeric_keys = [
        "baseline_score",
        "optimized_score",
        "upright_score",
        "energy_score",
        "joint_angle_score",
        "muscle_drive_proxy_score",
        "smoothness_score",
    ]
    for group in groups:
        subset = [row for row in rows if row["group"] == group]
        item = {"group": group, "num_motions": len(subset)}
        for key in numeric_keys:
            item[key] = float(np.mean([row[key] for row in subset]))
        result.append(item)
    return result


def plot_comparison(rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> None:
    labels = [row["motion_file"].replace("mdm_", "").replace(".pkl", "") for row in rows]
    x = np.arange(len(rows))
    baseline = np.array([row["baseline_score"] for row in rows], dtype=float)
    optimized = np.array([row["optimized_score"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.28), 5.5))
    ax.plot(x, baseline, marker="o", linewidth=1.4, label="优化前奖励")
    ax.plot(x, optimized, marker="o", linewidth=1.4, label="优化后奖励")
    ax.set_title("真实 T2M 动作：优化前/优化后奖励目标对比", fontsize=14, fontweight="bold")
    ax.set_ylabel("Reward score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_vs_optimized_reward_by_motion.png", dpi=160)
    plt.close(fig)

    group_labels = [row["group"] for row in groups]
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [row["baseline_score"] for row in groups], width, label="优化前")
    ax.bar(x + width / 2, [row["optimized_score"] for row in groups], width, label="优化后")
    ax.set_title("按动作类型汇总的奖励目标对比", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean reward score")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_vs_optimized_reward_by_group.png", dpi=160)
    plt.close(fig)

    component_labels = ["关节角度", "肌肉驱动代理", "能耗", "平滑性"]
    component_cost_keys = [
        "joint_angle_cost",
        "muscle_drive_proxy_cost",
        "energy_cost",
        "smoothness_cost",
    ]
    component_values = []
    for key in component_cost_keys:
        values = np.array([row[key] for row in rows], dtype=float)
        scale = np.nanpercentile(values, 95)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.nanmax(values)
        component_values.append(float(np.nanmean(values / max(scale, 1e-12))))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(component_labels, component_values, color=["#18A058", "#2080F0", "#F0A020", "#D03050"])
    ax.set_title("优化奖励新增生物合理性代价", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized cost (lower is better)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, component_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=10)
    ax.text(
        0.5,
        -0.18,
        "说明：T2M 离线动作不包含 MuJoCo 真实肌肉激活，肌肉驱动为运动学代理代价。",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "optimized_reward_components.png", dpi=160)
    plt.close(fig)


def write_results(rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "reward_optimization_results.json").open("w", encoding="utf-8") as f:
        json.dump({"motions": rows, "groups": groups}, f, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "reward_optimization_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", type=Path, default=DEFAULT_MOTION_DIR)
    parser.add_argument("--max-motions", type=int, default=50)
    args = parser.parse_args()

    motion_files = sorted(args.motion_dir.glob("*.pkl"))[: args.max_motions]
    if not motion_files:
        raise FileNotFoundError(f"没有找到真实 T2M 动作文件: {args.motion_dir}")

    rows = [score_motion(path) for path in motion_files]
    groups = group_means(rows)
    write_results(rows, groups)
    plot_comparison(rows, groups)

    print("奖励函数优化对比实验完成")
    print(f"  动作数量: {len(rows)}")
    print(f"  动作类型: {', '.join(row['group'] for row in groups)}")
    print(f"  优化前平均得分: {np.mean([row['baseline_score'] for row in rows]):.4f}")
    print(f"  优化后平均得分: {np.mean([row['optimized_score'] for row in rows]):.4f}")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
