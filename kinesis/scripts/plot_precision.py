"""Plot real Kinesis evaluation precision metrics.

This script intentionally does not generate synthetic values. Run a real
imitation evaluation first; AgentIM.eval_policy writes
output/precision/evaluation_metrics.json automatically.

Usage:
    python scripts/plot_precision.py
    python scripts/plot_precision.py --metrics output/precision/evaluation_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = PROJECT_DIR / "output" / "precision" / "evaluation_metrics.json"
OUTPUT_DIR = PROJECT_DIR / "output" / "precision"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _as_float(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    return float(np.nanmean(arr))


def load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"真实评估指标文件不存在: {path}\n"
            "请先运行真实评估，例如:\n"
            "python src/run.py exp_name=kinesis-moe-imitation epoch=-1 run=eval_run "
            "run.motion_file=data/kit_test_motion_dict.pkl "
            "run.initial_pose_file=data/initial_pose/initial_pose_test.pkl "
            "run.headless=True\n"
            "评估完成后会生成 output/precision/evaluation_metrics.json。"
        )

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            rows = payload
        elif "motions" in payload:
            rows = payload["motions"]
        elif "mpjpe_dict" in payload:
            rows = []
            mpjpe = payload["mpjpe_dict"]
            coverage = payload.get("frame_coverage_dict", {})
            success = payload.get("success_dict", {})
            for key, value in mpjpe.items():
                rows.append(
                    {
                        "motion_id": key,
                        "mpjpe_m": value,
                        "frame_coverage": coverage.get(key, np.nan),
                        "success": success.get(key, False),
                    }
                )
        else:
            raise ValueError("JSON 中没有 motions 或 mpjpe_dict 字段，无法绘图。")
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError("只支持 JSON 或 CSV 真实评估指标文件。")

    normalized = []
    for i, row in enumerate(rows):
        if "mpjpe_m" in row:
            mpjpe_m = _as_float(row["mpjpe_m"])
        elif "mpjpe" in row:
            mpjpe_m = _as_float(row["mpjpe"])
        elif "mpjpe_mm" in row:
            mpjpe_m = _as_float(row["mpjpe_mm"]) / 1000.0
        else:
            raise ValueError(f"第 {i} 条记录缺少 mpjpe_m/mpjpe/mpjpe_mm。")

        coverage = row.get("frame_coverage", row.get("coverage", np.nan))
        success = row.get("success", False)
        if isinstance(success, str):
            success = success.strip().lower() in {"1", "true", "yes", "y", "成功"}

        normalized.append(
            {
                "motion_id": str(row.get("motion_id", row.get("id", i))),
                "mpjpe_cm": mpjpe_m * 100.0,
                "frame_coverage": _as_float(coverage),
                "success": bool(success),
            }
        )

    if not normalized:
        raise ValueError("评估指标为空，无法绘图。")
    return normalized


def plot_mpjpe(metrics: list[dict[str, Any]]) -> None:
    labels = [m["motion_id"] for m in metrics]
    values = np.array([m["mpjpe_cm"] for m in metrics], dtype=float)
    order = np.argsort(values)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.45), 5))
    colors = ["#18A058" if metrics[i]["success"] else "#D03050" for i in order]
    ax.bar(np.arange(len(order)), values[order], color=colors)
    ax.axhline(values.mean(), color="#555555", linestyle="--", linewidth=1, label=f"平均 {values.mean():.2f} cm")
    ax.set_title("真实评估 MPJPE（按动作）", fontsize=14, fontweight="bold")
    ax.set_ylabel("MPJPE (cm)")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([labels[i] for i in order], rotation=60, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mpjpe_by_motion.png", dpi=160)
    plt.close(fig)


def plot_coverage(metrics: list[dict[str, Any]]) -> None:
    labels = [m["motion_id"] for m in metrics]
    values = np.array([m["frame_coverage"] for m in metrics], dtype=float) * 100.0
    order = np.argsort(values)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.45), 5))
    ax.bar(np.arange(len(order)), values[order], color="#2080F0")
    ax.axhline(np.nanmean(values), color="#555555", linestyle="--", linewidth=1, label=f"平均 {np.nanmean(values):.1f}%")
    ax.set_ylim(0, 105)
    ax.set_title("真实评估帧覆盖率（按动作）", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frame Coverage (%)")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([labels[i] for i in order], rotation=60, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "frame_coverage.png", dpi=160)
    plt.close(fig)


def plot_summary(metrics: list[dict[str, Any]]) -> None:
    mpjpe = np.array([m["mpjpe_cm"] for m in metrics], dtype=float)
    coverage = np.array([m["frame_coverage"] for m in metrics], dtype=float) * 100.0
    success = np.array([m["success"] for m in metrics], dtype=bool)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_title("KINESIS 真实控制精度评估报告", fontsize=18, fontweight="bold", pad=20)
    rows = [
        ("动作数量", f"{len(metrics)}"),
        ("成功率", f"{success.mean() * 100:.1f}%"),
        ("平均 MPJPE", f"{np.nanmean(mpjpe):.2f} cm"),
        ("中位 MPJPE", f"{np.nanmedian(mpjpe):.2f} cm"),
        ("平均帧覆盖率", f"{np.nanmean(coverage):.1f}%"),
    ]
    y = 0.72
    for label, value in rows:
        ax.text(0.28, y, label, fontsize=14, ha="right")
        ax.text(0.36, y, value, fontsize=14, ha="left", color="#18A058")
        y -= 0.12
    ax.text(
        0.5,
        0.08,
        "数据来源：真实评估输出 evaluation_metrics.json / CSV；本脚本不生成模拟指标。",
        fontsize=11,
        ha="center",
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "precision_summary.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.metrics)
    plot_mpjpe(metrics)
    plot_coverage(metrics)
    plot_summary(metrics)

    mpjpe = np.array([m["mpjpe_cm"] for m in metrics], dtype=float)
    coverage = np.array([m["frame_coverage"] for m in metrics], dtype=float) * 100
    success = np.array([m["success"] for m in metrics], dtype=bool)
    print("真实精度评估绘图完成")
    print(f"  动作数量: {len(metrics)}")
    print(f"  成功率: {success.mean() * 100:.2f}%")
    print(f"  平均 MPJPE: {np.nanmean(mpjpe):.2f} cm")
    print(f"  平均帧覆盖率: {np.nanmean(coverage):.2f}%")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
