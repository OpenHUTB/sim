"""人机交互评测指标：用于指向/跟踪精度、遥控驾驶（类方向盘）等离线或在线汇总。

支持两种使用方式：
  1. evaluator.py --metrics 自动评测（原有）
  2. teleop 脚本人工操控后调用 collect_episode_metrics / print_run_summary
"""
from __future__ import annotations

import csv
import io
import os
from typing import Any, Dict, List, Sequence

import numpy as np


def scalar_stats(values: Sequence[float]) -> Dict[str, float]:
  if not values:
    return {"n": 0.0, "rmse": float("nan"), "mae": float("nan"), "mean": float("nan"), "std": float("nan")}
  arr = np.asarray(values, dtype=np.float64)
  return {
      "n": float(arr.size),
      "rmse": float(np.sqrt(np.mean(arr ** 2))),
      "mae": float(np.mean(np.abs(arr))),
      "mean": float(np.mean(arr)),
      "std": float(np.std(arr)),
  }


def inside_fraction(flags: Sequence[bool]) -> float:
  if not flags:
    return float("nan")
  return float(np.mean(np.asarray(flags, dtype=np.float64)))


def format_stats_line(prefix: str, stats: Dict[str, float]) -> str:
  if stats.get("n", 0) == 0:
    return f"{prefix}: (无样本)"
  return (
      f"{prefix}: n={int(stats['n'])}, RMSE={stats['rmse']:.5f}, MAE={stats['mae']:.5f}, "
      f"mean={stats['mean']:.5f}, std={stats['std']:.5f}"
  )


# ---------------------------------------------------------------------------
#  Teleop episode metric collection
# ---------------------------------------------------------------------------

def collect_episode_metrics(
    task_name: str,
    dist_samples: List[float],
    inside_flags: List[bool] | None = None,
    targets_hit: int = 0,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
  ep: Dict[str, Any] = {
      "task": task_name,
      "dist_samples": list(dist_samples),
      "inside_target_frac": inside_fraction(inside_flags) if inside_flags else float("nan"),
      "targets_hit": targets_hit,
  }
  ep.update(scalar_stats(dist_samples))
  if extra:
    ep.update(extra)
  return ep


def export_csv(episodes: List[Dict[str, Any]], path: str) -> None:
  if not episodes:
    return
  cols = ["task", "n", "rmse", "mae", "mean", "std", "inside_target_frac", "targets_hit"]
  extra_keys = sorted({k for ep in episodes for k in ep if k not in cols and k != "dist_samples"
                        and k != "inside_flags" and k != "dist_car_samples"})
  cols.extend(extra_keys)
  with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for ep in episodes:
      w.writerow({k: ep.get(k, "") for k in cols})
  print(f"指标已导出: {os.path.abspath(path)}")


# ---------------------------------------------------------------------------
#  Run summary (used by evaluator and teleop scripts)
# ---------------------------------------------------------------------------

def print_run_summary(task_name: str, episodes: List[Dict[str, Any]], csv_path: str | None = None) -> None:
  print("\n========== 人机交互评测摘要 ==========")
  print(f"任务类型: {task_name}  |  回合数: {len(episodes)}")
  if not episodes:
    print("(无回合数据)")
    return

  if task_name in ("Pointing", "Tracking", "SmallObjectTouch"):
    pooled: List[float] = []
    insides: List[float] = []
    hits: List[int] = []
    for ep in episodes:
      pooled.extend(ep.get("dist_samples", []))
      x = ep.get("inside_target_frac")
      if x == x:
        insides.append(float(x))
      if "targets_hit" in ep:
        hits.append(int(ep["targets_hit"]))
    print(format_stats_line("指尖-目标距离 (全回合池化)", scalar_stats(pooled)))
    if insides:
      print(f"各回合目标内时间占比 均值: {float(np.mean(insides)):.4f} (std={float(np.std(insides)):.4f})")
    if hits:
      print(f"各回合命中目标数: mean={float(np.mean(hits)):.2f}, max={max(hits)}")
      print(f"至少命中一次回合: {sum(1 for h in hits if h > 0)}/{len(hits)}  |  总命中数: {sum(hits)}")

  elif task_name == "RemoteDriving":
    pooled_car: List[float] = []
    successes = 0
    for ep in episodes:
      pooled_car.extend(ep.get("dist_car_samples", []))
      if ep.get("target_parking_success"):
        successes += 1
    print(format_stats_line("车-目标距离 (全回合池化)", scalar_stats(pooled_car)))
    print(f"停车成功回合: {successes}/{len(episodes)}")

  print("======================================\n")

  if csv_path:
    export_csv(episodes, csv_path)
