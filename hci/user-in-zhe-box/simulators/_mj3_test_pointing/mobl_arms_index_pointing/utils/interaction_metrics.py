"""人机交互评测指标：用于指向/跟踪精度、遥控驾驶（类方向盘）等离线或在线汇总。"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def scalar_stats(values: Sequence[float]) -> Dict[str, float]:
  """对一组标量误差计算 RMSE、MAE 等。"""
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


def print_run_summary(task_name: str, episodes: List[Dict[str, Any]]) -> None:
  """在评测结束后打印多回合摘要。"""
  print("\n========== 人机交互评测摘要 ==========")
  print(f"任务类型: {task_name}")
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
      if x == x:  # not NaN
        insides.append(float(x))
      if "targets_hit" in ep:
        hits.append(int(ep["targets_hit"]))
    print(format_stats_line("指尖-目标距离 (全回合池化)", scalar_stats(pooled)))
    if insides:
      print(f"各回合目标内时间占比 均值: {float(np.mean(insides)):.4f} (std={float(np.std(insides)):.4f})")
    if hits:
      print(f"各回合命中目标数: mean={float(np.mean(hits)):.2f}, max={max(hits)}")

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
