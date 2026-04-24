#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HMI 统一启动器：三个场景对应三个辅助姿态遥控脚本。

1) 遥控驾驶 (arcade_drive_teleop.py)  — WASD 推拉摇杆
2) 控制精度 (pointing_teleop.py)      — WASD 控制手指指向目标
3) 小物体触碰 (touch_teleop.py)       — WASD 控制手指触碰小目标

加 --eval 可改为自动评测路径（需要 checkpoint）。

仿真目录可通过环境变量覆盖：
  HMI_SIM_REMOTE_DRIVING, HMI_SIM_PRECISION, HMI_SIM_TOUCH
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _root() -> str:
  return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _pick_sim(env_name: str, default_rel: str) -> str:
  v = os.environ.get(env_name)
  if v:
    return os.path.abspath(v)
  return os.path.join(_root(), default_rel.replace("/", os.sep))


def main() -> int:
  ap = argparse.ArgumentParser(
      description="肌肉驱动手臂 HMI 系统 — 统一启动器"
  )
  ap.add_argument(
      "what",
      nargs="?",
      choices=("1", "2", "3", "arcade", "precision", "touch"),
      default=None,
      help="1=遥控驾驶 | 2=控制精度 | 3=小物体触碰",
  )
  ap.add_argument("--eval", action="store_true", help="改为自动评测路径（需 checkpoint）")
  ap.add_argument("--episodes", type=int, default=10, help="自动评测回合数")
  ap.add_argument("--csv", type=str, default=None, help="导出指标 CSV 路径（人工操控模式）")
  ap.set_defaults(cloned=True)
  ap.add_argument("--uncloned", dest="cloned", action="store_false")
  args = ap.parse_args()

  scripts = os.path.join(_root(), "uitb", "scripts")

  if args.what is None:
    print("=" * 50)
    print("  肌肉驱动手臂 HMI 系统")
    print("=" * 50)
    print()
    print("用法: python uitb/scripts/hmi_quickstart.py <1|2|3>")
    print()
    print("  1 / arcade     场景1: 遥控驾驶（WASD 推拉摇杆）")
    print("  2 / precision  场景2: 控制精度（WASD 控制手指指向目标）")
    print("  3 / touch      场景3: 小物体触碰（WASD 控制手指触碰小目标）")
    print()
    print("  加 --eval 改为自动评测（需 checkpoint）")
    print("  加 --csv metrics.csv 导出指标")
    print()
    print("默认仿真目录：")
    print(f"  遥控驾驶: {_pick_sim('HMI_SIM_REMOTE_DRIVING', 'simulators/mobl_arms_index_remote_driving')}")
    print(f"  控制精度: {_pick_sim('HMI_SIM_PRECISION', 'simulators/mobl_arms_index_pointing')}")
    print(f"  小物体:   {_pick_sim('HMI_SIM_TOUCH', 'simulators/grad_small_object_touch')}")
    print(f"\n仓库根: {_root()}")
    return 0

  clone_flag = [] if args.cloned else ["--uncloned"]
  csv_flag = ["--csv", args.csv] if args.csv else []
  key = args.what

  if key in ("1", "arcade"):
    sim = _pick_sim("HMI_SIM_REMOTE_DRIVING", "simulators/mobl_arms_index_remote_driving")
    if not os.path.isdir(sim):
      print(f"目录不存在: {sim}", file=sys.stderr)
      return 1
    if args.eval:
      cmd = [sys.executable, "-m", "uitb.test.evaluator", sim,
             "--metrics", "--deterministic", "--num_episodes", str(args.episodes)] + clone_flag
    else:
      cmd = [sys.executable, os.path.join(scripts, "arcade_drive_teleop.py"), sim] + clone_flag + csv_flag
    print("启动:", " ".join(cmd))
    return subprocess.call(cmd, cwd=_root())

  if key in ("2", "precision"):
    sim = _pick_sim("HMI_SIM_PRECISION", "simulators/mobl_arms_index_pointing")
    if not os.path.isdir(sim):
      print(f"目录不存在: {sim}", file=sys.stderr)
      return 1
    if args.eval:
      cmd = [sys.executable, "-m", "uitb.test.evaluator", sim,
             "--metrics", "--deterministic", "--num_episodes", str(args.episodes)] + clone_flag
    else:
      cmd = [sys.executable, os.path.join(scripts, "pointing_teleop.py"), sim] + clone_flag + csv_flag
    print("启动:", " ".join(cmd))
    return subprocess.call(cmd, cwd=_root())

  # touch
  sim = _pick_sim("HMI_SIM_TOUCH", "simulators/grad_small_object_touch")
  if not os.path.isdir(sim):
    print(f"目录不存在: {sim}", file=sys.stderr)
    return 1
  if args.eval:
    cmd = [sys.executable, "-m", "uitb.test.evaluator", sim,
           "--metrics", "--deterministic", "--num_episodes", str(args.episodes)] + clone_flag
  else:
    cmd = [sys.executable, os.path.join(scripts, "touch_teleop.py"), sim] + clone_flag + csv_flag
  print("启动:", " ".join(cmd))
  return subprocess.call(cmd, cwd=_root())


if __name__ == "__main__":
  raise SystemExit(main())
