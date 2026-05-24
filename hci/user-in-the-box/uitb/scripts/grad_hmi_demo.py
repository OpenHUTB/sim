#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""毕业设计：肌肉骨骼手臂人机交互三场景说明与评测辅助。

场景 A — 视觉引导的「类方向盘」操控：遥控驾驶任务（拇指杆 + 小车停入目标区），配置见
  uitb/configs/mobl_arms_index_remote_driving.yaml

场景 B — 控制精度：指向 / 跟踪任务，使用评测脚本 --metrics

场景 C — 小物体触碰：SmallObjectTouch 任务，配置见
  uitb/configs/grad_small_object_touch.yaml

首次使用需用 Simulator.build 生成仿真包目录，再训练或运行评测。

键盘实时操控（肌肉通道增量，非策略网络）：
  python uitb/scripts/keyboard_teleop.py <仿真目录>
遥控驾驶 WASD 类「前推后拉 / 左右」近似映射：
  python uitb/scripts/arcade_drive_teleop.py <RemoteDriving 仿真目录>

一键三场景：python uitb/scripts/hmi_quickstart.py
  参数 1=遥控WASD  2=精度评测  3=小物体触碰键盘（可加 --eval 改为仅评测）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _repo_root() -> str:
  # uitb/scripts/grad_hmi_demo.py -> repo root
  return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _config(name: str) -> str:
  return os.path.join(_repo_root(), "uitb", "configs", name)


CONFIGS = {
    "steering": _config("mobl_arms_index_remote_driving.yaml"),
    "touch": _config("grad_small_object_touch.yaml"),
    "precision": _config("mobl_arms_index_pointing_precision.yaml"),
}


def _print_build_help() -> None:
  print("在仓库根目录执行：\n")
  for name, cfg in CONFIGS.items():
    print(f"  [{name}]")
    print(f"    python -c \"from uitb.simulator import Simulator; Simulator.build(r'{cfg}')\"")
  print("\n生成目录默认在 uitb/simulators/<simulator_name>/（或由 config 里的 simulator_folder 指定）。")


def _run_eval(sim_folder: str, episodes: int, human: bool, record: bool) -> int:
  cmd = [
      sys.executable, "-m", "uitb.test.evaluator",
      sim_folder,
      "--metrics",
      "--deterministic",
      "--num_episodes", str(episodes),
  ]
  if human:
    cmd.append("--human")
  if record:
    cmd.append("--record")
  print("运行:", " ".join(cmd))
  return subprocess.call(cmd, cwd=_repo_root())


def main() -> int:
  parser = argparse.ArgumentParser(description="毕业设计 HMI 三场景说明与评测")
  parser.add_argument(
      "simulator_folder",
      nargs="?",
      default=None,
      help="已构建的仿真目录（内含 config.yaml）；若省略则只打印构建说明",
  )
  parser.add_argument("--episodes", type=int, default=5, help="评测回合数")
  parser.add_argument("--human", action="store_true", help="PyGame 实时窗口")
  parser.add_argument("--record", action="store_true", help="录制 mp4")
  args = parser.parse_args()

  if args.simulator_folder is None:
    _print_build_help()
    return 0

  if not os.path.isdir(args.simulator_folder):
    print(f"目录不存在: {args.simulator_folder}")
    _print_build_help()
    return 1

  return _run_eval(os.path.abspath(args.simulator_folder), args.episodes, args.human, args.record)


if __name__ == "__main__":
  raise SystemExit(main())
