#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""键盘实时操控仿真（肌肉控制量 [-1,1]，与训练时 action_space 一致）。

操作（主键盘）：
  ← / →     切换当前驱动的执行器（肌肉）通道索引
  ↑ / ↓     增大 / 减小当前通道的控制量（步长见 --step，按住 Shift 为大步长）
  Home      将所有通道置零
  PageUp / PageDown   上一组 / 下一组：每次跳过 --group 个通道（便于高维快速浏览）
  ESC       退出

说明：本环境默认不是「键盘直接对应关节」，而是对 PPO 使用的同一套 muscle/motor
动作向量做增量编辑；适合毕设演示人机对比或手动试控。
若仿真为 RemoteDriving，可改用 arcade_drive_teleop.py（WASD 前推/左右分组）。
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pygame
import mujoco

from uitb.simulator import Simulator


def _actuator_labels(simulator: Simulator) -> list[str]:
  m = simulator._model
  n = simulator.bm_model.nu
  out = []
  for i in range(n):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    out.append(name if name else f"actuator_{i}")
  return out


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Keyboard teleop for uitb Simulator (muscle actions in [-1,1]).")
  p.add_argument("simulator_folder", type=str, help="已构建的仿真目录（含 config.yaml）")
  p.add_argument("--uncloned", dest="cloned", action="store_false", help="使用 uitb 源码而非克隆包")
  p.add_argument("--step", type=float, default=0.04, help="↑/↓ 单步调整量（默认 0.04）")
  p.add_argument("--step-fast", type=float, default=0.12, help="按住 Shift 时步长（默认 0.12）")
  p.add_argument("--group", type=int, default=4, help="PageUp/PageDown 每次跳过的通道数（默认 4）")
  return p.parse_args()


def main() -> int:
  args = _parse_args()
  if not os.path.isdir(args.simulator_folder):
    print(f"目录不存在: {args.simulator_folder}", file=sys.stderr)
    return 1

  run_params = {"evaluate": True}
  sim = Simulator.get(
      args.simulator_folder,
      render_mode="human",
      render_mode_perception="embed",
      run_parameters=run_params,
      use_cloned=args.cloned,
  )

  if "llc" in sim.config:
    print("当前仿真启用了 HRL/LLC，键盘脚本仅支持普通肌肉控制模式。", file=sys.stderr)
    sim.close()
    return 1

  nu_total = int(sim.action_space.shape[0])
  nu_bm = int(sim.bm_model.nu)
  if nu_bm < 1:
    print("bm_model.nu < 1，无可键盘操控的执行器。", file=sys.stderr)
    sim.close()
    return 1
  nu_perc = nu_total - nu_bm
  labels = _actuator_labels(sim)

  action = np.zeros(nu_total, dtype=np.float32)
  ch = 0
  repeat_set = False

  print("键盘操控已启动，请看弹出的 PyGame 窗口标题与下方说明。")
  print("←→ 换通道  ↑↓ 调值  Home 清零  PgUp/PgDn 跳组  Shift=大步长  ESC 退出")
  if nu_perc > 0:
    print(f"注意：另有 {nu_perc} 维感知执行器已置零（本脚本仅控制前 {nu_bm} 维肌肉）。")

  _, _ = sim.reset()
  terminated = truncated = False
  running = True

  try:
    while running:
      step_size = args.step_fast if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else args.step

      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
          continue
        if event.type != pygame.KEYDOWN:
          continue
        k = event.key
        if k == pygame.K_ESCAPE:
          running = False
          continue
        if k == pygame.K_LEFT:
          ch = (ch - 1) % nu_bm
        elif k == pygame.K_RIGHT:
          ch = (ch + 1) % nu_bm
        elif k == pygame.K_UP:
          action[ch] = float(np.clip(action[ch] + step_size, -1.0, 1.0))
        elif k == pygame.K_DOWN:
          action[ch] = float(np.clip(action[ch] - step_size, -1.0, 1.0))
        elif k == pygame.K_HOME:
          action[:] = 0.0
        elif k == pygame.K_PAGEUP:
          ch = (ch - max(1, args.group)) % nu_bm
        elif k == pygame.K_PAGEDOWN:
          ch = (ch + max(1, args.group)) % nu_bm

      _obs, reward, terminated, truncated, info = sim.step(action)

      if not repeat_set and sim._render_window is not None:
        pygame.key.set_repeat(280, 45)
        repeat_set = True

      cap_ch = max(nu_bm - 1, 0)
      caption = (
          f"Teleop | {ch}/{cap_ch} {labels[ch][:26]} | "
          f"u={action[ch]:+.2f} | r={reward:.3f} | ESC quit"
      )
      if sim._render_window is not None:
        pygame.display.set_caption(caption)

      if terminated or truncated:
        _, _ = sim.reset()
        terminated = truncated = False
  finally:
    sim.close()

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
