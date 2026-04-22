#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""控制精度测试：辅助姿态 + 方向键操控手指指向目标。

适用于 Pointing / SmallObjectTouch 任务。手臂从一个参考姿态出发，
方向键（或 WASD）在目标平面的 Y/Z 方向持续平移，松手后保持当前位置。

按键：
  W / ↑  手指向上
  S / ↓  手指向下
  A / ←  手指向左
  D / →  手指向右
  Home   偏置清零（回到参考姿态）
  R      重新开局
  ESC    退出
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

import mujoco
import numpy as np
import pygame

from uitb.simulator import Simulator
from uitb.utils import interaction_metrics

_REFERENCE_ARM_QPOS = np.array(
    [1.4777, 0.8877, -1.1609, 1.2665, -1.0318],
    dtype=np.float32,
)

_Y_VECTOR = np.array([0.00, 0.00, 0.18, 0.00, 0.03], dtype=np.float32)
_Z_VECTOR = np.array([0.10, 0.00, 0.00, 0.05, 0.00], dtype=np.float32)

_OFFSET_LIMIT = 2.3

_IS_WINDOWS = os.name == "nt"
_USER32 = ctypes.windll.user32 if _IS_WINDOWS else None

_VK_W = 0x57
_VK_A = 0x41
_VK_S = 0x53
_VK_D = 0x44
_VK_UP = 0x26
_VK_DOWN = 0x28
_VK_LEFT = 0x25
_VK_RIGHT = 0x27
_VK_HOME = 0x24
_VK_R = 0x52
_VK_ESC = 0x1B
_VK_SHIFT = 0x10


def _joint_ranges(sim: Simulator) -> tuple[np.ndarray, np.ndarray]:
  jidx = np.asarray(sim.bm_model._independent_joints, dtype=np.int32)
  rng = sim._model.jnt_range[jidx].astype(np.float32)
  return rng[:, 0], rng[:, 1]


def _apply_pose(sim: Simulator, arm_qpos: np.ndarray) -> None:
  idx_q = sim.bm_model._independent_qpos
  idx_d = sim.bm_model._independent_dofs
  muscles = sim.bm_model._muscle_actuators
  sim._data.qpos[idx_q] = arm_qpos
  sim._data.qvel[idx_d] = 0.0
  sim._data.act[muscles] = 0.0
  mujoco.mj_forward(sim._model, sim._data)


def _reset_to_ref(sim: Simulator, ref: np.ndarray) -> None:
  _, _ = sim.reset()
  sim._data.qvel[:] = 0.0
  if sim._data.act is not None and sim._data.act.size > 0:
    sim._data.act[:] = 0.0
  _apply_pose(sim, ref)


def _draw_hud(surface: pygame.Surface, lines: list[str]) -> None:
  try:
    font = pygame.font.SysFont("consolas", 16)
  except Exception:
    font = pygame.font.Font(None, 18)
  y = 6
  for txt in lines:
    rendered = font.render(txt, True, (255, 255, 255))
    bg = pygame.Surface((rendered.get_width() + 6, rendered.get_height() + 2), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 140))
    surface.blit(bg, (4, y))
    surface.blit(rendered, (7, y + 1))
    y += rendered.get_height() + 3


def _win_key_pressed(vk_code: int) -> bool:
  if not _IS_WINDOWS or _USER32 is None:
    return False
  return bool(_USER32.GetAsyncKeyState(vk_code) & 0x8000)


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Pointing / Touch precision teleop with assisted pose.")
  p.add_argument("simulator_folder", type=str)
  p.add_argument("--uncloned", dest="cloned", action="store_false")
  p.add_argument("--step", type=float, default=0.05, help="每帧偏置增量")
  p.add_argument("--return-rate", type=float, default=0.0, help="松手时偏置自动回中比例；0 表示保持当前位置")
  p.add_argument("--csv", type=str, default=None, help="导出指标 CSV 路径")
  p.add_argument(
      "--layout", choices=("wasd", "arrows"), default="wasd", help="wasd 或 方向键",
  )
  return p.parse_args()


def main() -> int:
  args = _parse_args()
  if not os.path.isdir(args.simulator_folder):
    print(f"目录不存在: {args.simulator_folder}", file=sys.stderr)
    return 1

  sim = Simulator.get(
      args.simulator_folder,
      render_mode="human",
      render_mode_perception="embed",
      run_parameters={"evaluate": True},
      use_cloned=args.cloned,
  )

  task_name = sim.task.__class__.__name__
  if task_name not in ("Pointing", "SmallObjectTouch"):
    print(f"本脚本仅用于 Pointing / SmallObjectTouch 任务。当前: {task_name}", file=sys.stderr)
    sim.close()
    return 1

  nu_total = int(sim.action_space.shape[0])
  if len(sim.bm_model._independent_qpos) != len(_REFERENCE_ARM_QPOS):
    print("独立关节数量与参考姿态不匹配。", file=sys.stderr)
    sim.close()
    return 1

  arm_lo, arm_hi = _joint_ranges(sim)
  arm_ref = np.clip(_REFERENCE_ARM_QPOS.copy(), arm_lo, arm_hi)

  if args.layout == "wasd":
    K_U, K_D, K_L, K_R = pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d
  else:
    K_U, K_D, K_L, K_R = pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT

  print(f"精度测试遥控（辅助姿态）：{args.layout.upper()} 控制手指  Shift 加速  R 重置  Home 回中  ESC 退出")
  print("请先点击弹出的仿真窗口使其获得焦点。")

  _reset_to_ref(sim, arm_ref)
  if sim._render_window is not None:
    try:
      pygame.key.set_repeat(0, 0)
    except pygame.error:
      pass

  action = np.zeros(nu_total, dtype=np.float32)
  off_y = 0.0
  off_z = 0.0
  running = True
  repeat_set = False
  key_state = {"up": False, "down": False, "left": False, "right": False, "home": False, "reset": False, "esc": False}
  last_key_event = "-"

  ep_dists: list[float] = []
  ep_insides: list[bool] = []
  all_episodes: list[dict] = []
  targets_hit = 0
  episode_idx = 0

  try:
    while running:
      pygame.event.pump()
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
        elif not _IS_WINDOWS and event.type == pygame.KEYDOWN:
          if event.key in (K_U,):
            key_state["up"] = True
            last_key_event = "KEYDOWN:UP"
          elif event.key in (K_D,):
            key_state["down"] = True
            last_key_event = "KEYDOWN:DOWN"
          elif event.key in (K_L,):
            key_state["left"] = True
            last_key_event = "KEYDOWN:LEFT"
          elif event.key in (K_R,):
            key_state["right"] = True
            last_key_event = "KEYDOWN:RIGHT"
          elif event.key == pygame.K_HOME:
            key_state["home"] = True
            last_key_event = "KEYDOWN:HOME"
          elif event.key == pygame.K_r:
            key_state["reset"] = True
            last_key_event = "KEYDOWN:RESET"
          elif event.key == pygame.K_ESCAPE:
            key_state["esc"] = True
            last_key_event = "KEYDOWN:ESC"
        elif not _IS_WINDOWS and event.type == pygame.KEYUP:
          if event.key in (K_U,):
            key_state["up"] = False
            last_key_event = "KEYUP:UP"
          elif event.key in (K_D,):
            key_state["down"] = False
            last_key_event = "KEYUP:DOWN"
          elif event.key in (K_L,):
            key_state["left"] = False
            last_key_event = "KEYUP:LEFT"
          elif event.key in (K_R,):
            key_state["right"] = False
            last_key_event = "KEYUP:RIGHT"
          elif event.key == pygame.K_HOME:
            key_state["home"] = False
            last_key_event = "KEYUP:HOME"
          elif event.key == pygame.K_r:
            key_state["reset"] = False
            last_key_event = "KEYUP:RESET"
          elif event.key == pygame.K_ESCAPE:
            key_state["esc"] = False
            last_key_event = "KEYUP:ESC"

      if not repeat_set and sim._render_window is not None:
        try:
          pygame.key.set_repeat(180, 35)
        except pygame.error:
          pass
        repeat_set = True

      if _IS_WINDOWS:
        up_pressed = _win_key_pressed(_VK_W) or _win_key_pressed(_VK_UP)
        down_pressed = _win_key_pressed(_VK_S) or _win_key_pressed(_VK_DOWN)
        left_pressed = _win_key_pressed(_VK_A) or _win_key_pressed(_VK_LEFT)
        right_pressed = _win_key_pressed(_VK_D) or _win_key_pressed(_VK_RIGHT)
        home_pressed = _win_key_pressed(_VK_HOME)
        reset_pressed = _win_key_pressed(_VK_R)
        esc_pressed = _win_key_pressed(_VK_ESC)
        shift_pressed = _win_key_pressed(_VK_SHIFT)
      else:
        up_pressed = key_state["up"]
        down_pressed = key_state["down"]
        left_pressed = key_state["left"]
        right_pressed = key_state["right"]
        home_pressed = key_state["home"]
        reset_pressed = key_state["reset"]
        esc_pressed = key_state["esc"]
        shift_pressed = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)

      mul = 2.2 if shift_pressed else 1.0
      step = args.step * mul

      if esc_pressed:
        running = False
        continue

      if reset_pressed:
        if ep_dists:
          all_episodes.append(interaction_metrics.collect_episode_metrics(
              task_name, ep_dists, ep_insides, targets_hit))
        ep_dists, ep_insides, targets_hit = [], [], 0
        off_y, off_z = 0.0, 0.0
        _reset_to_ref(sim, arm_ref)
        episode_idx += 1
        if not _IS_WINDOWS:
          key_state["reset"] = False
        continue

      if home_pressed:
        off_y, off_z = 0.0, 0.0
      else:
        if left_pressed:
          off_y -= step
        if right_pressed:
          off_y += step
        if args.return_rate > 0.0 and not (left_pressed or right_pressed):
          off_y *= max(0.0, 1.0 - args.return_rate)
        if up_pressed:
          off_z += step
        if down_pressed:
          off_z -= step
        if args.return_rate > 0.0 and not (up_pressed or down_pressed):
          off_z *= max(0.0, 1.0 - args.return_rate)

      off_y = float(np.clip(off_y, -_OFFSET_LIMIT, _OFFSET_LIMIT))
      off_z = float(np.clip(off_z, -_OFFSET_LIMIT, _OFFSET_LIMIT))
      if abs(off_y) < 1e-4:
        off_y = 0.0
      if abs(off_z) < 1e-4:
        off_z = 0.0

      arm_q = arm_ref + off_y * _Y_VECTOR + off_z * _Z_VECTOR
      arm_q = np.clip(arm_q, arm_lo, arm_hi)
      _apply_pose(sim, arm_q)

      _obs, reward, terminated, truncated, _info = sim.step(action)

      state = sim.get_state()
      dist = float(state.get("dist_to_target_center", float("nan")))
      inside = bool(state.get("inside_target", False))
      hit = bool(state.get("target_hit", False))
      t_hit = int(state.get("targets_hit", 0))
      t_radius = float(state.get("target_radius", 0.05))

      if dist == dist:
        ep_dists.append(dist)
        ep_insides.append(inside)
      if hit:
        targets_hit = t_hit

      focus_ok = bool(pygame.key.get_focused())
      keys_now: list[str] = []
      if up_pressed:
        keys_now.append("UP")
      if down_pressed:
        keys_now.append("DOWN")
      if left_pressed:
        keys_now.append("LEFT")
      if right_pressed:
        keys_now.append("RIGHT")
      if home_pressed:
        keys_now.append("HOME")
      if reset_pressed:
        keys_now.append("RESET")
      if esc_pressed:
        keys_now.append("ESC")
      key_text = ",".join(keys_now) if keys_now else "-"
      last_key_event = key_text

      hud = [
          f"Ep {episode_idx}  |  dist={dist:.4f}  radius={t_radius:.3f}",
          f"inside={inside}  hits={targets_hit}  r={reward:.3f}",
          f"pos_y={off_y:+.2f}  pos_z={off_z:+.2f}  hold={'ON' if args.return_rate <= 0.0 else 'OFF'}",
          f"focus={'YES' if focus_ok else 'NO '}  keys={key_text}",
          f"last={last_key_event}",
          f"{args.layout.upper()} move | Shift fast | Home center | R new | ESC quit",
      ]
      if ep_dists:
        stats = interaction_metrics.scalar_stats(ep_dists)
        hud.append(f"RMSE={stats['rmse']:.4f}  MAE={stats['mae']:.4f}")

      if sim._render_window is not None:
        _draw_hud(pygame.display.get_surface(), hud)
        pygame.display.flip()

      if terminated or truncated:
        if ep_dists:
          all_episodes.append(interaction_metrics.collect_episode_metrics(
              task_name, ep_dists, ep_insides, targets_hit))
        ep_dists, ep_insides, targets_hit = [], [], 0
        off_y, off_z = 0.0, 0.0
        _reset_to_ref(sim, arm_ref)
        episode_idx += 1

  finally:
    if ep_dists:
      all_episodes.append(interaction_metrics.collect_episode_metrics(
          task_name, ep_dists, ep_insides, targets_hit))
    if all_episodes:
      interaction_metrics.print_run_summary(task_name, all_episodes, csv_path=args.csv)
    sim.close()

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
