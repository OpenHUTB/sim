#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""遥控驾驶任务专用：姿态保持 + 键盘微调的辅助遥控。

仅适用于 task 为 RemoteDriving 的仿真。这个版本不再把 WASD 直接映射到 26 维肌肉增量，
而是使用一个“辅助姿态”：

1. 零输入时，把手臂拉回离摇杆很近的参考姿态，避免 reset 后随机姿态/肌肉导致手臂自己漂。
2. W/S 直接给拇指杆一个小幅辅助偏转，保证车真的会动。
3. A/D 只对手臂做横向微调，用于演示“人手正在操控摇杆”。
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys

# 直接运行 `python uitb/scripts/本脚本.py` 时，sys.path 不含仓库根目录，会导致 import uitb 失败
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

import mujoco
import numpy as np
import pygame

from uitb.simulator import Simulator
from uitb.utils import interaction_metrics

_REFERENCE_ARM_QPOS = np.array(
    [0.73238167, 1.17101506, 0.27191627, 0.98772476, 1.08483110],
    dtype=np.float32,
)
_PUSH_QPOS_VECTOR = np.array([0.03, -0.06, 0.01, -0.08, 0.0], dtype=np.float32)
_LATERAL_QPOS_VECTOR = np.array([0.00, 0.02, 0.12, 0.00, 0.05], dtype=np.float32)
_JOYSTICK_LIMIT = 0.30

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


def _independent_joint_ranges(sim: Simulator) -> tuple[np.ndarray, np.ndarray]:
  jidx = np.asarray(sim.bm_model._independent_joints, dtype=np.int32)
  rng = sim._model.jnt_range[jidx].astype(np.float32)
  return rng[:, 0], rng[:, 1]


def _apply_assisted_pose(sim: Simulator, arm_qpos: np.ndarray, joystick_q: float) -> None:
  arm_qpos_idx = sim.bm_model._independent_qpos
  arm_dof_idx = sim.bm_model._independent_dofs
  muscles = sim.bm_model._muscle_actuators

  sim._data.qpos[arm_qpos_idx] = arm_qpos
  sim._data.qvel[arm_dof_idx] = 0.0
  sim._data.act[muscles] = 0.0

  sim._data.joint("thumb-stick-1:rot-x").qpos[0] = joystick_q
  sim._data.joint("thumb-stick-1:rot-x").qvel[0] = 0.0
  sim._data.joint("thumb-stick-2:rot-x").qpos[0] = 0.0
  sim._data.joint("thumb-stick-2:rot-x").qvel[0] = 0.0

  mujoco.mj_forward(sim._model, sim._data)


def _reset_episode_to_reference(sim: Simulator, arm_qpos_ref: np.ndarray) -> None:
  _, _ = sim.reset()

  # 保留任务随机生成的车/目标位置，只把手臂与摇杆复位到可控的参考状态。
  car_q = float(sim._data.joint("car").qpos[0])
  target_q = float(sim._data.joint("target").qpos[0])

  sim._data.qpos[:] = 0.0
  sim._data.qvel[:] = 0.0
  if sim._data.act is not None and sim._data.act.size > 0:
    sim._data.act[:] = 0.0

  sim._data.joint("car").qpos[0] = car_q
  sim._data.joint("target").qpos[0] = target_q
  _apply_assisted_pose(sim, arm_qpos_ref, joystick_q=0.0)


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


def _joystick_bar(value: float, limit: float, width: int = 20) -> str:
  frac = value / limit if abs(limit) > 1e-8 else 0.0
  frac = max(-1.0, min(1.0, frac))
  mid = width // 2
  pos = int(round(frac * mid))
  bar = ["-"] * width
  bar[mid] = "|"
  idx = mid + pos
  idx = max(0, min(width - 1, idx))
  bar[idx] = "#"
  return "[" + "".join(bar) + "]"


def _win_key_pressed(vk_code: int) -> bool:
  if not _IS_WINDOWS or _USER32 is None:
    return False
  return bool(_USER32.GetAsyncKeyState(vk_code) & 0x8000)


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Assistive WASD teleop for RemoteDriving.")
  p.add_argument("simulator_folder", type=str)
  p.add_argument("--uncloned", dest="cloned", action="store_false")
  p.add_argument("--step", type=float, default=0.035, help="每帧摇杆目标角增量（弧度）")
  p.add_argument("--return-rate", type=float, default=0.30, help="松手时摇杆每帧回中比例")
  p.add_argument("--lateral-step", type=float, default=0.08, help="A/D 手臂横向偏置增量（归一化尺度）")
  p.add_argument("--lateral-return", type=float, default=0.35, help="松开 A/D 后横向偏置每帧衰减比例")
  p.add_argument("--csv", type=str, default=None, help="导出指标 CSV 路径")
  p.add_argument(
      "--layout",
      choices=("wasd", "arrows"),
      default="wasd",
      help="wasd 或 方向键",
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

  if sim.task.__class__.__name__ != "RemoteDriving":
    print("本脚本仅用于 RemoteDriving 任务（遥控驾驶 + 拇指杆）。当前任务："
          f"{sim.task.__class__.__name__}", file=sys.stderr)
    sim.close()
    return 1

  if "llc" in sim.config:
    print("当前仿真启用了 HRL/LLC，不支持本脚本。", file=sys.stderr)
    sim.close()
    return 1

  nu_total = int(sim.action_space.shape[0])
  if len(sim.bm_model._independent_qpos) != len(_REFERENCE_ARM_QPOS):
    print("独立关节数量与参考姿态不匹配，无法启用辅助姿态遥控。", file=sys.stderr)
    sim.close()
    return 1

  arm_lo, arm_hi = _independent_joint_ranges(sim)
  arm_qpos_ref = np.clip(_REFERENCE_ARM_QPOS.copy(), arm_lo, arm_hi)

  if args.layout == "wasd":
    K_F, K_B, K_L, K_R = pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d
  else:
    K_F, K_B, K_L, K_R = pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT

  print(
      "Arcade 遥控（辅助姿态）：W/S 推拉摇杆  A/D 手臂横移  R 重置局  Home 回中  ESC 退出"
      f"  | 回中={args.return_rate:.2f}/步"
  )
  print("请先点击弹出的仿真窗口使其获得焦点，再按键（否则 WASD 可能无响应）。")

  _reset_episode_to_reference(sim, arm_qpos_ref)
  if sim._render_window is not None:
    try:
      pygame.key.set_repeat(0, 0)
    except pygame.error:
      pass

  action = np.zeros(nu_total, dtype=np.float32)
  joystick_cmd = 0.0
  lateral_cmd = 0.0
  running = True
  repeat_set = False
  key_state = {"fwd": False, "back": False, "left": False, "right": False, "home": False, "reset": False, "esc": False}
  last_key_event = "-"
  ep_car_dists: list[float] = []
  all_episodes: list[dict] = []
  episode_idx = 0
  ep_reward = 0.0

  try:
    while running:
      pygame.event.pump()
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
        elif not _IS_WINDOWS and event.type == pygame.KEYDOWN:
          if event.key in (pygame.K_w, pygame.K_UP):
            key_state["fwd"] = True
            last_key_event = "KEYDOWN:FWD"
          elif event.key in (pygame.K_s, pygame.K_DOWN):
            key_state["back"] = True
            last_key_event = "KEYDOWN:BACK"
          elif event.key in (pygame.K_a, pygame.K_LEFT):
            key_state["left"] = True
            last_key_event = "KEYDOWN:LEFT"
          elif event.key in (pygame.K_d, pygame.K_RIGHT):
            key_state["right"] = True
            last_key_event = "KEYDOWN:RIGHT"
          elif event.key == pygame.K_HOME:
            key_state["home"] = True
            last_key_event = "KEYDOWN:HOME"
          if event.key == pygame.K_ESCAPE:
            running = False
          elif event.key == pygame.K_r:
            key_state["reset"] = True
            last_key_event = "KEYDOWN:RESET"
        elif not _IS_WINDOWS and event.type == pygame.KEYUP:
          if event.key in (pygame.K_w, pygame.K_UP):
            key_state["fwd"] = False
            last_key_event = "KEYUP:FWD"
          elif event.key in (pygame.K_s, pygame.K_DOWN):
            key_state["back"] = False
            last_key_event = "KEYUP:BACK"
          elif event.key in (pygame.K_a, pygame.K_LEFT):
            key_state["left"] = False
            last_key_event = "KEYUP:LEFT"
          elif event.key in (pygame.K_d, pygame.K_RIGHT):
            key_state["right"] = False
            last_key_event = "KEYUP:RIGHT"
          elif event.key == pygame.K_HOME:
            key_state["home"] = False
            last_key_event = "KEYUP:HOME"

      speed_mul = 1.6 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1.0
      joy_step = args.step * speed_mul
      lat_step = args.lateral_step * speed_mul

      if not repeat_set and sim._render_window is not None:
        try:
          pygame.key.set_repeat(180, 35)
        except pygame.error:
          pass
        repeat_set = True

      if _IS_WINDOWS:
        shift_pressed = _win_key_pressed(_VK_SHIFT)
        speed_mul = 1.6 if shift_pressed else 1.0
        joy_step = args.step * speed_mul
        lat_step = args.lateral_step * speed_mul

        fwd_pressed = _win_key_pressed(_VK_W) or _win_key_pressed(_VK_UP)
        back_pressed = _win_key_pressed(_VK_S) or _win_key_pressed(_VK_DOWN)
        left_pressed = _win_key_pressed(_VK_A) or _win_key_pressed(_VK_LEFT)
        right_pressed = _win_key_pressed(_VK_D) or _win_key_pressed(_VK_RIGHT)
        home_pressed = _win_key_pressed(_VK_HOME)
        reset_pressed = _win_key_pressed(_VK_R)
        esc_pressed = _win_key_pressed(_VK_ESC)

        key_state["fwd"] = fwd_pressed
        key_state["back"] = back_pressed
        key_state["left"] = left_pressed
        key_state["right"] = right_pressed
        key_state["home"] = home_pressed
        key_state["reset"] = reset_pressed
        key_state["esc"] = esc_pressed

        pressed_names: list[str] = []
        if fwd_pressed:
          pressed_names.append("FWD")
        if back_pressed:
          pressed_names.append("BACK")
        if left_pressed:
          pressed_names.append("LEFT")
        if right_pressed:
          pressed_names.append("RIGHT")
        if home_pressed:
          pressed_names.append("HOME")
        if reset_pressed:
          pressed_names.append("RESET")
        if esc_pressed:
          pressed_names.append("ESC")
        last_key_event = "WIN32:" + (",".join(pressed_names) if pressed_names else "-")
      else:
        fwd_pressed = key_state["fwd"]
        back_pressed = key_state["back"]
        left_pressed = key_state["left"]
        right_pressed = key_state["right"]
        home_pressed = key_state["home"]
        reset_pressed = key_state["reset"]
        esc_pressed = key_state["esc"]

      if esc_pressed:
        running = False
        continue

      if reset_pressed:
        if ep_car_dists:
          all_episodes.append({"task": "RemoteDriving", "dist_car_samples": list(ep_car_dists),
                               "target_parking_success": False, "ep_reward": ep_reward})
        ep_car_dists, ep_reward = [], 0.0
        joystick_cmd = 0.0
        lateral_cmd = 0.0
        _reset_episode_to_reference(sim, arm_qpos_ref)
        episode_idx += 1
        if not _IS_WINDOWS:
          key_state["reset"] = False
        continue

      if home_pressed:
        joystick_cmd = 0.0
        lateral_cmd = 0.0
      else:
        if fwd_pressed:
          joystick_cmd -= joy_step
        if back_pressed:
          joystick_cmd += joy_step
        if not (fwd_pressed or back_pressed):
          joystick_cmd *= max(0.0, 1.0 - args.return_rate)

        if left_pressed:
          lateral_cmd -= lat_step
        if right_pressed:
          lateral_cmd += lat_step
        if not (left_pressed or right_pressed):
          lateral_cmd *= max(0.0, 1.0 - args.lateral_return)

      joystick_cmd = float(np.clip(joystick_cmd, -_JOYSTICK_LIMIT, _JOYSTICK_LIMIT))
      lateral_cmd = float(np.clip(lateral_cmd, -1.0, 1.0))

      push_scale = joystick_cmd / _JOYSTICK_LIMIT if _JOYSTICK_LIMIT > 1e-8 else 0.0
      arm_qpos = arm_qpos_ref + push_scale * _PUSH_QPOS_VECTOR + lateral_cmd * _LATERAL_QPOS_VECTOR
      arm_qpos = np.clip(arm_qpos, arm_lo, arm_hi)
      _apply_assisted_pose(sim, arm_qpos, joystick_cmd)

      _obs, reward, terminated, truncated, _info = sim.step(action)
      ep_reward += reward

      jpos = float(sim._data.joint("thumb-stick-1:rot-x").qpos[0])
      state = sim.get_state()
      d_car = float(state.get("dist_car_to_target", float("nan")))
      inside_target = bool(state.get("inside_target", False))
      success_now = bool(
          state.get("target_parking_success", False) or state.get("target_hit", False)
      )
      if d_car == d_car:
        ep_car_dists.append(d_car)

      focus_ok = bool(pygame.key.get_focused())
      keys_now: list[str] = []
      if fwd_pressed:
        keys_now.append("FWD")
      if back_pressed:
        keys_now.append("BACK")
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

      bar = _joystick_bar(joystick_cmd, _JOYSTICK_LIMIT)
      hud = [
          f"Ep {episode_idx}  |  stick {bar}  q={jpos:+.3f}",
          f"car_dist={d_car:.3f}  lat={lateral_cmd:+.2f}  r={reward:.3f}",
          f"inside_target={inside_target}  success={success_now}",
          f"focus={'YES' if focus_ok else 'NO '}  keys={key_text}",
          f"last={last_key_event}",
          f"W/S push | A/D lateral | Home reset | R new | ESC quit",
      ]

      if sim._render_window is not None:
        _draw_hud(pygame.display.get_surface(), hud)
        pygame.display.flip()

      if terminated or truncated:
        # RemoteDriving 的 get_state 里是 target_hit；evaluator 里映射为 target_parking_success
        success = bool(
            state.get("target_parking_success", False) or state.get("target_hit", False)
        )
        if ep_car_dists:
          all_episodes.append({"task": "RemoteDriving", "dist_car_samples": list(ep_car_dists),
                               "target_parking_success": success, "ep_reward": ep_reward})
        ep_car_dists, ep_reward = [], 0.0
        joystick_cmd = 0.0
        lateral_cmd = 0.0
        _reset_episode_to_reference(sim, arm_qpos_ref)
        episode_idx += 1
  finally:
    if ep_car_dists:
      all_episodes.append({"task": "RemoteDriving", "dist_car_samples": list(ep_car_dists),
                           "target_parking_success": False, "ep_reward": ep_reward})
    if all_episodes:
      interaction_metrics.print_run_summary("RemoteDriving", all_episodes, csv_path=args.csv)
    sim.close()

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
