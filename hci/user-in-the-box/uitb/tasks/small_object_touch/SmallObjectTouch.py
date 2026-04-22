import numpy as np
import mujoco

from .reward_functions import NegativeExpDistanceWithHitBonus
from ..base import BaseTask


class SmallObjectTouch(BaseTask):
  """指尖接近并触碰小目标球体；在 Pointing 式停留判定基础上增加「真实接触」奖励，适合毕业设计「抓取/触碰」演示。"""

  def __init__(self, model, data, end_effector, shoulder, **kwargs):
    super().__init__(model, data, **kwargs)

    if not isinstance(end_effector, list) or len(end_effector) != 2:
      raise RuntimeError("'end_effector' must be a list [type, name], e.g. [\"geom\", \"hand_2distph\"]")
    self._end_effector = end_effector

    if not isinstance(shoulder, list) or len(shoulder) != 2:
      raise RuntimeError("'shoulder' must be a list with two elements")
    self._shoulder = shoulder

    self._touch_bonus = float(kwargs.get("touch_bonus", 0.25))
    self._touch_margin_m = float(kwargs.get("touch_margin_m", 0.004))

    self._steps_since_last_hit = 0
    self._max_steps_without_hit = self._action_sample_freq * 4

    self._info = {"target_hit": False, "inside_target": False, "target_spawned": False, "terminated": False,
                  "truncated": False, "termination": False, "object_contact": False}

    self._trial_idx = 0
    self._max_trials = kwargs.get("max_trials", 8)
    self._targets_hit = 0

    self._steps_inside_target = 0
    self._dwell_threshold = int(0.5 * self._action_sample_freq)

    self._target_radius_limit = kwargs.get("target_radius_limit", np.array([0.02, 0.05]))
    self._target_radius = self._target_radius_limit[0]
    self._new_target_distance_threshold = 2 * self._target_radius_limit[1]

    self._reward_function = NegativeExpDistanceWithHitBonus(k=12)

    mujoco.mj_forward(model, data)

    self._target_origin = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos + np.array([0.55, -0.1, 0])
    self._target_position = self._target_origin.copy()
    self._target_limits_y = np.array([-0.28, 0.28])
    self._target_limits_z = np.array([-0.28, 0.28])

    model.geom("target-plane").size = np.array([0.005,
                                                (self._target_limits_y[1] - self._target_limits_y[0]) / 2,
                                                (self._target_limits_z[1] - self._target_limits_z[0]) / 2])
    model.body("target-plane").pos = self._target_origin

    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "for_testing")] = np.array([1.1, -0.9, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "for_testing")] = np.array(
        [0.6582, 0.6577, 0.2590, 0.2588])

  def _proximity_touch(self, dist_surface):
    return dist_surface <= self._touch_margin_m

  def _update(self, model, data):
    terminated = False
    truncated = False
    self._info["target_spawned"] = False

    ee_position = getattr(data, self._end_effector[0])(self._end_effector[1]).xpos
    dist = np.linalg.norm(self._target_position - (ee_position - self._target_origin))
    dist_surface = dist - self._target_radius
    touch = self._proximity_touch(dist_surface)
    self._info["object_contact"] = touch

    if dist < self._target_radius:
      self._steps_inside_target += 1
      self._info["inside_target"] = True
    else:
      self._steps_inside_target = 0
      self._info["inside_target"] = False

    if self._info["inside_target"] and self._steps_inside_target >= self._dwell_threshold:
      self._info["target_hit"] = True
      self._trial_idx += 1
      self._targets_hit += 1
      self._steps_since_last_hit = 0
      self._steps_inside_target = 0
      self._info["acc_dist"] += dist
      self._spawn_target(model, data)
      self._info["target_spawned"] = True
    else:
      self._info["target_hit"] = False
      self._steps_since_last_hit += 1
      if self._steps_since_last_hit >= self._max_steps_without_hit:
        self._steps_since_last_hit = 0
        self._trial_idx += 1
        self._info["acc_dist"] += dist
        self._spawn_target(model, data)
        self._info["target_spawned"] = True

    if self._trial_idx >= self._max_trials:
      self._info["dist_from_target"] = self._info["acc_dist"] / max(self._trial_idx, 1)
      truncated = True
      self._info["termination"] = "max_trials_reached"

    reward = self._reward_function.get(self, dist - self._target_radius, self._info.copy())
    if touch:
      reward += self._touch_bonus

    return reward, terminated, truncated, self._info.copy()

  def _get_state(self, model, data):
    state = dict()
    ee_position = getattr(data, self._end_effector[0])(self._end_effector[1]).xpos.copy()
    dist_center = float(np.linalg.norm(self._target_position - (ee_position - self._target_origin)))
    state["ee_position"] = ee_position
    state["dist_to_target_center"] = dist_center
    state["dist_to_target_surface"] = float(dist_center - self._target_radius)
    state["target_position"] = self._target_origin.copy() + self._target_position.copy()
    state["target_radius"] = self._target_radius
    state["trial_idx"] = self._trial_idx
    state["targets_hit"] = self._targets_hit
    state.update(self._info)
    return state

  def _reset(self, model, data):
    self._steps_since_last_hit = 0
    self._steps_inside_target = 0
    self._trial_idx = 0
    self._targets_hit = 0
    self._info = {"target_hit": False, "inside_target": False, "target_spawned": False,
                  "terminated": False, "truncated": False, "termination": False,
                  "llc_dist_from_target": 0, "dist_from_target": 0, "acc_dist": 0, "object_contact": False}
    self._spawn_target(model, data)
    return self._info

  def _spawn_target(self, model, data):
    for _ in range(10):
      target_y = self._rng.uniform(*self._target_limits_y)
      target_z = self._rng.uniform(*self._target_limits_z)
      new_position = np.array([0, target_y, target_z])
      distance = np.linalg.norm(self._target_position - new_position)
      if distance > self._new_target_distance_threshold:
        break
    self._target_position = new_position
    model.body("target").pos[:] = self._target_origin + self._target_position
    self._target_radius = self._rng.uniform(*self._target_radius_limit)
    model.geom("target").size[0] = self._target_radius
    mujoco.mj_forward(model, data)

  def get_stateful_information(self, model, data):
    targets_hit = -1.0 + 2 * (self._trial_idx / self._max_trials)
    dwell_time = -1.0 + 2 * np.min([1.0, self._steps_inside_target / self._dwell_threshold])
    return np.array([dwell_time, targets_hit])
