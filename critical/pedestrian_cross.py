# 场景6: 行人横穿 — 慢速横穿, 在车头前停留1s, 自车急刹同时行人暂停
import carla
from scenarios.base_scenario import BaseScenario
from utils.carla_utils import spawn_pedestrian_at


class PedestrianCrossScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "pedestrian_cross"
        self.category = "pedestrian_danger"
        self.weather = carla.WeatherParameters(
            cloudiness=5.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=0.0, fog_density=0.0, fog_distance=0.0,
            wetness=0.0, sun_azimuth_angle=90.0, sun_altitude_angle=45.0)
        self.ego_speed_ms = 30.0 / 3.6
        self._trigger_step = int(2.5 * 20)      # 2.5s 开始横穿
        self._half_cross_time = int(2.0 * 20)    # 半程 2s（慢速）
        self._pause_time = int(1.0 * 20)         # 车头前停留 1s

    def get_env_config(self):
        cfg = super().get_env_config()
        cfg["action_space"] = 2
        return cfg

    def _spawn_actors(self):
        self._spawn_ego()
        self._spawn_pedestrian()
        self.world.tick()

    def _spawn_scenario_actors_impl(self):
        self._phase = "idle"           # idle → first_half → paused → second_half → done
        self._phase_start = 0
        self._ped_start_y = None
        self._spawn_pedestrian()

    def _spawn_pedestrian(self):
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        ped_x = ego_loc.x + fwd.x * 30 + right.x * 6.0
        ped_y = ego_loc.y + fwd.y * 30 + right.y * 6.0
        self._ped_yaw = ego_tf.rotation.yaw - 90.0

        ped_loc = carla.Location(x=ped_x, y=ped_y, z=ego_loc.z + 0.5)
        ped, ctrl = spawn_pedestrian_at(self.world, ped_loc, speed_ms=0.0)
        ped.set_transform(carla.Transform(ped_loc, carla.Rotation(yaw=self._ped_yaw)))
        self.pedestrians.append((ped, ctrl))
        self._ped = ped
        self._start_y = ped_y                          # 右侧人行道
        self._lane_y = ego_loc.y                        # 自车车道中心（停留点）
        self._target_y = ego_loc.y - right.y * 6.0     # 对侧人行道

    # ================================================================
    # RL 回调
    # ================================================================
    def step_callback(self, step_count):
        if self._ped is None:
            return

        # 触发横穿
        if step_count >= self._trigger_step and self._phase == "idle":
            self._phase = "first_half"
            self._phase_start = step_count
            self._ped_start_y = self._ped.get_location().y

        if self._phase == "first_half":
            self._move_ped(step_count, self._ped_start_y, self._lane_y,
                           self._half_cross_time, "paused")

        elif self._phase == "paused":
            if step_count - self._phase_start >= self._pause_time:
                self._phase = "second_half"
                self._phase_start = step_count
                self._ped_start_y = self._ped.get_location().y

        elif self._phase == "second_half":
            self._move_ped(step_count, self._ped_start_y, self._target_y,
                           self._half_cross_time, "done")

    def _move_ped(self, step_count, start_y, end_y, duration, next_phase):
        """线性位移行人 + 到达后切状态"""
        elapsed = step_count - self._phase_start
        if elapsed >= duration:
            self._set_ped_y(end_y)
            self._phase = next_phase
            self._phase_start = step_count
        else:
            t = elapsed / duration
            self._set_ped_y(start_y + (end_y - start_y) * t)

    def _set_ped_y(self, y):
        loc = self._ped.get_location()
        loc.y = y
        self._ped.set_transform(carla.Transform(loc, carla.Rotation(yaw=self._ped_yaw)))

    # ================================================================
    # 手动模式
    # ================================================================
    def _control_loop(self):
        phase, phase_start = "idle", 0
        ped_start_y = None
        ego_braked = False

        for tick in range(int(60 * 20)):
            if not self._running: break

            self.ego_vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=0.8, steer=0.0)
                if ego_braked else carla.VehicleControl(throttle=0.25, steer=0.0))
            self.world.tick()

            # 触发
            if tick >= self._trigger_step and phase == "idle" and self._ped:
                phase, phase_start = "first_half", tick
                ped_start_y = self._ped.get_location().y
                ego_braked = True           # 自车发现行人，立即急刹

            # 前半程 → 车头前暂停
            if phase == "first_half" and self._ped:
                elapsed = tick - phase_start
                if elapsed >= self._half_cross_time:
                    self._set_ped_y(self._lane_y)
                    phase, phase_start = "paused", tick
                else:
                    t = elapsed / self._half_cross_time
                    self._set_ped_y(ped_start_y + (self._lane_y - ped_start_y) * t)

            # 停留 1s
            elif phase == "paused":
                if tick - phase_start >= self._pause_time:
                    phase, phase_start = "second_half", tick
                    ped_start_y = self._ped.get_location().y

            # 后半程 → 对侧
            elif phase == "second_half" and self._ped:
                elapsed = tick - phase_start
                if elapsed >= self._half_cross_time:
                    self._set_ped_y(self._target_y)
                    phase = "done"
                else:
                    t = elapsed / self._half_cross_time
                    self._set_ped_y(ped_start_y + (self._target_y - ped_start_y) * t)

            if tick % 5 == 0: self._record_frame(tick)
            if self.collision_sensor and self.collision_sensor.collided: break

    def cleanup(self):
        self._ped = None
        super().cleanup()
