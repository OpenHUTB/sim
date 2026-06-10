# 场景8: 行人闯红灯 — 晴天白天 ego40 行人从前方红绿灯旁人行道闯红灯横穿
import carla
from scenarios.base_scenario import BaseScenario
from utils.carla_utils import spawn_pedestrian_at


class JaywalkingScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "jaywalking"
        self.category = "pedestrian_danger"
        self.weather = carla.WeatherParameters(
            cloudiness=5.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=0.0, fog_density=0.0, fog_distance=0.0,
            wetness=0.0, sun_azimuth_angle=90.0, sun_altitude_angle=45.0)
        self.ego_speed_ms = 30.0 / 3.6
        self._trigger_step = int(1.0 * 20)   # 1s 即急刹，确保红灯前停下
        self._cross_time = int(2.0 * 20)
        self._pause_time = int(1.0 * 20)

    def get_env_config(self):
        cfg = super().get_env_config()
        cfg["action_space"] = 2
        return cfg

    def _spawn_actors(self):
        self._spawn_ego()
        self._spawn_pedestrian()
        self.world.tick()

    def _spawn_scenario_actors_impl(self):
        self._phase = "idle"
        self._phase_start = 0
        self._ped_start_y = None
        self._spawn_pedestrian()

    def _spawn_pedestrian(self):
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        # 行人生成在自车前方 30m 右侧红绿灯旁人行道上
        ped_x = ego_loc.x + fwd.x * 30 + right.x * 6.0
        ped_y = ego_loc.y + fwd.y * 30 + right.y * 6.0
        self._ped_yaw = ego_tf.rotation.yaw - 90.0  # 从右侧面向车道

        ped_loc = carla.Location(x=ped_x, y=ped_y, z=ego_loc.z + 0.5)
        ped, ctrl = spawn_pedestrian_at(self.world, ped_loc, speed_ms=0.0)
        ped.set_transform(carla.Transform(ped_loc, carla.Rotation(yaw=self._ped_yaw)))
        self.pedestrians.append((ped, ctrl))
        self._ped = ped
        self._start_y = ped_y                              # 左侧人行道
        self._lane_y = ego_loc.y                            # 自车车道（停留点）
        self._target_y = ego_loc.y - right.y * 6.0         # 对侧（左侧）人行道

    # ================================================================
    # RL 回调
    # ================================================================
    def step_callback(self, step_count):
        if self._ped is None: return
        if step_count >= self._trigger_step and self._phase == "idle":
            self._phase = "first_half"; self._phase_start = step_count
            self._ped_start_y = self._ped.get_location().y
        if self._phase == "first_half":
            self._move_ped(step_count, self._ped_start_y, self._lane_y, self._cross_time, "paused")
        elif self._phase == "paused":
            if step_count - self._phase_start >= self._pause_time:
                self._phase = "second_half"; self._phase_start = step_count
                self._ped_start_y = self._ped.get_location().y
        elif self._phase == "second_half":
            self._move_ped(step_count, self._ped_start_y, self._target_y, self._cross_time, "done")

    def _move_ped(self, step_count, start_y, end_y, duration, next_phase):
        elapsed = step_count - self._phase_start
        if elapsed >= duration:
            self._set_ped_y(end_y); self._phase = next_phase; self._phase_start = step_count
        else:
            self._set_ped_y(start_y + (end_y - start_y) * (elapsed / duration))

    def _set_ped_y(self, y):
        loc = self._ped.get_location(); loc.y = y
        self._ped.set_transform(carla.Transform(loc, carla.Rotation(yaw=self._ped_yaw)))

    # ================================================================
    # 手动模式
    # ================================================================
    def _control_loop(self):
        phase, phase_start = "idle", 0
        ped_start_y = None; ego_braked = False
        for tick in range(int(60 * 20)):
            if not self._running: break
            self.ego_vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=0.8, steer=0.0)
                if ego_braked else carla.VehicleControl(throttle=0.3, steer=0.0))
            self.world.tick()
            if tick >= self._trigger_step and phase == "idle" and self._ped:
                phase, phase_start = "first_half", tick
                ped_start_y = self._ped.get_location().y; ego_braked = True
            if phase == "first_half" and self._ped:
                e = tick - phase_start
                if e >= self._cross_time: self._set_ped_y(self._lane_y); phase, phase_start = "paused", tick
                else: self._set_ped_y(ped_start_y + (self._lane_y - ped_start_y) * (e / self._cross_time))
            elif phase == "paused":
                if tick - phase_start >= self._pause_time:
                    phase, phase_start = "second_half", tick; ped_start_y = self._ped.get_location().y
            elif phase == "second_half" and self._ped:
                e = tick - phase_start
                if e >= self._cross_time: self._set_ped_y(self._target_y); phase = "done"
                else: self._set_ped_y(ped_start_y + (self._target_y - ped_start_y) * (e / self._cross_time))
            if tick % 5 == 0: self._record_frame(tick)
            if self.collision_sensor and self.collision_sensor.collided: break

    def cleanup(self):
        self._ped = None
        super().cleanup()
