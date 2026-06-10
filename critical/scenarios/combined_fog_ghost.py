# 场景10: 雾天鬼探头 — 浓雾90% 货车相邻车道, 行人车头盲区冲出, 最高危
import carla
from scenarios.base_scenario import BaseScenario
from utils.carla_utils import spawn_pedestrian_at


class FogGhostScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "combined_fog_ghost"
        self.category = "multi_factor_coupled"
        self.weather = carla.WeatherParameters(
            cloudiness=90.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=10.0, fog_density=90.0, fog_distance=15.0,
            wetness=20.0, sun_azimuth_angle=90.0, sun_altitude_angle=45.0)
        self.ego_speed_ms = 25.0 / 3.6
        self._trigger_step = int(4.0 * 20)       # 4s 行人冲出
        self._cross_time = int(2.0 * 20)          # 横穿 2s
        self._pause_time = int(1.0 * 20)          # 车头前停 1s

    def get_env_config(self):
        cfg = super().get_env_config()
        cfg["action_space"] = 2
        return cfg

    def _spawn_actors(self):
        self._spawn_ego()
        self._spawn_obstacle_and_pedestrian()
        self.world.tick()

    def _spawn_scenario_actors_impl(self):
        self._phase = "idle"
        self._phase_start = 0
        self._ped_start_y = None
        self._spawn_obstacle_and_pedestrian()

    def _spawn_obstacle_and_pedestrian(self):
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        bp_lib = self.world.get_blueprint_library()

        # 货车在相邻车道，自车前方 32m（同鬼探头场景）
        truck_dist = 32.0
        truck_x = ego_loc.x + fwd.x * truck_dist + right.x * 3.5
        truck_y = ego_loc.y + fwd.y * truck_dist + right.y * 3.5
        truck_bp = bp_lib.filter("vehicle.carlamotors.carlacola")[0] \
            if bp_lib.filter("vehicle.carlamotors.carlacola") else bp_lib.filter("vehicle.*")[2]
        truck_sp = carla.Transform(
            carla.Location(x=truck_x, y=truck_y, z=ego_loc.z + 0.5), ego_tf.rotation)
        obs = self.world.try_spawn_actor(truck_bp, truck_sp)
        if obs is not None:
            obs.apply_control(carla.VehicleControl(brake=1.0))
            self.npc_vehicles.append(obs)

        # 行人生成在货车车头前方 3m（雾+盲区双重隐蔽）
        ped_x = ego_loc.x + fwd.x * (truck_dist + 3.0)
        ped_y = ego_loc.y + right.y * 5.0
        self._ped_yaw = ego_tf.rotation.yaw - 90.0

        ped_loc = carla.Location(x=ped_x, y=ped_y, z=ego_loc.z + 0.5)
        ped, ctrl = spawn_pedestrian_at(self.world, ped_loc, speed_ms=0.0)
        ped.set_transform(carla.Transform(ped_loc, carla.Rotation(yaw=self._ped_yaw)))
        self.pedestrians.append((ped, ctrl))
        self._ped = ped
        self._start_y = ped_y
        self._lane_y = ego_loc.y
        self._target_y = ego_loc.y - right.y * 5.0

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
                carla.VehicleControl(throttle=0.0, brake=0.9, steer=0.0)
                if ego_braked else carla.VehicleControl(throttle=0.15, steer=0.0))
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
        self._ped = None; super().cleanup()
