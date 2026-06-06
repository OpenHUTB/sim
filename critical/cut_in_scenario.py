# 场景5: 车辆强行加塞 — 旁车同行后直接位移切入自车车道
import carla
from scenarios.base_scenario import BaseScenario


class CutInScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "cut_in"
        self.category = "vehicle_adversarial"
        self.weather = carla.WeatherParameters(
            cloudiness=5.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=0.0, fog_density=0.0, fog_distance=0.0,
            wetness=0.0, sun_azimuth_angle=90.0, sun_altitude_angle=45.0)
        self.ego_speed_ms = 50.0 / 3.6
        self.adv_speed_ms = 55.0 / 3.6
        self._cut_trigger_step = int(4.0 * 20)      # RL: 4s 后切入
        self._cut_duration_steps = int(1.5 * 20)     # 1.5s 完成切入
        self._cut_triggered = False
        self._cut_start_step = 0
        self._adv_start_y = None

    def get_env_config(self):
        cfg = super().get_env_config()
        cfg["action_space"] = 2
        return cfg

    def _spawn_actors(self):
        self._spawn_ego()
        self._spawn_adv_side()
        self.world.tick()

    def _spawn_scenario_actors_impl(self):
        self._cut_triggered = False
        self._cut_start_step = 0
        self._adv_start_y = None
        self._spawn_adv_side()

    def _spawn_adv_side(self):
        self.world.tick()
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        bp_lib = self.world.get_blueprint_library()
        adv_bp = bp_lib.find("vehicle.audi.a2") or bp_lib.filter("vehicle.*")[1]

        # 旁车在右侧车道，自车后方 15m
        adv_loc = carla.Location(
            x=ego_loc.x - fwd.x * 15 + right.x * 3.5,
            y=ego_loc.y - fwd.y * 15 + right.y * 3.5,
            z=ego_loc.z + 0.5)
        adv_sp = carla.Transform(adv_loc, ego_tf.rotation)
        self.adv_vehicle = self.world.try_spawn_actor(adv_bp, adv_sp)
        if self.adv_vehicle is None:
            raise RuntimeError("加塞车辆生成失败")

        # 先纯直行
        self.adv_vehicle.set_target_velocity(carla.Vector3D(
            float(fwd.x * self.adv_speed_ms),
            float(fwd.y * self.adv_speed_ms), 0.0))

        self._adv_start_y = adv_loc.y     # 记录右侧车道 y 坐标
        self._ego_lane_y = ego_loc.y       # 记录自车车道 y 坐标

    # ================================================================
    # RL 模式回调：渐进位移旁车到自车车道
    # ================================================================
    def step_callback(self, step_count):
        if self.adv_vehicle is None:
            return

        if step_count >= self._cut_trigger_step and not self._cut_triggered:
            self._cut_triggered = True
            self._cut_start_step = step_count
            self._adv_start_y = self.adv_vehicle.get_location().y
            self._ego_lane_y = self.ego_vehicle.get_location().y

        if self._cut_triggered:
            elapsed = step_count - self._cut_start_step
            if elapsed <= self._cut_duration_steps:
                # 线性插值：从右车道 y 平滑过渡到自车车道 y
                t = elapsed / self._cut_duration_steps
                target_y = self._adv_start_y + (self._ego_lane_y - self._adv_start_y) * t
                loc = self.adv_vehicle.get_location()
                loc.y = target_y
                self.adv_vehicle.set_location(loc)

    # ================================================================
    # 手动模式
    # ================================================================
    def _control_loop(self):
        cut_start = int(4.0 * 20)   # 4s 后切入
        cut_dur = int(1.5 * 20)
        ego_braked = False
        adv_y0 = None
        ego_y0 = None

        for tick in range(int(60 * 20)):
            if not self._running:
                break

            if ego_braked:
                self.ego_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=0.8, steer=0.0))
            else:
                self.ego_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.4, steer=0.0))

            self.world.tick()

            # 2s 后旁车开始位移切入自车车道
            if tick == cut_start and self.adv_vehicle:
                adv_y0 = self.adv_vehicle.get_location().y
                ego_y0 = self.ego_vehicle.get_location().y

            if adv_y0 is not None and tick < cut_start + cut_dur:
                elapsed = tick - cut_start
                t = elapsed / cut_dur
                target_y = adv_y0 + (ego_y0 - adv_y0) * t
                loc = self.adv_vehicle.get_location()
                loc.y = target_y
                self.adv_vehicle.set_location(loc)
                # 同时打方向产生视觉效果
                self.adv_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.7, steer=-0.6))

            # 切入完成后回正 + 自车急刹
            if adv_y0 is not None and tick >= cut_start + cut_dur and not ego_braked:
                self.adv_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.4, steer=0.0))
                ego_braked = True

            if tick % 5 == 0:
                self._record_frame(tick)
            if self.collision_sensor and self.collision_sensor.collided:
                break
