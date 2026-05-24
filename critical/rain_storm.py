# 场景1: 暴雨跟车 — 雨90% 积水60% 自车匀速跟车 前车偶有减速
import carla
from scenarios.base_scenario import BaseScenario
from utils.carla_utils import apply_brake


class RainStormScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "rain_storm"
        self.category = "extreme_weather"
        self.weather = carla.WeatherParameters(
            cloudiness=95.0, precipitation=90.0, precipitation_deposits=60.0,
            wind_intensity=30.0, fog_density=20.0, fog_distance=100.0,
            wetness=60.0, sun_azimuth_angle=90.0, sun_altitude_angle=45.0)
        self.ego_speed_ms = 45.0 / 3.6
        self.adv_speed_ms = 40.0 / 3.6

    def get_env_config(self):
        cfg = super().get_env_config()
        cfg["action_space"] = 2
        cfg["brake_mode"] = "coast"        # 松油门滑行，永不刹停
        return cfg

    def _spawn_actors(self):
        self._spawn_ego()
        self._spawn_adv_in_front(20.0, self.adv_speed_ms)
        self.world.tick()

    def _spawn_scenario_actors_impl(self):
        self._spawn_adv_in_front(20.0, self.adv_speed_ms)

    # ================================================================
    # RL 回调：前车周期性减速，测试自车跟车反应
    # ================================================================
    def step_callback(self, step_count):
        """每 ~3 秒前车轻微减速一次，自车需感知并调整距离"""
        if self.adv_vehicle is None:
            return
        cycle = step_count % 60          # 3s 周期 (60 步 @20fps)
        if cycle == 0:
            # 前车减速到 35 km/h
            fwd = self.adv_vehicle.get_transform().get_forward_vector()
            self.adv_vehicle.set_target_velocity(carla.Vector3D(
                float(fwd.x * 35.0 / 3.6), float(fwd.y * 35.0 / 3.6), 0.0))
        elif cycle == 20:
            # 恢复到 40 km/h
            fwd = self.adv_vehicle.get_transform().get_forward_vector()
            self.adv_vehicle.set_target_velocity(carla.Vector3D(
                float(fwd.x * self.adv_speed_ms),
                float(fwd.y * self.adv_speed_ms), 0.0))

    # ================================================================
    # 手动模式
    # ================================================================
    def _control_loop(self):
        for tick in range(int(60 * 20)):
            if not self._running: break
            fwd = self.ego_vehicle.get_transform().get_forward_vector()
            self.ego_vehicle.set_target_velocity(carla.Vector3D(
                float(fwd.x * self.ego_speed_ms),
                float(fwd.y * self.ego_speed_ms), 0.0))
            self.world.tick()
            # 前车周期性轻微减速
            if tick % 60 == 0 and self.adv_vehicle:
                apply_brake(self.adv_vehicle, 0.15)
            if tick % 10 == 0: self._record_frame(tick)
            if self.collision_sensor and self.collision_sensor.collided: break
