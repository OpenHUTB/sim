# 场景2: 浓雾巡航 — 雾100% 能见度20m 自车低速跟车 前车偶有减速
import carla
from scenarios.base_scenario import BaseScenario
from utils.carla_utils import apply_brake


class HeavyFogScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "heavy_fog"
        self.category = "extreme_weather"
        self.weather = carla.WeatherParameters(
            cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=10.0, fog_density=100.0, fog_distance=20.0,
            wetness=0.0, sun_azimuth_angle=90.0, sun_altitude_angle=55.0)
        self.ego_speed_ms = 28.0 / 3.6
        self.adv_speed_ms = 25.0 / 3.6

    def get_env_config(self):
        cfg = super().get_env_config()
        cfg["action_space"] = 2
        cfg["brake_mode"] = "coast"        # 松油门滑行，永不刹停
        return cfg

    def _spawn_actors(self):
        self._spawn_ego()
        self._spawn_adv_in_front(30.0, self.adv_speed_ms)
        self.world.tick()

    def _spawn_scenario_actors_impl(self):
        self._spawn_adv_in_front(30.0, self.adv_speed_ms)

    def step_callback(self, step_count):
        """每 4 秒前车轻微减速，测试雾天跟车反应"""
        if self.adv_vehicle is None:
            return
        cycle = step_count % 80          # 4s 周期
        if cycle == 0:
            fwd = self.adv_vehicle.get_transform().get_forward_vector()
            self.adv_vehicle.set_target_velocity(carla.Vector3D(
                float(fwd.x * 22.0 / 3.6), float(fwd.y * 22.0 / 3.6), 0.0))
        elif cycle == 25:
            fwd = self.adv_vehicle.get_transform().get_forward_vector()
            self.adv_vehicle.set_target_velocity(carla.Vector3D(
                float(fwd.x * self.adv_speed_ms),
                float(fwd.y * self.adv_speed_ms), 0.0))

    def _control_loop(self):
        for tick in range(int(60 * 20)):
            if not self._running: break
            fwd = self.ego_vehicle.get_transform().get_forward_vector()
            self.ego_vehicle.set_target_velocity(carla.Vector3D(
                float(fwd.x * self.ego_speed_ms), float(fwd.y * self.ego_speed_ms), 0.0))
            self.world.tick()
            if tick % 80 == 0 and self.adv_vehicle:
                apply_brake(self.adv_vehicle, 0.12)
            if tick % 10 == 0: self._record_frame(tick)
            if self.collision_sensor and self.collision_sensor.collided: break
