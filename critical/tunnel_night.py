# 场景3: 夜间黑暗行驶 — 晴天 夜间亮度5% ego40 无其他车辆 无行人
import carla
from scenarios.base_scenario import BaseScenario


class TunnelNightScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "tunnel_night"
        self.category = "extreme_weather"
        self.weather = carla.WeatherParameters(
            cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=0.0, fog_density=0.0, fog_distance=0.0,
            wetness=0.0, sun_azimuth_angle=180.0, sun_altitude_angle=-90.0)
        self.ego_speed_ms = 40.0 / 3.6

    def get_env_config(self):
        cfg = super().get_env_config()
        cfg["action_space"] = 2
        cfg["brake_mode"] = "coast"
        return cfg

    def _control_loop(self):
        for tick in range(int(60 * 20)):
            if not self._running: break
            fwd = self.ego_vehicle.get_transform().get_forward_vector()
            self.ego_vehicle.set_target_velocity(carla.Vector3D(
                float(fwd.x * self.ego_speed_ms), float(fwd.y * self.ego_speed_ms), 0.0))
            self.world.tick()
            if tick % 10 == 0: self._record_frame(tick)
            if self.collision_sensor and self.collision_sensor.collided: break
