# 场景4: 前车紧急制动 — 晴天白天 ego55 adv50 2s后adv急刹-8m/s² 距离10m
import carla
from scenarios.base_scenario import BaseScenario
from utils.carla_utils import apply_brake


class EmergencyBrakeScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "emergency_brake"
        self.category = "vehicle_adversarial"
        self.weather = carla.WeatherParameters(
            cloudiness=5.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=0.0, fog_density=0.0, fog_distance=0.0,
            wetness=0.0, sun_azimuth_angle=90.0, sun_altitude_angle=45.0)
        self.ego_speed_ms = 55.0 / 3.6   # 55 km/h
        self.adv_speed_ms = 50.0 / 3.6   # 50 km/h

    def get_env_config(self):
        cfg = super().get_env_config()
        cfg["action_space"] = 2
        return cfg

    def _spawn_actors(self):
        self._spawn_ego()
        self._spawn_adv_in_front(10.0, self.adv_speed_ms)  # 间距仅 10m
        self.world.tick()

    def _spawn_scenario_actors_impl(self):
        self._brake_done = False         # 每 episode 重置
        self._spawn_adv_in_front(10.0, self.adv_speed_ms)

    def step_callback(self, step_count):
        """RL: 2s 后前车急刹"""
        if step_count >= 40 and not self._brake_done:    # 40步 = 2s @20fps
            apply_brake(self.adv_vehicle, 1.0)
            self._brake_done = True

    def _control_loop(self):
        brake_tick = int(2.0 * 20); done_brake = False
        for tick in range(int(60 * 20)):
            if not self._running: break
            if done_brake:
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.9, steer=0.0))
            else:
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
            self.world.tick()
            if tick >= brake_tick and not done_brake:
                apply_brake(self.adv_vehicle, 1.0); done_brake = True
            if tick % 5 == 0: self._record_frame(tick)
            if self.collision_sensor and self.collision_sensor.collided: break
            if done_brake and self.adv_vehicle:
                v = self.adv_vehicle.get_velocity()
                if (v.x**2+v.y**2)**0.5 < 0.1 and tick > brake_tick + 60: break
