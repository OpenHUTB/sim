# 场景8: 行人闯红灯 — 晴天白天 ego40绿灯 行人从红绿灯旁人行道闯红灯横穿
import carla
from scenarios.base_scenario import BaseScenario
from utils.carla_utils import spawn_pedestrian_at, walk_to_location, apply_brake


class JaywalkingScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "jaywalking"
        self.category = "pedestrian_danger"
        self.weather = carla.WeatherParameters(
            cloudiness=5.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=0.0, fog_density=0.0, fog_distance=0.0,
            wetness=0.0, sun_azimuth_angle=90.0, sun_altitude_angle=45.0)
        self.ego_speed_ms = 40.0 / 3.6

    def get_env_config(self):
        cfg = super().get_env_config()
        cfg["action_space"] = 2
        return cfg

    def _spawn_actors(self):
        self._spawn_ego()
        self._spawn_pedestrian()
        self.world.tick()

    def _spawn_scenario_actors_impl(self):
        self._spawn_pedestrian()

    def _spawn_pedestrian(self):
        ego_loc = self.ego_vehicle.get_location()
        # 行人从红绿灯旁左侧人行道闯出
        ped_loc = carla.Location(x=ego_loc.x + 18, y=ego_loc.y + 6.0, z=ego_loc.z)
        ped, ctrl = spawn_pedestrian_at(self.world, ped_loc, speed_ms=1.8)
        # 闯红灯横穿到对面
        walk_to_location(ctrl, self.world,
                         carla.Location(x=ped_loc.x, y=ego_loc.y - 6.0, z=ped_loc.z))
        self.pedestrians.append((ped, ctrl))

    def _control_loop(self):
        braked = False
        for tick in range(int(60 * 20)):
            if not self._running: break
            if braked:
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.8, steer=0.0))
            else:
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
            self.world.tick()
            if tick == int(2.5 * 20):
                braked = True
            if tick % 5 == 0: self._record_frame(tick)
            if self.collision_sensor and self.collision_sensor.collided: break
