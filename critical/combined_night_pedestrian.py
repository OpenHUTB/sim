# 场景9: 夜间行人横穿（耦合） — 晴天夜间亮度5% ego30 行人从人行道黑暗中横穿
import carla
from scenarios.base_scenario import BaseScenario
from utils.carla_utils import spawn_pedestrian_at, walk_to_location, apply_brake


class NightPedestrianScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "combined_night_pedestrian"
        self.category = "multi_factor_coupled"
        self.weather = carla.WeatherParameters(
            cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=0.0, fog_density=0.0, fog_distance=0.0,
            wetness=0.0, sun_azimuth_angle=180.0, sun_altitude_angle=-90.0)
        self.ego_speed_ms = 30.0 / 3.6

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
        # 行人从右侧车旁人行道出现（黑暗中无灯光）
        ped_loc = carla.Location(x=ego_loc.x + 20, y=ego_loc.y - 6.0, z=ego_loc.z)
        ped, ctrl = spawn_pedestrian_at(self.world, ped_loc, speed_ms=1.5)
        walk_to_location(ctrl, self.world,
                         carla.Location(x=ped_loc.x, y=ego_loc.y + 6.0, z=ped_loc.z))
        self.pedestrians.append((ped, ctrl))

    def _control_loop(self):
        braked = False
        for tick in range(int(60 * 20)):
            if not self._running: break
            if braked:
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.8, steer=0.0))
            else:
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.25, steer=0.0))
            self.world.tick()
            if tick == int(3.0 * 20):
                braked = True
            if tick % 5 == 0: self._record_frame(tick)
            if self.collision_sensor and self.collision_sensor.collided: break
