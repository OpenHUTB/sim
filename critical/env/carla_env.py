# env/carla_env.py
# 底层仿真环境：搭建仿真底座、通用交互接口、统一奖励规则
# 按 env/CLAUDE.md 规范 —— 不内置任何场景差异化参数与危险行为

import sys
import os
import numpy as np

try:
    import carla
except ImportError:
    carla_root = os.environ.get("CARLA_ROOT", r"D:\hutb\hutb")
    egg_dir = os.path.join(carla_root, "PythonAPI", "carla", "dist")
    egg_file = os.path.join(egg_dir,
        "carla-0.9.16-py3.7-win-amd64.egg" if os.name == "nt"
        else "carla-0.9.16-py3.7-linux-x86_64.egg")
    if os.path.exists(egg_file):
        sys.path.insert(0, egg_dir)
    import carla

from config.carla_config import (
    CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT,
    SYNC_MODE, FIXED_DELTA_SECONDS, SIMULATION_FPS,
    EGO_VEHICLE_BLUEPRINT, DEFAULT_SPAWN_INDEX,
    MAX_EPISODE_STEPS, EPISODE_TIMEOUT,
)
from config.dqn_config import STATE_SIZE, ACTION_SIZE
from env.reward import RewardCalculator
from utils.sensor_utils import CollisionSensor, DistanceMonitor
from utils.geometry_utils import distance_between_vehicles, speed_ms


class CarlaEnv:
    """
    通用底层仿真交互环境。

    只负责:
      1. CARLA 连接与同步模式
      2. 自车生成与基础控制
      3. 传感器挂载与状态采集
      4. 标准化 reset() / step() 接口
      5. 接收场景层注入的自定义参数

    不负责:
      - 天气设置（由场景层注入）
      - 对抗车辆 / 行人生成（由场景层注入）
      - 危险行为触发（由场景层控制）
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, scenario_params=None):
        """
        scenario_params: dict，由场景层传入，可包含:
            - weather: carla.WeatherParameters 实例
            - ego_speed_ms: 自车初始速度 (m/s)
            - max_steps: 最大步数
            - action_space: 动作空间大小（2=仅巡航/制动, 4=全动作, 默认4）
            - brake_strength: 制动强度（0~1, 默认0.8）
            - brake_mode: "brake"=主动制动 / "coast"=松油门滑行（跟车场景用, 不会刹停）
        """
        self.scenario_params = scenario_params or {}
        self._brake_strength = self.scenario_params.get("brake_strength", 0.8)
        self._brake_mode = self.scenario_params.get("brake_mode", "brake")

        # 连接 CARLA
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(CARLA_TIMEOUT)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # 同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = SYNC_MODE
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        # Actors —— 环境只负责自车
        self.ego_vehicle = None
        self.adv_vehicle = None        # 由场景层注入引用
        self.pedestrians = []          # 由场景层注入引用
        self.npc_vehicles = []         # 由场景层注入引用

        # 传感器
        self.collision_sensor = None
        self.distance_monitor = None

        # 状态 / 动作空间（场景可覆盖）
        self.observation_space = STATE_SIZE
        self.action_space = self.scenario_params.get("action_space", ACTION_SIZE)

        # 奖励计算器（通用规则）
        self.reward_calc = RewardCalculator()

        # 渲染
        self._pygame = None
        self._screen = None
        self._clock = None

        # 内部状态
        self._step_count = 0
        self._max_steps = self.scenario_params.get(
            "max_steps", MAX_EPISODE_STEPS)

    # ==================================================================
    # 标准 Gym 接口
    # ==================================================================

    def reset(self):
        """重置环境：清理 → 生成自车 → 应用场景参数 → 返回 observation"""
        self._cleanup_actors()
        self._step_count = 0
        self.reward_calc.reset()

        # 应用场景注入的天气
        weather = self.scenario_params.get("weather")
        if weather is not None:
            self.world.set_weather(weather)

        # 生成自车（环境唯一负责的 actor）
        self._spawn_ego()

        # 挂载传感器
        self._attach_sensors()

        # 稳定物理
        for _ in range(5):
            self.world.tick()

        return self._get_state()

    def step(self, action):
        """执行动作，返回 (state, reward, done, info)"""
        self._step_count += 1
        self._apply_action(action)
        self.world.tick()

        state = self._get_state()
        reward = self.reward_calc.compute(self)
        done = self._is_done()
        info = self._get_info()

        return state, reward, done, info

    def close(self):
        """恢复异步模式，强制销毁世界所有 actor 防止残留"""
        self._cleanup_actors()
        # 额外保险：销毁世界中所有车辆和行人（防止任何泄漏）
        try:
            for actor in self.world.get_actors():
                if actor is not None:
                    try:
                        actor.destroy()
                    except RuntimeError:
                        pass
        except Exception:
            pass
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

    def render(self, mode="human"):
        """可选 Pygame 渲染"""
        if self._pygame is None:
            try:
                import pygame as pg
                self._pygame = pg
                pg.init()
                self._screen = pg.display.set_mode((800, 600))
                pg.display.set_caption("CARLA RL Environment")
                self._clock = pg.time.Clock()
            except ImportError:
                return
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self._pygame.quit()
                self._pygame = None
                return
        self._screen.fill((240, 240, 240))
        font = self._pygame.font.Font(None, 28)
        y = 20
        for line in [
            f"Step: {self._step_count}",
            f"Ego Speed: {self._get_ego_speed() * 3.6:.1f} km/h",
            f"Distance: {self._get_distance():.1f} m",
            f"Reward: {self.reward_calc.cumulative_reward:.2f}",
        ]:
            self._screen.blit(font.render(line, True, (0, 0, 0)), (20, y))
            y += 32
        self._pygame.display.flip()
        if self._clock:
            self._clock.tick(20)

    # ==================================================================
    # 参数注入（由场景层调用）
    # ==================================================================

    def set_scenario_params(self, params):
        """场景层注入自定义参数"""
        self.scenario_params.update(params)
        if "max_steps" in params:
            self._max_steps = params["max_steps"]

    def register_adv_vehicle(self, vehicle):
        """场景层注入对抗车辆引用（供状态采集用）"""
        self.adv_vehicle = vehicle
        if self.ego_vehicle is not None:
            self.distance_monitor = DistanceMonitor(self.ego_vehicle, vehicle)

    def register_pedestrians(self, pedestrian_list):
        """场景层注入行人引用"""
        self.pedestrians = pedestrian_list

    def register_npc_vehicles(self, npc_list):
        """场景层注入 NPC 车辆引用"""
        self.npc_vehicles = npc_list

    # ==================================================================
    # 内部：自车生成
    # ==================================================================

    def _spawn_ego(self):
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("地图无可用生成点")
        n = len(spawn_points)

        ego_bp = bp_lib.find(EGO_VEHICLE_BLUEPRINT)
        if ego_bp is None:
            ego_bp = bp_lib.filter("vehicle.*")[0]

        start_idx = self.scenario_params.get(
            "spawn_index", DEFAULT_SPAWN_INDEX) % n
        for offset in range(n):
            idx = (start_idx + offset) % n
            self.ego_vehicle = self.world.try_spawn_actor(ego_bp, spawn_points[idx])
            if self.ego_vehicle is not None:
                break

        if self.ego_vehicle is None:
            raise RuntimeError(
                "自车生成失败: 所有 %d 个 spawn point 均被占用" % n)

        init_speed = self.scenario_params.get("ego_speed_ms", 0.0)
        if init_speed > 0:
            forward = self.ego_vehicle.get_transform().get_forward_vector()
            self.ego_vehicle.set_target_velocity(carla.Vector3D(
                float(forward.x * init_speed),
                float(forward.y * init_speed),
                0.0))

    def _attach_sensors(self):
        if self.ego_vehicle is not None:
            self.collision_sensor = CollisionSensor(self.ego_vehicle)

    # ==================================================================
    # 动作控制
    # ==================================================================

    def _apply_action(self, action):
        control = carla.VehicleControl()
        brk = self._brake_strength
        if self.action_space == 2:
            if action == 0:
                control.throttle, control.brake, control.steer = 0.4, 0.0, 0.0
            elif action == 1:
                if self._brake_mode == "coast":
                    # 滑行：松油门自然减速，永不刹停
                    control.throttle, control.brake, control.steer = 0.0, 0.0, 0.0
                else:
                    control.throttle, control.brake, control.steer = 0.0, brk, 0.0
            else:
                control.throttle, control.brake, control.steer = 0.4, 0.0, 0.0
        else:
            if action == 0:
                control.throttle, control.brake, control.steer = 0.4, 0.0, 0.0
            elif action == 1:
                control.throttle, control.brake, control.steer = 0.4, 0.0, -0.3
            elif action == 2:
                control.throttle, control.brake, control.steer = 0.4, 0.0, 0.3
            elif action == 3:
                control.throttle, control.brake, control.steer = 0.0, brk, 0.0
            else:
                control.throttle, control.brake, control.steer = 0.4, 0.0, 0.0
        if self.ego_vehicle is not None:
            self.ego_vehicle.apply_control(control)

    # ==================================================================
    # 状态采集
    # ==================================================================

    def _get_state(self):
        """6 维观测: [ego_x, ego_y, adv_x, adv_y, ego_speed, distance]"""
        ego_loc = self.ego_vehicle.get_location()
        ego_s = self._get_ego_speed()

        if self.adv_vehicle is not None:
            adv_loc = self.adv_vehicle.get_location()
            dist = distance_between_vehicles(self.ego_vehicle, self.adv_vehicle)
            return np.array([ego_loc.x, ego_loc.y, adv_loc.x, adv_loc.y,
                             ego_s, dist], dtype=np.float32)
        return np.array([ego_loc.x, ego_loc.y, 0.0, 0.0,
                         ego_s, 100.0], dtype=np.float32)

    def _get_ego_speed(self):
        return speed_ms(self.ego_vehicle)

    def _get_distance(self):
        if self.adv_vehicle is not None:
            return distance_between_vehicles(self.ego_vehicle, self.adv_vehicle)
        return 100.0

    def _get_rel_speed(self):
        ego_s = self._get_ego_speed()
        if self.adv_vehicle is not None:
            return ego_s - speed_ms(self.adv_vehicle)
        return ego_s

    # ==================================================================
    # 终止判定
    # ==================================================================

    def _is_done(self):
        if self.collision_sensor is not None and self.collision_sensor.collided:
            return True
        if self._step_count >= self._max_steps:
            return True
        return False

    def _get_info(self):
        info = {
            "step": self._step_count,
            "distance": self._get_distance(),
            "ego_speed_ms": self._get_ego_speed(),
            "ego_speed_kmh": self._get_ego_speed() * 3.6,
            "collided": self.collision_sensor.collided if self.collision_sensor else False,
            "collision_intensity": (
                self.collision_sensor.collision_intensity if self.collision_sensor else 0.0),
        }
        if self.adv_vehicle is not None:
            info["adv_speed_kmh"] = speed_ms(self.adv_vehicle) * 3.6
        return info

    # ==================================================================
    # 清理
    # ==================================================================

    def _cleanup_actors(self):
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        self.distance_monitor = None
        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.destroy()
            except RuntimeError:
                pass
        self.ego_vehicle = None
        # 销毁 adv / pedestrians / npc 引用对应的 CARLA actor
        if self.adv_vehicle is not None:
            try:
                self.adv_vehicle.destroy()
            except RuntimeError:
                pass
        self.adv_vehicle = None
        for ped, ctrl in self.pedestrians:
            for a in (ctrl, ped):
                if a is not None:
                    try:
                        a.destroy()
                    except RuntimeError:
                        pass
        self.pedestrians.clear()
        for v in self.npc_vehicles:
            if v is not None:
                try:
                    v.destroy()
                except RuntimeError:
                    pass
        self.npc_vehicles.clear()
        self._step_count = 0
