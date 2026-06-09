# scenarios/base_scenario.py
# 场景基类：定义场景通用接口，所有差异化逻辑由子类实现
# 按 scenarios/CLAUDE.md 规范 —— 每个子类的天气/速度/行为/危险触发完全不同

import os
import time
import carla

from config.carla_config import (
    CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT,
    EGO_VEHICLE_BLUEPRINT, ADV_VEHICLE_BLUEPRINT,
    SIMULATION_FPS, FIXED_DELTA_SECONDS, DEFAULT_SPAWN_INDEX,
    EPISODE_TIMEOUT,
)
from utils.data_saver import init_result_dir, save_vehicle_log
from utils.sensor_utils import CollisionSensor, DistanceMonitor


class BaseScenario:
    """
    场景基类。

    子类必须定义:
      - name:         场景名称
      - category:     场景类别
      - weather:      carla.WeatherParameters 实例
      - ego_speed_ms: 自车初始速度 (m/s)

    子类可覆盖:
      - _setup_world():    设置天气、同步模式
      - _spawn_actors():   生成场景专属参与者
      - _trigger_danger(): 触发危险事件
      - run():             运行场景
    """

    def __init__(self):
        self.name = "BaseScenario"
        self.category = "unknown"

        # 天气 —— 子类必须覆盖
        self.weather = carla.WeatherParameters.ClearNoon

        # 速度 —— 子类必须覆盖
        self.ego_speed_ms = 0.0  # m/s

        # CARLA 连接
        self.client = None
        self.world = None
        self.map = None

        # Actors
        self.ego_vehicle = None
        self.adv_vehicle = None
        self.npc_vehicles = []
        self.pedestrians = []

        # 传感器
        self.collision_sensor = None
        self.distance_monitor = None

        # 数据
        self.logs = []
        self.result_dir = None
        self._start_time = None
        self._running = False

        # RL 环境引用（延迟创建）
        self._env = None

    # ================================================================
    # 生命周期
    # ================================================================

    def setup(self):
        """连接 CARLA，设置世界，生成所有参与者"""
        self._connect()
        self._setup_world()
        self._spawn_actors()
        self._attach_sensors()
        self._start_time = time.time()
        self.result_dir = init_result_dir(self.name)["root"]

    def run(self):
        """手动运行演示 —— 子类覆盖 _control_loop()"""
        if self.ego_vehicle is None:
            self.setup()
        self._running = True
        try:
            self._control_loop()
        finally:
            self._running = False

    def cleanup(self):
        """清理所有参与者 + 全量世界清理 + 恢复异步模式"""
        self._running = False
        for sensor in [self.collision_sensor]:
            if sensor is not None:
                try: sensor.destroy()
                except Exception: pass
        self.collision_sensor = None
        self.distance_monitor = None
        # 销毁已知 actor
        for actor in self.npc_vehicles + self.pedestrians:
            if isinstance(actor, (list, tuple)):
                for a in actor:
                    if a is not None:
                        try: a.destroy()
                        except Exception: pass
            elif actor is not None:
                try: actor.destroy()
                except Exception: pass
        for v in [self.ego_vehicle, self.adv_vehicle]:
            if v is not None:
                try: v.destroy()
                except Exception: pass
        self.ego_vehicle = None
        self.adv_vehicle = None
        self.npc_vehicles.clear()
        self.pedestrians.clear()
        # 全量清理世界中所有残留 actor
        if self.world is not None:
            try:
                for actor in self.world.get_actors():
                    if actor is not None:
                        try: actor.destroy()
                        except RuntimeError: pass
            except Exception: pass
            # 恢复异步模式
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except Exception: pass
        if self.logs:
            save_vehicle_log(
                os.path.join(self.result_dir, "logs", "%s_trajectory.csv" % self.name),
                self.logs)

    # ================================================================
    # 外部接口（供 experiments 使用）
    # ================================================================

    def get_env_config(self):
        """返回供 CarlaEnv 使用的参数字典"""
        return {
            "weather": self.weather,
            "ego_speed_ms": self.ego_speed_ms,
            "max_steps": 500,
        }

    def create_env(self):
        """创建关联的 CarlaEnv 实例"""
        from env.carla_env import CarlaEnv
        self._env = CarlaEnv(scenario_params=self.get_env_config())
        return self._env

    def spawn_scenario_actors(self, env):
        """
        RL 模式专用：在 env.reset() 之后生成场景专属参与者。
        env 已生成自车，此方法只生成 adv / 行人 / 障碍物等。
        子类可覆盖 _spawn_scenario_actors_impl() 实现差异化。
        """
        self.ego_vehicle = env.ego_vehicle  # 复用 env 的自车引用
        self.world = env.world
        self._spawn_scenario_actors_impl()
        # 让物理引擎和行人AI控制器初始化（关键：否则行人不会动）
        for _ in range(10):
            self.world.tick()
        # 注册给 env
        if self.adv_vehicle is not None:
            env.register_adv_vehicle(self.adv_vehicle)
        if self.pedestrians:
            env.register_pedestrians(self.pedestrians)
        if self.npc_vehicles:
            env.register_npc_vehicles(self.npc_vehicles)

    def _spawn_scenario_actors_impl(self):
        """子类覆盖：生成场景专属参与者（不含自车）"""
        pass

    def step_callback(self, step_count):
        """
        每步回调（RL 训练循环中调用）。
        子类覆盖以在特定时机触发场景事件（如延迟切入、急刹等）。
        """
        pass

    def inject_actors_to_env(self, env):
        """将已生成的参与者注入 RL 环境（手动模式用）"""
        if self.adv_vehicle is not None:
            env.register_adv_vehicle(self.adv_vehicle)
        if self.pedestrians:
            env.register_pedestrians(self.pedestrians)
        if self.npc_vehicles:
            env.register_npc_vehicles(self.npc_vehicles)

    # ================================================================
    # 内部步骤
    # ================================================================

    def _connect(self):
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(CARLA_TIMEOUT)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

    def _setup_world(self):
        """设置天气和同步模式"""
        self.world.set_weather(self.weather)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

    def _spawn_actors(self):
        """子类覆盖：生成该场景的专属参与者"""
        self._spawn_ego()

    def _spawn_ego(self, spawn_index=None):
        """生成自车（带重试回退）"""
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.map.get_spawn_points()
        n = len(spawn_points)
        ego_bp = bp_lib.find(EGO_VEHICLE_BLUEPRINT) or bp_lib.filter("vehicle.*")[0]

        # 尝试默认 spawn point 及附近点
        start_idx = (spawn_index if spawn_index is not None
                     else DEFAULT_SPAWN_INDEX) % n
        for offset in range(n):
            idx = (start_idx + offset) % n
            self.ego_vehicle = self.world.try_spawn_actor(ego_bp, spawn_points[idx])
            if self.ego_vehicle is not None:
                break

        if self.ego_vehicle is None:
            raise RuntimeError(
                "自车生成失败: 所有 %d 个 spawn point 均被占用，请重启 CARLA 或切换地图" % n)

        if self.ego_speed_ms > 0:
            forward = self.ego_vehicle.get_transform().get_forward_vector()
            self.ego_vehicle.set_target_velocity(carla.Vector3D(
                float(forward.x * self.ego_speed_ms),
                float(forward.y * self.ego_speed_ms),
                0.0))

    def _spawn_adv_in_front(self, distance_m, speed_ms=None):
        """在自车前方生成对抗车辆（多层回退保证成功）"""
        self.world.tick()  # 让 CARLA 注册自车位置
        ego_loc = self.ego_vehicle.get_location()
        ego_rot = self.ego_vehicle.get_transform().rotation
        bp_lib = self.world.get_blueprint_library()
        adv_bp = bp_lib.find(ADV_VEHICLE_BLUEPRINT) or bp_lib.filter("vehicle.*")[1]

        # 计算自车前方方向向量
        forward = self.ego_vehicle.get_transform().get_forward_vector()

        # 策略1: 沿自车前方偏移
        for dist in [distance_m, distance_m + 5, distance_m + 10, distance_m + 15]:
            for lat in [0.0, 1.5, -1.5, 3.0, -3.0]:
                sp = carla.Transform(
                    carla.Location(
                        x=ego_loc.x + forward.x * dist + forward.y * lat,
                        y=ego_loc.y + forward.y * dist - forward.x * lat,
                        z=ego_loc.z + 0.5,
                    ),
                    ego_rot,
                )
                self.adv_vehicle = self.world.try_spawn_actor(adv_bp, sp)
                if self.adv_vehicle is not None:
                    break
            if self.adv_vehicle is not None:
                break

        # 策略2: 用地图 spawn points 中离自车前方最近的空闲点
        if self.adv_vehicle is None:
            spawn_points = self.map.get_spawn_points()
            best_sp = None
            best_dist = float("inf")
            for sp in spawn_points:
                d = sp.location.distance(ego_loc)
                dx = sp.location.x - ego_loc.x
                dy = sp.location.y - ego_loc.y
                # 判断在自车前方（点积 >0）
                if forward.x * dx + forward.y * dy > 0 and d > 5:
                    if d < best_dist:
                        # 确认该点可用
                        test = self.world.try_spawn_actor(adv_bp, sp)
                        if test is not None:
                            if best_sp is not None:
                                test.destroy()
                            best_sp = sp
                            best_dist = d
                        # 否则该点被占用，跳过

            if best_sp is not None:
                self.adv_vehicle = self.world.try_spawn_actor(adv_bp, best_sp)

        if self.adv_vehicle is None:
            raise RuntimeError(
                "对抗车辆生成失败: 地图=%s ego=(%.1f,%.1f) distance=%d"
                % (self.map.name, ego_loc.x, ego_loc.y, distance_m))

        if speed_ms is not None:
            fwd = self.adv_vehicle.get_transform().get_forward_vector()
            self.adv_vehicle.set_target_velocity(carla.Vector3D(
                float(fwd.x * speed_ms), float(fwd.y * speed_ms), 0.0))
        return self.adv_vehicle

    def _attach_sensors(self):
        if self.ego_vehicle is not None:
            self.collision_sensor = CollisionSensor(self.ego_vehicle)

    def _control_loop(self):
        """默认手动控制循环"""
        self.ego_vehicle.set_autopilot(True)
        max_ticks = int(EPISODE_TIMEOUT * SIMULATION_FPS)
        for tick in range(max_ticks):
            if not self._running:
                break
            self.world.tick()
            if tick % 10 == 0:
                self._record_frame(tick)
            if self.collision_sensor is not None and self.collision_sensor.collided:
                self._record_frame(tick)
                break

    def _record_frame(self, tick):
        ego_loc = self.ego_vehicle.get_location()
        vel = self.ego_vehicle.get_velocity()
        ego_speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
        self.logs.append([
            round(time.time() - self._start_time, 2),
            round(ego_loc.x, 2), round(ego_loc.y, 2), round(ego_loc.z, 2),
            round(ego_speed * 3.6, 2),
        ])
