# ======================== 导入模块 ========================
import os
import csv
import sys
import time
import carla
import gymnasium as gym
import networkx as nx
import numpy as np
import random
import threading
import torch
import gc
from heapq import nsmallest
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QPushButton, QTextEdit, QHBoxLayout, QGroupBox, QProgressBar,
    QSpinBox, QDoubleSpinBox, QFileDialog, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt6.QtGui import QFont
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ======================== 核心导航系统 ========================
ACTION_DICT = {
    0: (0.0, 0.0),  # 停止
    1: (0.0, 1.0),  # 直行
    2: (25.0, 0.8),  # 左转
    3: (-25.0, 0.8),  # 右转
    4: (0.0, 2.0)  # 奔跑
}

# ======================== 环境初始化模块 ========================
def reset_environment(env):
    try:
        env.close()
        env.reset()
    except Exception as e:
        print(f"重置环境错误: {str(e)}")

class EnhancedPedestrianEnv(gym.Env):
    def __init__(self, start_index=0, end_index=1, target_location=None, enable_camera_follow=True, log_callback=None):
        super().__init__()
        # ======================== Carla连接配置 ========================
        self.log = log_callback if log_callback else print
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(30.0)
        self._connect_to_server()

        # ======================== 获取地图生成点 ========================
        all_spawn_points = self.client.get_world().get_map().get_spawn_points()
        self.spawn_points = all_spawn_points
        self.valid_spawn_mask = []

        walker_bp = self.client.get_world().get_blueprint_library().filter("walker.pedestrian.*")[0]

        for i, spawn in enumerate(all_spawn_points):
            actor = self.client.get_world().try_spawn_actor(walker_bp, spawn)
            if actor:
                self.valid_spawn_mask.append(True)
                actor.destroy()
            else:
                self.valid_spawn_mask.append(False)

        valid_count = sum(self.valid_spawn_mask)
        print(f"✅ 共检测地图 spawn 点 {len(self.spawn_points)} 个，其中可用点数: {valid_count}")

        # ======================== 设置起点和目标点位置 ========================
        self.start_location = self.spawn_points[start_index].location
        self.target_location = (
            self.spawn_points[end_index].location if target_location is None else target_location
        )

        # ======================== 状态变量初始化 ========================
        self.trace_points = []
        self.planned_waypoints = []
        self.pedestrian = None
        self.controller = None
        self.current_road_id = None
        self.path_deviation = 0.0
        self.path_radius = 2.0
        self.stagnant_steps = 0
        self.last_location = carla.Location()
        self.last_reward = 0.0
        self.previous_speed = 0.0
        self.current_speed = 0.0
        self.collision_occurred = False
        self.min_obstacle_distance = 5.0
        self.previous_target_distance = 0.0
        self.episode_step = 0
        self.sensors = []
        self.target_actor = None
        self.cleanup_lock = threading.Lock()
        self.enable_camera_follow = enable_camera_follow

        # ======================== 预加载资源和设置视角 ========================
        self._preload_assets()
        self._setup_spectator(follow=self.enable_camera_follow)

        # ======================== 定义动作与观察空间 ========================
        self.action_space = spaces.Discrete(len(ACTION_DICT))
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    # ======================== Carla服务器连接 ========================
    def _connect_to_server(self):
        for retry in range(5):
            try:
                self.world = self.client.load_world("Town01")
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.02
                self.world.apply_settings(settings)
                if "Town01" in self.world.get_map().name:
                    self.log("status", f"✅ 成功加载Town01地图 (Carla v{self.client.get_server_version()})")
                    return
            except Exception as e:
                self.log("error", f"🔌 连接失败（尝试 {retry + 1}/5）：{str(e)}")
                time.sleep(2)
        raise ConnectionError("无法连接到Carla服务器")

    # ======================== 资源预加载 ========================
    def _preload_assets(self):
        self.blueprint_library = self.world.get_blueprint_library()
        self.walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        self.controller_bp = self.blueprint_library.find('controller.ai.walker')
        self.vehicle_bps = self.blueprint_library.filter('vehicle.*')
        self.lidar_bp = self._configure_lidar()
        self.collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.target_marker_bp = self.blueprint_library.find('static.prop.streetbarrier')

    # ======================== 传感器配置 ========================
    def _configure_lidar(self):
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '10.0')
        lidar_bp.set_attribute('points_per_second', '10000')
        return lidar_bp

    # ======================== 观察视角控制 ========================
    def _setup_spectator(self, follow=True):
        if not follow:
            return
        try:
            self.spectator = self.world.get_spectator()
        except Exception as e:
            self.log("error", f"🎥 获取观测视角失败: {str(e)}")

    def _update_spectator_view(self):
        try:
            if not hasattr(self, "spectator") or not self.spectator:
                return
            if not self.pedestrian or not self.pedestrian.is_alive:
                return
            ped_loc = self.pedestrian.get_transform().location
            self.spectator.set_transform(carla.Transform(
                carla.Location(x=ped_loc.x, y=ped_loc.y, z=20),
                carla.Rotation(pitch=-90)
            ))
        except Exception as e:
            print(f"视角更新失败: {str(e)}")

    # ======================== 路径可视化 ========================
    def _draw_planned_waypoints(self):
        try:
            arrow_interval = 2.0  # 每隔2米绘制一个箭头
            for i in range(len(self.planned_waypoints) - 1):
                wp1 = self.get_location_from_wp(self.planned_waypoints[i])
                wp2 = self.get_location_from_wp(self.planned_waypoints[i + 1])

                vec = wp2 - wp1
                dist = vec.length()
                direction = vec.make_unit_vector()

                num_arrows = int(dist // arrow_interval)
                for j in range(num_arrows):
                    start = wp1 + direction * (j * arrow_interval)
                    end = wp1 + direction * ((j + 1) * arrow_interval)
                    self.world.debug.draw_arrow(
                        start + carla.Location(z=0.5),
                        end + carla.Location(z=0.5),
                        thickness=0.1,
                        arrow_size=0.2,
                        color=carla.Color(255, 0, 0),
                        life_time=10.0,
                        persistent_lines=False
                    )
        except Exception as e:
            self.log("error", f"❌ 路径绘制失败: {str(e)}")

    def _draw_trace_points(self):
        try:
            if len(self.trace_points) < 1:
                return
            for point in self.trace_points:
                loc = point + carla.Location(z=0.3)
                self.world.debug.draw_point(
                    loc,
                    size=0.1,
                    color=carla.Color(0, 255, 0),
                    life_time=3.0,
                    persistent_lines=False
                )
        except Exception as e:
            self.log("error", f"📍 轨迹绘制失败: {str(e)}")

    # ======================== 初始点与目标点生成 ========================
    def _spawn_target_marker(self):
        if self.target_actor and self.target_actor.is_alive:
            self.target_actor.destroy()
        self.target_actor = self.world.spawn_actor(
            self.target_marker_bp,
            carla.Transform(self.target_location, carla.Rotation())
        )
        # 起点标记
        self.world.debug.draw_string(
            self.start_location + carla.Location(z=1.5),
            "Start",
            draw_shadow=False,
            color=carla.Color(0, 255, 0),
            life_time=0.0,
            persistent_lines=True
        )
        # 终点标记
        self.world.debug.draw_string(
            self.target_location + carla.Location(z=1.5),
            "Goal",
            draw_shadow=False,
            color=carla.Color(255, 0, 0),
            life_time=0.0,
            persistent_lines=True
        )
    # ======================== 环境重置 ========================
    def reset(self, **kwargs):
        with self.cleanup_lock:
            self._cleanup_actors()
            time.sleep(0.5)
            if self.controller and self.controller.is_alive:
                self.controller.stop()
            self._spawn_pedestrian()
            self._attach_sensors()
            self._spawn_target_marker()
            self.trace_points.clear()
            if self.pedestrian and self.pedestrian.is_alive:
                self._update_spectator_view()
            else:
                self.log("error", "❌ 重置失败：行人未生成")
            self.episode_step = 0
            self.collision_occurred = False
            self.last_reward = 0.0
            self.previous_speed = 0.0
            self.current_speed = 0.0
            self.min_obstacle_distance = 5.0
            self.previous_target_distance = 0.0
            self.planned_waypoints = self._generate_path(self.start_location, self.target_location)
            if self.controller and self.pedestrian and self.controller.is_alive:
                if len(self.planned_waypoints) > 1:
                    goal = self.get_location_from_wp(self.planned_waypoints[-1])
                    self.controller.go_to_location(goal)
                    self.log("status", f"👣 控制器导航目标已设置: {goal}")
            self.stagnant_steps = 0
            self.last_location = self.pedestrian.get_location()
            return self._get_obs(), {}

    # ======================== 行人生成与控制 ========================
    def _spawn_pedestrian(self):
        for i in range(5):  # 多尝试几次
            try:
                spawn_point = carla.Transform(
                    self.start_location,
                    carla.Rotation(yaw=random.randint(0, 360))
                )
                blueprint = random.choice(self.walker_bps)
                self.pedestrian = self.world.try_spawn_actor(blueprint, spawn_point)
                if self.pedestrian is not None:
                    break
            except Exception as e:
                self.log("error", f"🚶 第 {i + 1} 次行人生成失败: {str(e)}")
                time.sleep(0.5)

        if self.pedestrian is None:
            raise RuntimeError(f"行人对象未正确初始化，起点位置可能非法（{self.start_location}）")

        # 生成控制器
        self.controller = self.world.spawn_actor(
            self.controller_bp,
            carla.Transform(),
            attach_to=self.pedestrian,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.controller.start()

    # ======================== 传感器附加 ========================
    def _attach_sensors(self):
        try:
            collision_sensor = self.world.spawn_actor(
                self.collision_bp,
                carla.Transform(),
                attach_to=self.pedestrian
            )
            collision_sensor.listen(lambda e: self._on_collision(e))
            lidar = self.world.spawn_actor(
                self.lidar_bp,
                carla.Transform(carla.Location(z=2.5)),
                attach_to=self.pedestrian
            )
            lidar.listen(lambda d: self._process_lidar(d))
            self.sensors = [collision_sensor, lidar]
        except Exception as e:
            self.log("error", f"📡 传感器初始化失败: {str(e)}")
            self._cleanup_actors()
            raise

    # ======================== 碰撞处理 ========================
    def _on_collision(self, event):
        self.collision_occurred = True

    # ======================== 激光雷达处理 ========================
    def _process_lidar(self, data):
        try:
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
            with self.cleanup_lock:
                if len(points) > 0 and hasattr(self, 'min_obstacle_distance'):
                    distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
                    self.min_obstacle_distance = np.min(distances)
                else:
                    self.min_obstacle_distance = 5.0
        except Exception as e:
            print(f"激光雷达处理错误: {str(e)}")

    # ======================== 观测数据获取 ========================
    def _get_obs(self):
        try:
            transform = self.pedestrian.get_transform()
            current_loc = transform.location
            current_rot = transform.rotation
            target_vector = self.target_location - current_loc
            target_distance = target_vector.length()
            target_dir = target_vector.make_unit_vector() if target_distance > 0 else carla.Vector3D()
            yaw = np.radians(current_rot.yaw)
            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            local_target = rotation_matrix @ np.array([target_dir.x, target_dir.y, target_dir.z])
            if len(self.planned_waypoints) > 0:
                next_wp = self.planned_waypoints[0]
                next_wp_vector = next_wp.transform.location - current_loc
                local_next_wp = rotation_matrix @ np.array([next_wp_vector.x, next_wp_vector.y, next_wp_vector.z])
            else:
                local_next_wp = np.array([0, 0, 0])
            return np.array([
                current_loc.x / 200 - 1,
                current_loc.y / 200 - 1,
                local_target[0],
                local_target[1],
                np.clip(self.min_obstacle_distance / 5, 0, 1),
                self.current_speed / 3,
                target_distance / 100,
                self.path_deviation / 5.0,
                1.0 if self._is_on_sidewalk() else 0.0,
                yaw / 360.0,
                local_next_wp[0],
                local_next_wp[1]
            ], dtype=np.float32)
        except Exception as e:
            self.log("error", f"👁️ 观测获取失败: {str(e)}")
            return np.zeros(self.observation_space.shape)

    # ======================== 路径规划 ========================
    def get_location_from_wp(self, wp):
        """从 waypoint 或 (waypoint, option) 中提取位置"""
        if isinstance(wp, tuple):
            wp = wp[0]
        return wp.transform.location

    def _build_nav_graph_from_csv(self, csv_path="walkable_points_Town01.csv", k=3):
        G = nx.Graph()
        points = []

        if not os.path.exists(csv_path):
            self.log("error", f"❌ 缺少导航点文件：{csv_path}")
            return G

        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["index"])
                loc = carla.Location(x=float(row["x"]), y=float(row["y"]), z=float(row["z"]))
                points.append((idx, loc))
                G.add_node(loc, index=idx)

        # k近邻连边
        for i in range(len(points)):
            current_idx, current_loc = points[i]
            others = [points[j][1] for j in range(len(points)) if j != i]
            nearest = nsmallest(k, others, key=lambda p: current_loc.distance(p))
            for neighbor in nearest:
                dist = current_loc.distance(neighbor)
                G.add_edge(current_loc, neighbor, weight=dist)

        self.log("info", f"🌐 图初步构建完毕：{len(G.nodes)} 节点，{len(G.edges)} 边")

        # 保留最大连通块
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()

        valid_ids = [G.nodes[n]["index"] for n in G.nodes]
        self.valid_nav_indices = set(valid_ids)  # 👈 缓存用于后续起点终点检查

        self.log("info", f"🧩 最大连通块节点数: {len(G.nodes)}，合法编号范围: {min(valid_ids)} ~ {max(valid_ids)}")
        return G

    def _find_a_star_path_from_locations(self, graph, start_loc, end_loc):
        def closest(node_list, loc):
            if not node_list:
                raise ValueError("图结构中无可用节点")
            return min(node_list, key=lambda p: p.distance(loc))

        node_list = list(graph.nodes)
        start_node = closest(node_list, start_loc)
        end_node = closest(node_list, end_loc)

        try:
            path = nx.astar_path(
                graph,
                source=start_node,
                target=end_node,
                heuristic=lambda a, b: a.distance(b),
                weight="weight"
            )
            return path
        except Exception as e:
            self.log("error", f"❌ A* 搜索失败: {str(e)}")
            return []

    def get_point_index(self, loc):
        """根据Location获取在导航图中对应的编号 index"""
        if not hasattr(self, "nav_graph"):
            return -1
        for node in self.nav_graph.nodes:
            if loc.distance(node) < 0.5:  # 容差半径
                return self.nav_graph.nodes[node]["index"]
        return -1

    def _generate_path(self, start_location, end_location):
        try:
            # 如果没有图就构建一次
            if not hasattr(self, 'nav_graph'):
                self.nav_graph = self._build_nav_graph_from_csv()

            # 合法性检查：编号必须在最大连通图中
            if not hasattr(self, "valid_nav_indices"):
                self.log("error", "❌ 未检测到合法编号集合")
                return []

            # index 映射
            start_index = self.get_point_index(start_location)
            end_index = self.get_point_index(end_location)

            if start_index not in self.valid_nav_indices or end_index not in self.valid_nav_indices:
                self.log("error", f"❌ 起点或终点编号不在最大连通图中，请更换编号！")
                return []

            # 路径查找（A*）
            path = self._find_a_star_path_from_locations(self.nav_graph, start_location, end_location)

            if not path or len(path) < 2:
                raise ValueError("路径搜索失败，建议更换起点或终点")

            self.log("status",
                     f"✅ 路径规划成功，路径点数: {len(path)}，直线距离: {start_location.distance(end_location):.1f} m")

            # 返回 Dummy Waypoint 列表，兼容原路径处理逻辑
            return [type("DummyWP", (), {"transform": carla.Transform(loc)}) for loc in path]

        except Exception as e:
            self.log("error", f"❌ A* 路径生成出错: {str(e)}")
            return []

    # ======================== 路径状态更新 ========================
    def _update_path_status(self):
        if not self.planned_waypoints:
            return
        try:
            current_loc = self.pedestrian.get_location()
            nearest_wp = min(
                self.planned_waypoints,
                key=lambda wp: wp.transform.location.distance(current_loc)
            )
            wp_transform = nearest_wp.transform
            current_vector = current_loc - wp_transform.location
            forward_vector = wp_transform.get_forward_vector()
            cross_product = current_vector.cross(forward_vector)
            self.path_deviation = abs(cross_product.length()) / forward_vector.length()
            self.current_road_id = nearest_wp.road_id
        except Exception as e:
            self.log("error", f"🧭 路径状态更新失败: {str(e)}")

    # ======================== 人行道检测 ========================
    def _is_on_sidewalk(self):
        try:
            current_wp = self.world.get_map().get_waypoint(
                self.pedestrian.get_location(),
                project_to_road=True
            )
            return current_wp.lane_type == carla.LaneType.Sidewalk
        except:
            return False

    # ======================== 步进执行 ========================
    def step(self, action_idx):
        try:
            current_transform = self.pedestrian.get_transform()
            current_location = current_transform.location
            current_yaw = current_transform.rotation.yaw

            # == 立即停止判断 ==
            if current_location.distance(self.target_location) < 1.5:
                self.controller.stop()
                self.pedestrian.apply_control(carla.WalkerControl())  # 停止移动
                self.log("status", f"✅ 行人已到达目标，位置: {current_location}")
                return self._get_obs(), 1000.0, True, False, {}

            # == 解析动作 ==
            if isinstance(action_idx, np.ndarray):
                action_idx = int(action_idx[0]) if action_idx.ndim > 0 else int(action_idx)
            elif isinstance(action_idx, list):
                action_idx = int(action_idx[0])
            else:
                action_idx = int(action_idx)
            yaw_offset, speed_ratio = ACTION_DICT[action_idx]

            # == 导航控制 ==
            target_vector = self.target_location - current_location
            target_dist = target_vector.length()
            target_yaw = np.degrees(np.arctan2(-target_vector.y, target_vector.x))
            yaw_diff = np.arctan2(np.sin(np.radians(target_yaw - current_yaw)),
                                  np.cos(np.radians(target_yaw - current_yaw)))
            yaw_diff = np.degrees(yaw_diff)

            if target_dist < 5.0:
                auto_steer = np.clip(yaw_diff / 15, -1, 1) * 30
            else:
                auto_steer = np.clip(yaw_diff / 30, -1, 1) * 45
            final_yaw = current_yaw + np.clip(yaw_offset * 0.05 + auto_steer, -45, 45)

            self.pedestrian.set_transform(carla.Transform(current_location, carla.Rotation(yaw=final_yaw)))
            self.trace_points.append(self.pedestrian.get_location())
            self._draw_trace_points()

            base_speed = 1.5 + 1.5 * speed_ratio
            safe_speed = min(base_speed, 3) if self.min_obstacle_distance > 2 else 0.8
            self.previous_speed = self.current_speed
            self.current_speed = safe_speed

            yaw_rad = np.radians(final_yaw)
            direction = carla.Vector3D(x=np.cos(yaw_rad), y=np.sin(yaw_rad), z=0)
            control = carla.WalkerControl(direction=direction, speed=safe_speed)
            self.pedestrian.apply_control(control)

            self.world.tick()
            self._update_spectator_view()
            new_obs = self._get_obs()

            # == 奖励系统 ==
            reward = 0.0
            done = False

            if target_dist < 3.0:
                reward += 1000
                done = True
            else:
                progress = self.previous_target_distance - target_dist
                distance_factor = np.clip(1 - (target_dist / 100), 0.1, 1.0)
                reward += progress * 50 * distance_factor

            if self.collision_occurred:
                reward -= 500
                done = True
            else:
                if self.min_obstacle_distance < 2.0:
                    reward -= 0.5 / (self.min_obstacle_distance + 0.5)
                if (self.previous_speed - self.current_speed) > 1.0:
                    reward -= 1.0 * (self.previous_speed - self.current_speed)

            path_follow_bonus = 1.5 * (1 - self.path_deviation / self.path_radius)
            reward += path_follow_bonus if self.path_deviation < self.path_radius else -1.0
            reward -= 0.01

            if target_dist < 5.0:
                if 0.3 <= self.current_speed <= 1.0:
                    reward += 0.2
                elif self.current_speed > 1.0:
                    reward -= 0.2 * (self.current_speed - 1.0)
            else:
                if 0.5 <= self.current_speed <= 1.5:
                    reward += 0.1

            self.previous_target_distance = target_dist

            if not done and target_dist < 2:
                direction_vector = self.target_location - current_location
                yaw_diff = abs(
                    current_transform.rotation.yaw - np.degrees(np.arctan2(-direction_vector.y, direction_vector.x)))
                if yaw_diff < 45:
                    reward += 1000
                    done = True
                    self.log("status", f"🎯 成功到达目标，剩余距离：{target_dist:.2f}m")

            self._draw_planned_waypoints()
            return new_obs, reward, done, False, {}

        except Exception as e:
            self.log("error", f"⚙️ 执行步骤错误: {str(e)}")
            return np.zeros(self.observation_space.shape), 0, True, False, {}

    # ======================== 清理资源 ========================
    def _cleanup_actors(self):
        destroy_list = []
        try:
            for sensor in self.sensors:
                try:
                    if sensor.is_alive:
                        sensor.stop()
                        sensor.destroy()
                        self.log("info", f"🧹 传感器 {sensor.id} 已销毁")
                except Exception as e:
                    self.log("error", f"🔥 传感器销毁失败: {str(e)}")
            self.sensors = []
            if hasattr(self, 'controller') and self.controller is not None:
                try:
                    if self.controller.is_alive:
                        self.controller.stop()
                        time.sleep(0.1)
                        self.controller.destroy()
                        self.log("info", "🧹 控制器已销毁")
                except Exception as e:
                    self.log("error", f"🔥 控制器销毁失败: {str(e)}")
                finally:
                    self.controller = None
            if hasattr(self, 'pedestrian') and self.pedestrian is not None:
                try:
                    if self.pedestrian.is_alive:
                        self.pedestrian.apply_control(carla.WalkerControl())
                        time.sleep(0.1)
                        self.pedestrian.destroy()
                        self.log("info", "🧹 行人已销毁")
                except Exception as e:
                    self.log("error", f"🔥 行人销毁失败: {str(e)}")
                finally:
                    self.pedestrian = None
            if self.target_actor and self.target_actor.is_alive:
                try:
                    self.target_actor.destroy()
                    self.log("info", "🧹 目标标记已销毁")
                except Exception as e:
                    self.log("error", f"🔥 目标标记销毁失败: {str(e)}")
                finally:
                    self.target_actor = None
            for _ in range(10):
                self.world.tick()
                time.sleep(0.1)
            gc.collect()
            self.log("status", "✅ 所有Actor清理完成")
        except Exception as e:
            self.log("error", f"💥 清理过程中发生严重错误: {str(e)}")
        finally:
            self.sensors = []
            self.controller = None
            self.pedestrian = None
            self.target_actor = None

    def close(self):
        self._cleanup_actors()
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        time.sleep(1)


# ======================== 训练封装模块 ========================
class TrainingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_count = 0
        self.model = None
        self.episode_rewards = []

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.episode_rewards.append(reward)
        info.update({
            'current_speed': self.env.current_speed,
            'min_obstacle_distance': self.env.min_obstacle_distance,
            'target_distance': self.env.previous_target_distance
        })
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_count += 1
        if self.episode_count % 50 == 0:
            self.save_checkpoint()
        return self.env.reset(**kwargs)

    def save_checkpoint(self):
        if self.model:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.model.save(f"ped_model_{timestamp}")


# ======================== 演示运行模块 ========================
def run_navigation_demo(model_path, episodes=1, gui_callback=None, start_index=0, end_index=1):
    try:
        env = EnhancedPedestrianEnv(
            start_index=start_index,
            end_index=end_index,
            enable_camera_follow=False,
            log_callback=gui_callback.emit if gui_callback else None
        )
        settings = env.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.02
        env.world.apply_settings(settings)
        model = PPO.load(model_path)
        for episode in range(episodes):
            reset_environment(env)
            obs, _ = env.reset()
            done = False
            step_count = 0
            while not done and step_count < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                current_loc = env.pedestrian.get_transform().location
                target_dist = current_loc.distance(env.target_location)
                if gui_callback:
                    msg = (f"步骤 {step_count}: 位置({current_loc.x:.1f}, {current_loc.y:.1f}) "
                           f"剩余距离: {target_dist:.1f}m 速度: {env.current_speed:.1f}m/s")
                    gui_callback.emit("log", msg)
                step_count += 1
                time.sleep(0.05)
        return True
    except Exception as e:
        if gui_callback:
            gui_callback.emit("error", f"演示错误: {str(e)}")
        return False
    finally:
        if env:
            env.close()


# ======================== GUI模块 ========================
class TrainingThread(QThread):
    update_signal = pyqtSignal(str, str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, env_params, train_params):
        super().__init__()
        self.env_params = env_params
        self.train_params = train_params
        self._is_running = True
        self.mutex = QMutex()

    def run(self):
        try:
            self.update_signal.emit("status", "正在初始化训练环境...")
            env = EnhancedPedestrianEnv(**self.env_params)
            reset_environment(env)
            wrapped_env = TrainingWrapper(env)
            vec_env = DummyVecEnv([lambda: wrapped_env])
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=self.train_params["learning_rate"],
                n_steps=self.train_params["n_steps"],
                batch_size=self.train_params["batch_size"],
                gamma=self.train_params["gamma"],
                gae_lambda=self.train_params["gae_lambda"],
                clip_range=self.train_params["clip_range"],
                ent_coef=self.train_params["ent_coef"],
                vf_coef=self.train_params["vf_coef"],
                policy_kwargs=self.train_params["policy_kwargs"],
                verbose=0,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            total_steps = self.train_params["total_steps"]
            self.progress_signal.emit(0)
            for step in range(0, total_steps, self.train_params["n_steps"]):
                self.mutex.lock()
                if not self._is_running:
                    break
                self.mutex.unlock()
                model.learn(self.train_params["n_steps"])
                progress = min(step + self.train_params['n_steps'], total_steps)
                self.update_signal.emit("log",
                                        f"已训练 {progress}/{total_steps} 步 | "
                                        f"平均奖励: {np.mean(wrapped_env.episode_rewards[-10:]) if wrapped_env.episode_rewards else 0:.1f}")
                self.progress_signal.emit(progress)
            model_path = "pedestrian_ppo"
            model.save(model_path)
            self.finished_signal.emit(True, model_path)
        except Exception as e:
            self.finished_signal.emit(False, f"训练失败: {str(e)}")
        finally:
            try:
                vec_env.close()
                env.close()
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    def stop(self):
        self.mutex.lock()
        self._is_running = False
        self.mutex.unlock()

class CarlaPedestrianGUI(QMainWindow):
    def _log_system_info(self):
        version = "Carla Pedestrian Nav v1.4.0"
        features = [
            "1 GUI控制面板",
            "2 支持Carla 0.9.15",
            "3 基于CSV可行点 + A*算法的路径规划",
            "4 行人走过路径用点来可视化",
            "5 到达目标自动停止",
            "6 可配置训练参数增多",
            "7 碰撞惩罚机制",
            "8 轨迹奖励设计",
            "新增： 导航图支持k近邻连边 + 最大连通子图提取",
            "新增： 路径点合法性检测，自动提示路径不可达",
            "新增： 输出附带表情，提升用户体验",
            "新增： 系统可视化界面升级，迁移使用PyQt6库",
        ]
        self.status_label.setText(f"{version} 已加载")
        self.log_area.append(f"[版本] {version}")
        self.log_area.append("[已实现功能列表]:")
        for feat in features:
            self.log_area.append(f"  {feat}")
        self.log_area.append("系统初始化完成。")

    def find_valid_path_pairs(self):
        try:
            filepath = "walkable_points_Town01.csv"
            if not os.path.exists(filepath):
                self.log_message("error", f"❌ 缺少导航点文件：{filepath}")
                return

            self.log_message("status", f"📁 正在加载导航点文件 {filepath}...")
            client = carla.Client("localhost", 2000)
            client.set_timeout(10.0)
            world = client.get_world()

            # === 1. 读取导航点 ===
            with open(filepath, mode='r') as f:
                reader = csv.DictReader(f)
                points = []
                for row in reader:
                    idx = int(row["index"])
                    loc = carla.Location(x=float(row["x"]), y=float(row["y"]), z=float(row["z"]))
                    points.append((idx, loc))

            self.log_message("info", f"🧠 共加载可行点: {len(points)}")

            # === 2. 构建图（k近邻） ===
            k = 6
            G = nx.Graph()
            for _, loc in points:
                G.add_node(loc)

            for i in range(len(points)):
                current_idx, current_loc = points[i]
                other_locs = [points[j][1] for j in range(len(points)) if j != i]
                closest_neighbors = nsmallest(k, other_locs, key=lambda p: current_loc.distance(p))
                for neighbor in closest_neighbors:
                    dist = current_loc.distance(neighbor)
                    G.add_edge(current_loc, neighbor, weight=dist)

            self.log_message("info", f"🌐 图构建完成: {len(G.nodes)} 节点，{len(G.edges)} 边")

            # === 3. 连通性分析 ===
            components = list(nx.connected_components(G))
            largest_size = max(len(c) for c in components)
            self.log_message("info", f"🧩 图中连通区域数: {len(components)}，最大连通块节点数: {largest_size}")

            # === 4. A* 可达性判断 + 阈值过滤 ===
            min_path_length = 20
            max_path_length = 200
            valid_pairs = []

            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    try:
                        path = nx.astar_path(
                            G,
                            source=points[i][1],
                            target=points[j][1],
                            heuristic=lambda a, b: a.distance(b),
                            weight="weight"
                        )
                        if min_path_length <= len(path) <= max_path_length:
                            valid_pairs.append((points[i][0], points[j][0], len(path)))
                            self.log_area.append(
                                f"✅ 可用对: 起点{points[i][0]} → 终点{points[j][0]}，路径点数: {len(path)}")
                    except:
                        continue

            self.log_message("status",
                             f"检测完成，找到 {len(valid_pairs)} 对合法路径（路径点数≥{min_path_length} 且 ≤{max_path_length}）")

        except Exception as e:
            self.log_message("error", f"路径检测失败: {str(e)}")

    def clear_debug_markers(self):
        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(5.0)
            world = client.get_world()
            world.debug.clear()
            self.log_message("status", "所有可视化标记已清除")
        except Exception as e:
            self.log_message("error", f"清除失败: {str(e)}")

    def show_walkable_spawn_points(self):
        try:
            filepath = "walkable_points_Town01.csv"
            client = carla.Client("localhost", 2000)
            client.set_timeout(5.0)
            world = client.get_world()

            if not os.path.exists(filepath):
                self.log_message("error", f"❌ 未找到缓存文件：{filepath}，请先初始化环境生成可行点")
                return

            self.log_message("status", f"📁 已找到缓存文件 {filepath}，正在加载...")
            with open(filepath, mode='r') as f:
                reader = csv.DictReader(f)
                points = list(reader)

            self.log_message("status", f"✅ 可生成行人位置总数：{len(points)}")

            for row in points:
                idx = int(row["index"])
                x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
                loc = carla.Location(x=x, y=y, z=z)
                label = f"P{idx}"

                world.debug.draw_string(
                    loc,
                    label,
                    draw_shadow=False,
                    color=carla.Color(0, 255, 0),
                    life_time=60.0,
                    persistent_lines=True
                )

                self.log_area.append(f"{label}: ({x:.1f}, {y:.1f}, {z:.1f})")

        except Exception as e:
            self.log_message("error", f"获取可用行人点失败: {str(e)}")

    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.demo_thread = None
        self.current_model = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Carla 行人导航系统")
        self.setGeometry(200, 200, 1000, 800)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # === 系统控制区 ===
        control_group = QGroupBox("系统控制")
        control_layout = QHBoxLayout()
        self.camera_follow_checkbox = QCheckBox("跟随摄像头视角")
        self.camera_follow_checkbox.setChecked(True)
        self.btn_init = QPushButton("初始化环境")
        self.btn_train = QPushButton("开始训练")
        self.btn_demo = QPushButton("运行演示")
        self.btn_stop = QPushButton("终止进程")
        self.btn_load = QPushButton("加载模型")
        control_layout.addWidget(self.camera_follow_checkbox)
        control_layout.addWidget(self.btn_init)
        control_layout.addWidget(self.btn_train)
        control_layout.addWidget(self.btn_demo)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_load)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # === 自定义起点与终点区 ===
        location_group = QGroupBox("自定义起点与终点")
        location_layout = QHBoxLayout()
        self.start_idx_spin = QSpinBox()
        self.start_idx_spin.setPrefix("起点编号 ")
        self.start_idx_spin.setValue(0)
        self.end_idx_spin = QSpinBox()
        self.end_idx_spin.setPrefix("终点编号 ")
        self.end_idx_spin.setValue(1)
        location_layout.addWidget(self.start_idx_spin)
        location_layout.addWidget(self.end_idx_spin)
        location_group.setLayout(location_layout)
        main_layout.addWidget(location_group)

        # === 可视化工具区 ===
        visual_group = QGroupBox("可视化工具")
        visual_layout = QHBoxLayout()
        self.btn_show_walkable = QPushButton("显示可生成行人位置")
        self.btn_show_walkable.clicked.connect(self.show_walkable_spawn_points)
        self.btn_clear_debug = QPushButton("清除可视化标记")
        self.btn_clear_debug.clicked.connect(self.clear_debug_markers)
        self.btn_check_paths = QPushButton("检测路径合法性")
        self.btn_check_paths.clicked.connect(lambda: self.find_valid_path_pairs())
        visual_layout.addWidget(self.btn_show_walkable)
        visual_layout.addWidget(self.btn_clear_debug)
        visual_layout.addWidget(self.btn_check_paths)
        visual_group.setLayout(visual_layout)
        main_layout.addWidget(visual_group)

        # === 初始化训练参数控件 ===
        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setRange(64, 16384)
        self.n_steps_spin.setValue(4096)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setDecimals(3)
        self.gamma_spin.setSingleStep(0.001)
        self.gamma_spin.setRange(0.8, 0.999)
        self.gamma_spin.setValue(0.990)

        self.lam_spin = QDoubleSpinBox()
        self.lam_spin.setDecimals(3)
        self.lam_spin.setSingleStep(0.001)
        self.lam_spin.setRange(0.8, 1.0)
        self.lam_spin.setValue(0.95)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setRange(1e-5, 1e-2)
        self.lr_spin.setValue(0.0003)

        self.entropy_coef_spin = QDoubleSpinBox()
        self.entropy_coef_spin.setDecimals(3)
        self.entropy_coef_spin.setSingleStep(0.001)
        self.entropy_coef_spin.setRange(0.0, 0.05)
        self.entropy_coef_spin.setValue(0.01)

        self.clip_range_spin = QDoubleSpinBox()
        self.clip_range_spin.setDecimals(2)
        self.clip_range_spin.setSingleStep(0.01)
        self.clip_range_spin.setRange(0.1, 0.5)
        self.clip_range_spin.setValue(0.20)

        self.vf_coef_spin = QDoubleSpinBox()
        self.vf_coef_spin.setDecimals(2)
        self.vf_coef_spin.setSingleStep(0.01)
        self.vf_coef_spin.setRange(0.0, 1.0)
        self.vf_coef_spin.setValue(0.50)

        self.total_timesteps_spin = QSpinBox()
        self.total_timesteps_spin.setRange(10000, 5000000)
        self.total_timesteps_spin.setValue(100000)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(32, 2048)
        self.batch_size_spin.setValue(256)

        # === 训练参数区 ===
        training_param_group = QGroupBox("训练参数")
        training_param_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("每次训练步数:"))
        row1.addWidget(self.n_steps_spin)
        row1.addWidget(QLabel("折扣因子 γ:"))
        row1.addWidget(self.gamma_spin)
        row1.addWidget(QLabel("优势估计 λ:"))
        row1.addWidget(self.lam_spin)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("学习率 α:"))
        row2.addWidget(self.lr_spin)
        row2.addWidget(QLabel("熵系数 Entropy:"))
        row2.addWidget(self.entropy_coef_spin)
        row2.addWidget(QLabel("截断范围 Clip:"))
        row2.addWidget(self.clip_range_spin)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("值函数权重 Vf:"))
        row3.addWidget(self.vf_coef_spin)
        row3.addWidget(QLabel("总训练步数:"))
        row3.addWidget(self.total_timesteps_spin)
        row3.addWidget(QLabel("批大小:"))
        row3.addWidget(self.batch_size_spin)

        training_param_layout.addLayout(row1)
        training_param_layout.addLayout(row2)
        training_param_layout.addLayout(row3)
        training_param_group.setLayout(training_param_layout)
        main_layout.addWidget(training_param_group)

        # === 状态显示区 ===
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 10))
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("font-weight: bold; color: #444;")
        status_layout.addWidget(self.log_area)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        # === 信号连接区 ===
        self.btn_init.clicked.connect(self.init_environment)
        self.btn_train.clicked.connect(self.start_training)
        self.btn_demo.clicked.connect(self.start_demo)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_load.clicked.connect(self.load_model)
        self.toggle_controls(True)

        # === 系统信息输出 ===
        self._log_system_info()

    def toggle_controls(self, ready):
        self.btn_init.setEnabled(ready)
        self.btn_train.setEnabled(ready and self.current_model is None)
        self.btn_demo.setEnabled(ready and self.current_model is not None)
        self.btn_load.setEnabled(ready)
        self.btn_stop.setEnabled(not ready)

    def log_message(self, msg_type, message):
        emoji = {
            "error": "❌",
            "status": "✅",
            "info": "📝"
        }.get(msg_type, "🔔")

        if msg_type == "error":
            self.log_area.append(f'<span style="color: red;">{emoji} [ERROR] {message}</span>')
            QMessageBox.critical(self, "错误", message)
        elif msg_type == "status":
            self.status_label.setText(f"{emoji} {message}")
            self.log_area.append(f'{emoji} [STATUS] {message}')
        else:
            self.log_area.append(f'{emoji} [INFO] {message}')

    def init_environment(self):
        try:
            self.log_message("status", "正在连接Carla服务器并检测可用行人位置...")
            temp_env = EnhancedPedestrianEnv(
                enable_camera_follow=self.camera_follow_checkbox.isChecked(),
                log_callback=self.log_message
            )
            total_spawn_count = len(temp_env.spawn_points)
            valid_spawn_count = sum(temp_env.valid_spawn_mask)
            temp_env.close()

            # === 使用缓存中最大 index 设置 SpinBox 范围 ===
            filepath = "walkable_points_Town01.csv"
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    reader = csv.DictReader(f)
                    indices = [int(row["index"]) for row in reader]
                    if indices:
                        max_idx = max(indices)
                        self.start_idx_spin.setRange(0, max_idx)
                        self.end_idx_spin.setRange(0, max_idx)
            else:
                # 如果没缓存，就默认用全部点数设置
                self.start_idx_spin.setRange(0, total_spawn_count - 1)
                self.end_idx_spin.setRange(0, total_spawn_count - 1)

            self.log_message("status",f"环境初始化成功：共 {total_spawn_count} 个生成点，其中可用点数: {valid_spawn_count}")
            QMessageBox.information(self, "成功",f"Carla连接成功！共 {total_spawn_count} 个生成点，其中 {valid_spawn_count} 可用于路径。")

        except Exception as e:
            self.log_message("error", f"连接失败: {str(e)}")

    def start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            return

        train_params = {
            "learning_rate": self.lr_spin.value(),
            "n_steps": self.n_steps_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "total_steps": self.total_timesteps_spin.value(),
            "gamma": self.gamma_spin.value(),
            "gae_lambda": self.lam_spin.value(),
            "clip_range": self.clip_range_spin.value(),
            "ent_coef": self.entropy_coef_spin.value(),
            "vf_coef": self.vf_coef_spin.value(),
            "start_index": self.start_idx_spin.value(),
            "end_index": self.end_idx_spin.value(),
            "policy_kwargs": {
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "activation_fn": torch.nn.ReLU,
                "ortho_init": True
            }
        }

        env_params = {
            "start_index": self.start_idx_spin.value(),
            "end_index": self.end_idx_spin.value(),
            "enable_camera_follow": self.camera_follow_checkbox.isChecked(),
            "log_callback": self.log_message
        }

        self.training_thread = TrainingThread(env_params, train_params)
        self.training_thread.update_signal.connect(self.log_message)
        self.training_thread.progress_signal.connect(lambda v: self.progress_bar.setValue(v))
        self.training_thread.finished_signal.connect(self.training_finished)
        self.progress_bar.setRange(0, train_params["total_steps"])
        self.toggle_controls(False)
        self.training_thread.start()

    def training_finished(self, success, message):
        self.toggle_controls(True)
        if success:
            self.current_model = message
            self.log_message("status", f"训练完成！模型路径: {message}")
        else:
            self.log_message("error", message)

    def start_demo(self):
        if not self.current_model:
            QMessageBox.warning(self, "警告", "请先加载或训练模型！")
            return

        start_idx = self.start_idx_spin.value()
        end_idx = self.end_idx_spin.value()
        def demo_run():
            run_navigation_demo(
                self.current_model,
                gui_callback=self.log_message,
                start_index=start_idx,
                end_index=end_idx
            )
        self.demo_thread = QThread()
        self.demo_thread.run = demo_run
        self.demo_thread.finished.connect(lambda: self.toggle_controls(True))
        self.toggle_controls(False)
        self.demo_thread.start()

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "ZIP Files (*.zip)")
        if path:
            try:
                PPO.load(path)
                self.current_model = path
                self.log_message("status", f"模型加载成功: {path}")
            except Exception as e:
                self.log_message("error", f"加载失败: {str(e)}")

    def stop_all(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.training_thread.quit()
            self.log_message("status", "训练已终止")
        if self.demo_thread and self.demo_thread.isRunning():
            self.demo_thread.quit()
            self.log_message("status", "演示已停止")
        self.toggle_controls(True)

    def closeEvent(self, event):
        self.stop_all()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarlaPedestrianGUI()
    window.show()
    sys.exit(app.exec())