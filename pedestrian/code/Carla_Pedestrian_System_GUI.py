import sys
import time
import carla
import gymnasium as gym
import numpy as np
import random
import threading
import torch
import gc
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QLineEdit, QGroupBox,
                             QSpinBox, QDoubleSpinBox, QFileDialog, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QFont
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

def reset_environment(env):
    try:
        env.close()
        env.reset()
    except Exception as e:
        print(f"重置环境错误: {str(e)}")

class EnhancedPedestrianEnv(gym.Env):
    def __init__(self, target_location=carla.Location(x=202, y=65, z=0)):
        super().__init__()
        # 初始化关键属性
        self.planned_waypoints = []
        self.pedestrian = None
        self.controller = None
        self.current_road_id = None
        self.path_deviation = 0.0
        self.path_radius = 2.0
        self.stagnant_steps = 0
        self.last_location = carla.Location()
        self.target_location = target_location
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
        # Carla连接配置
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(30.0)
        self._connect_to_server()
        # 空间定义
        self.action_space = spaces.Discrete(len(ACTION_DICT))
        # 观测空间定义（12个特征）
        self.observation_space = spaces.Box(
            low=np.array([
                -1.0,
                -1.0,  # current_loc.y /200 -1
                -1.0,  # local_target.x
                -1.0,  # local_target.y
                0.0,  # min_obstacle_distance /5
                0.0,  # current_speed /3
                0.0,  # target_distance /100
                0.0,  # path_deviation /5
                0.0,  # is_on_sidewalk
                0.0,  # yaw /360
                -1.0,  # next_wp.x
                -1.0  # next_wp.y
            ], dtype=np.float32),
            high=np.array([
                1.0,  # x坐标范围
                1.0,  # y坐标
                1.0,  # local_target.x
                1.0,  # local_target.y
                1.0,  # obstacle_distance
                1.0,  # speed
                3.0,  # target_distance（支持300米）
                1.0,  # path_deviation
                1.0,  # sidewalk状态
                1.0,  # yaw角度
                1.0,  # next_wp.x
                1.0  # next_wp.y
            ], dtype=np.float32),
            dtype=np.float32
        )
        # 初始化组件
        self._preload_assets()
        self._setup_spectator()

    def _connect_to_server(self):
        """连接Carla服务器"""
        for retry in range(5):
            try:
                self.world = self.client.load_world("Town01")
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)

                if "Town01" in self.world.get_map().name:
                    print(f"成功加载Town01地图 (Carla v{self.client.get_server_version()})")
                    return
            except Exception as e:
                print(f"连接失败（尝试 {retry + 1}/5）：{str(e)}")
                time.sleep(2)
        raise ConnectionError("无法连接到Carla服务器")

    def _preload_assets(self):
        """预加载蓝图资产"""
        self.blueprint_library = self.world.get_blueprint_library()
        self.walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        self.controller_bp = self.blueprint_library.find('controller.ai.walker')
        self.vehicle_bps = self.blueprint_library.filter('vehicle.*')
        self.lidar_bp = self._configure_lidar()
        self.collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.target_marker_bp = self.blueprint_library.find('static.prop.streetbarrier')

    def _configure_lidar(self):
        """配置激光雷达"""
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '10.0')
        lidar_bp.set_attribute('points_per_second', '10000')
        return lidar_bp

    def _setup_spectator(self):
        """初始化观察视角"""
        self.spectator = self.world.get_spectator()
        self._update_spectator_view()

    def _update_spectator_view(self):
        """更新俯视视角"""
        try:
            # 空值检查
            if self.pedestrian is not None and self.pedestrian.is_alive:
                ped_loc = self.pedestrian.get_transform().location
                self.spectator.set_transform(carla.Transform(
                    carla.Location(x=ped_loc.x, y=ped_loc.y, z=20),
                    carla.Rotation(pitch=-90)
                ))
            else:
                print("警告：行人未生成或已销毁，无法更新视角")
        except Exception as e:
            print(f"视角更新失败: {str(e)}")

    def _spawn_target_marker(self):
        """生成目标点标记"""
        if self.target_actor and self.target_actor.is_alive:
            self.target_actor.destroy()
        self.target_actor = self.world.spawn_actor(
            self.target_marker_bp,
            carla.Transform(self.target_location, carla.Rotation())
        )

    def reset(self, **kwargs):
        """重置环境"""
        with self.cleanup_lock:
            self._cleanup_actors()
            time.sleep(0.5)

            # 显式停止控制器
            if self.controller and self.controller.is_alive:
                self.controller.stop()

            # 生成新实例
            self._spawn_pedestrian()
            self._attach_sensors()
            self._spawn_target_marker()

            # 确保行人存在后再更新视角
            if self.pedestrian and self.pedestrian.is_alive:
                self._update_spectator_view()
            else:
                print("重置失败：行人未生成")

            # 重置状态变量
            self.episode_step = 0
            self.collision_occurred = False
            self.last_reward = 0.0
            self.previous_speed = 0.0
            self.current_speed = 0.0
            self.min_obstacle_distance = 5.0
            self.previous_target_distance = 0.0
            self._generate_path()  # 再生成路径（依赖行人位置）
            self.stagnant_steps = 0
            self.last_location = self.pedestrian.get_location()

            return self._get_obs(), {}

    def _spawn_pedestrian(self):
        """生成受控行人"""
        for _ in range(3):
            try:
                # 设置行人生成位置
                spawn_point = carla.Transform(
                    carla.Location(x=160, y=138, z=1.0),
                    carla.Rotation(yaw=random.randint(0, 360))
                )
                self.pedestrian = self.world.spawn_actor(
                    random.choice(self.walker_bps),
                    spawn_point
                )
                break
            except Exception as e:
                print(f"行人生成失败: {str(e)}")
                time.sleep(0.5)
                # 清理残留对象
                if self.pedestrian and self.pedestrian.is_alive:
                    self.pedestrian.destroy()
            else:
                raise RuntimeError("无法生成行人，请检查Carla服务器状态")

                # 添加控制器前检查行人对象
            if self.pedestrian is None:
                raise RuntimeError("行人对象未正确初始化")

        # 添加控制器
        self.controller = self.world.spawn_actor(
            self.controller_bp,
            carla.Transform(),
            attach_to=self.pedestrian,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.controller.start()

    def _attach_sensors(self):
        """附加传感器"""
        try:
            # 碰撞传感器
            collision_sensor = self.world.spawn_actor(
                self.collision_bp,
                carla.Transform(),
                attach_to=self.pedestrian
            )
            collision_sensor.listen(lambda e: self._on_collision(e))

            # 激光雷达
            lidar = self.world.spawn_actor(
                self.lidar_bp,
                carla.Transform(carla.Location(z=2.5)),
                attach_to=self.pedestrian
            )
            lidar.listen(lambda d: self._process_lidar(d))

            self.sensors = [collision_sensor, lidar]
        except Exception as e:
            print(f"传感器初始化失败: {str(e)}")
            self._cleanup_actors()
            raise

    def _on_collision(self, event):
        """碰撞处理"""
        self.collision_occurred = True

    def _process_lidar(self, data):
        """处理激光雷达数据"""
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

    def _get_obs(self):
        """获取观测数据"""
        try:
            transform = self.pedestrian.get_transform()
            current_loc = transform.location
            current_rot = transform.rotation

            # 计算目标方向
            target_vector = self.target_location - current_loc
            target_distance = target_vector.length()
            target_dir = target_vector.make_unit_vector() if target_distance > 0 else carla.Vector3D()

            # 转换到局部坐标系
            yaw = np.radians(current_rot.yaw)
            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            local_target = rotation_matrix @ np.array([target_dir.x, target_dir.y, target_dir.z])

            # 获取下一个路径点的方向
            if len(self.planned_waypoints) > 0:
                next_wp = self.planned_waypoints[0]
                next_wp_vector = next_wp.transform.location - current_loc
                local_next_wp = rotation_matrix @ np.array([next_wp_vector.x, next_wp_vector.y, next_wp_vector.z])
            else:
                local_next_wp = np.array([0, 0, 0])

            # 更新观测值（新增两个维度）
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
                local_next_wp[0],  # 新增：下一个路径点的局部x方向
                local_next_wp[1]  # 新增：下一个路径点的局部y方向
            ], dtype=np.float32)
        except Exception as e:
            print(f"观测获取失败: {str(e)}")
            return np.zeros(self.observation_space.shape)

    def _generate_path(self):
        """生成从当前位置到目标的路径（使用手动路径点连接）"""
        try:
            if self.pedestrian is None or not self.pedestrian.is_alive:
                raise RuntimeError("行人未生成或已销毁")

            carla_map = self.world.get_map()
            start_loc = self.pedestrian.get_location()
            end_loc = self.target_location

            # 获取起点和终点的最近道路点
            start_wp = carla_map.get_waypoint(start_loc, project_to_road=True)
            end_wp = carla_map.get_waypoint(end_loc, project_to_road=True)

            # 手动生成路径（替代失效的generate_waypoints）
            self.planned_waypoints = []
            current_wp = start_wp
            max_steps = 500  # 防止无限循环

            while current_wp and max_steps > 0:
                self.planned_waypoints.append(current_wp)
                if current_wp.transform.location.distance(end_loc) < 10.0:
                    self.planned_waypoints.append(end_wp)
                    break

                # 选择下一个路径点（优先朝向终点方向）
                next_wps = current_wp.next(1.0)
                if not next_wps:
                    break

                # 计算到终点的方向向量
                direction_to_target = (end_wp.transform.location - current_wp.transform.location).make_unit_vector()

                # 选择方向最接近的路径点
                current_wp = max(next_wps,
                                 key=lambda wp: wp.transform.get_forward_vector().dot(direction_to_target))

                max_steps -= 1

            # 添加路径平滑处理（减少突变）
            if len(self.planned_waypoints) > 2:
                self.planned_waypoints = [wp for i, wp in enumerate(self.planned_waypoints) if i % 2 == 0]

            # 添加终点路径点
            self.planned_waypoints.append(end_wp)

            # 可视化路径（红色圆点）
            for wp in self.planned_waypoints:
                self.world.debug.draw_string(
                    wp.transform.location + carla.Location(z=0.5),
                    '•',
                    life_time=100.0,
                    color=carla.Color(255, 0, 0)
                )
            print(
                f"生成路径点数量: {len(self.planned_waypoints)}，起点到终点距离: {start_loc.distance(end_loc):.1f}m")

        except Exception as e:
            print(f"路径生成失败: {str(e)}")
            # 回退到直线路径
            self.planned_waypoints = [start_wp, end_wp] if start_wp and end_wp else []

    def _update_path_status(self):
        """更新路径偏离状态"""
        if not self.planned_waypoints:
            return

        try:
            current_loc = self.pedestrian.get_location()

            # 查找最近路径点
            nearest_wp = min(
                self.planned_waypoints,
                key=lambda wp: wp.transform.location.distance(current_loc)
            )

            # 计算横向偏离
            wp_transform = nearest_wp.transform
            current_vector = current_loc - wp_transform.location
            forward_vector = wp_transform.get_forward_vector()

            # 横向偏离 = |当前向量 × 前进方向| / 前进方向长度
            cross_product = current_vector.cross(forward_vector)
            self.path_deviation = abs(cross_product.length()) / forward_vector.length()

            # 更新道路信息
            self.current_road_id = nearest_wp.road_id

        except Exception as e:
            print(f"路径状态更新失败: {str(e)}")

    def _is_on_sidewalk(self):
        """检测是否在人行道上"""
        try:
            current_wp = self.world.get_map().get_waypoint(
                self.pedestrian.get_location(),
                project_to_road=True
            )
            return current_wp.lane_type == carla.LaneType.Sidewalk
        except:
            return False

    def step(self, action_idx):
        """执行动作"""
        try:
            # 获取行人当前状态
            current_transform = self.pedestrian.get_transform()
            current_location = current_transform.location
            current_yaw = current_transform.rotation.yaw

            # 解析动作
            if isinstance(action_idx, (np.ndarray, list)):
                action_idx = int(action_idx[0])
            else:
                action_idx = int(action_idx)
            yaw_offset, speed_ratio = ACTION_DICT[action_idx]

            # 计算目标向量
            target_vector = self.target_location - current_location
            target_dist = target_vector.length()
            target_yaw = np.degrees(np.arctan2(-target_vector.y, target_vector.x))

            # 转向控制
            yaw_diff = np.arctan2(np.sin(np.radians(target_yaw - current_yaw)),
                                  np.cos(np.radians(target_yaw - current_yaw)))
            yaw_diff = np.degrees(yaw_diff)

            # 转向控制部分
            if target_dist < 5.0:
                # 近距离时降低转向幅度
                auto_steer = np.clip(yaw_diff / 15, -1, 1) * 30
            else:
                auto_steer = np.clip(yaw_diff / 30, -1, 1) * 45

            # 限制最大转向角度
            final_yaw = current_yaw + np.clip(yaw_offset * 0.05 + auto_steer, -45, 45)
            self.pedestrian.set_transform(carla.Transform(
                current_location,
                carla.Rotation(yaw=final_yaw)
            ))

            # 速度控制
            base_speed = 1.5 + 1.5 * speed_ratio
            safe_speed = min(base_speed, 3) if self.min_obstacle_distance > 2 else 0.8
            self.previous_speed = self.current_speed
            self.current_speed = safe_speed
            control = carla.WalkerControl(direction=carla.Vector3D(1, 0, 0), speed=safe_speed)
            self.pedestrian.apply_control(control)
            self.world.tick()
            self._update_spectator_view()

            # 获取新观测数据
            new_obs = self._get_obs()

            # ===== 完整奖励计算 =====
            reward = 0.0

            # 1. 核心目标奖励（提高近距离奖励系数）
            if target_dist < 3.0:
                reward += 1000
                done = True
            else:
                progress = self.previous_target_distance - target_dist
                # 动态奖励系数：距离越近，奖励权重越高
                distance_factor = np.clip(1 - (target_dist / 100), 0.1, 1.0)
                reward += progress * 50 * distance_factor  # 系数从30提升至50，并加入距离因子

            # 2. 安全惩罚优化（减少近距离惩罚强度）
            if self.collision_occurred:
                reward -= 500
            else:
                if self.min_obstacle_distance < 2.0:
                    # 调整公式，避免过大的负值
                    reward -= 0.5 / (self.min_obstacle_distance + 0.5)  # 原为1.5/(d+0.1)

                if (self.previous_speed - self.current_speed) > 1.0:
                    reward -= 1.0 * (self.previous_speed - self.current_speed)  # 惩罚减半

            # 3. 路径相关奖励（放宽路径偏离容忍）
            path_follow_bonus = 1.5 * (1 - self.path_deviation / self.path_radius)  # 系数从1.0提升至1.5
            reward += path_follow_bonus if self.path_deviation < self.path_radius else -1.0  # 惩罚从-3.0减至-1.0

            # 4. 时间效率惩罚（减少固定惩罚）
            reward -= 0.01  # 从0.02降低到0.01

            # 5. 速度合规优化（允许接近目标时减速）
            if target_dist < 5.0:  # 距离5米内时，速度合规范围调整
                if 0.3 <= self.current_speed <= 1.0:
                    reward += 0.2
                elif self.current_speed > 1.0:
                    reward -= 0.2 * (self.current_speed - 1.0)
            else:
                if 0.5 <= self.current_speed <= 1.5:
                    reward += 0.1

            # 更新状态变量
            self.previous_target_distance = target_dist

            # 终止条件，允许更早结束并添加方向判断
            done = False
            if self.collision_occurred:
                done = True
            elif target_dist < 2:  # 放宽终止条件到2米
                # 检查是否朝向目标
                direction_vector = self.target_location - current_location
                yaw_diff = abs(
                    current_transform.rotation.yaw - np.degrees(np.arctan2(-direction_vector.y, direction_vector.x)))
                if yaw_diff < 45:  # 角度偏差小于45度时才算成功
                    reward += 1000
                    done = True
                    print(f"成功到达目标！剩余距离：{target_dist:.2f}m")

            return new_obs, reward, done, False, {}

        except Exception as e:
            print(f"执行步骤错误: {str(e)}")
            return np.zeros(self.observation_space.shape), 0, True, False, {}

    def _cleanup_actors(self):
        """同步清理所有Actor并确保彻底销毁"""
        destroy_list = []

        try:
            # 清理传感器（强制同步解除监听）
            for sensor in self.sensors:
                try:
                    if sensor.is_alive:
                        sensor.stop()  # 先停止监听
                        sensor.destroy()  # 同步销毁
                        print(f"传感器 {sensor.id} 已销毁")
                except Exception as e:
                    print(f"传感器销毁失败: {str(e)}")
            self.sensors = []  # 立即清空列表

            # 清理控制器（必须显式停止）
            if hasattr(self, 'controller') and self.controller is not None:
                try:
                    if self.controller.is_alive:
                        self.controller.stop()  # 先停止AI控制
                        time.sleep(0.1)
                        self.controller.destroy()
                        print("控制器已销毁")
                except Exception as e:
                    print(f"控制器销毁失败: {str(e)}")
                finally:
                    self.controller = None

            # 清理行人（同步销毁）
            if hasattr(self, 'pedestrian') and self.pedestrian is not None:
                try:
                    if self.pedestrian.is_alive:
                        # 解除所有可能绑定的组件
                        self.pedestrian.apply_control(carla.WalkerControl())  # 清除控制指令
                        time.sleep(0.1)
                        self.pedestrian.destroy()
                        print("行人已销毁")
                except Exception as e:
                    print(f"行人销毁失败: {str(e)}")
                finally:
                    self.pedestrian = None

            # 清理目标标记
            if self.target_actor and self.target_actor.is_alive:
                try:
                    self.target_actor.destroy()
                    print("目标标记已销毁")
                except Exception as e:
                    print(f"目标标记销毁失败: {str(e)}")
                finally:
                    self.target_actor = None

            # 等待确保销毁完成
            for _ in range(10):
                self.world.tick()
                time.sleep(0.1)

            # 触发垃圾回收
            gc.collect()
            print("所有Actor清理完成")

        except Exception as e:
            print(f"清理过程中发生严重错误: {str(e)}")
        finally:
            # 确保所有引用置空
            self.sensors = []
            self.controller = None
            self.pedestrian = None
            self.target_actor = None

    def close(self):
        """关闭环境"""
        self._cleanup_actors()
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        time.sleep(1)

class TrainingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_count = 0
        self.model = None

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # 记录关键指标
        info.update({
            'current_speed': self.env.current_speed,
            'min_obstacle_distance': self.env.min_obstacle_distance,
            'target_distance': self.env.previous_target_distance
        })

        # 每50步打印训练状态
        if self.episode_count % 50 == 0:
            print(
                f"Episode {self.episode_count} | "
                f"Avg Reward: {np.mean(self.episode_rewards):.1f} | "
                f"Collisions: {self.env.collision_occurred}"
            )

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_count += 1
        if self.episode_count % 50 == 0:
            self.save_checkpoint()
        return self.env.reset(**kwargs)

    def save_checkpoint(self):
        if self.model:
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                self.model.save(f"ped_model_{timestamp}")
                print(f"检查点已保存: ped_model_{timestamp}")
            except Exception as e:
                print(f"保存失败: {str(e)}")

class TrainingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_count = 0
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
        self.episode_rewards = []
        return self.env.reset(**kwargs)


def run_navigation_demo(model_path, episodes=1, gui_callback=None):
    try:
        env = EnhancedPedestrianEnv()
        settings = env.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
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


# ======================== 图形用户界面 ========================
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
                policy_kwargs=self.train_params["policy_kwargs"],
                verbose=0
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
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.demo_thread = None
        self.current_model = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Carla 行人导航系统 v3.0")
        self.setGeometry(200, 200, 1000, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 控制面板
        control_group = QGroupBox("系统控制")
        control_layout = QHBoxLayout()
        self.btn_init = QPushButton("初始化环境")
        self.btn_train = QPushButton("开始训练")
        self.btn_demo = QPushButton("运行演示")
        self.btn_stop = QPushButton("终止进程")
        self.btn_load = QPushButton("加载模型")
        control_layout.addWidget(self.btn_init)
        control_layout.addWidget(self.btn_train)
        control_layout.addWidget(self.btn_demo)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_load)
        control_group.setLayout(control_layout)

        # 参数配置
        param_group = QGroupBox("训练参数")
        param_layout = QHBoxLayout()
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setValue(1e-4)
        self.lr_spin.setPrefix("学习率: ")
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1000, 1000000)
        self.steps_spin.setValue(100000)
        self.steps_spin.setSuffix(" 步")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(64, 2048)
        self.batch_spin.setValue(256)
        self.batch_spin.setSuffix(" 批大小")
        param_layout.addWidget(QLabel("训练参数:"))
        param_layout.addWidget(self.lr_spin)
        param_layout.addWidget(self.steps_spin)
        param_layout.addWidget(self.batch_spin)
        param_group.setLayout(param_layout)

        # 状态显示
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 10))
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("font-weight: bold; color: #444;")
        status_layout.addWidget(self.log_area)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)

        main_layout.addWidget(control_group)
        main_layout.addWidget(param_group)
        main_layout.addWidget(status_group)

        # 信号连接
        self.btn_init.clicked.connect(self.init_environment)
        self.btn_train.clicked.connect(self.start_training)
        self.btn_demo.clicked.connect(self.start_demo)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_load.clicked.connect(self.load_model)
        self.toggle_controls(True)

    def toggle_controls(self, ready):
        self.btn_init.setEnabled(ready)
        self.btn_train.setEnabled(ready and self.current_model is None)
        self.btn_demo.setEnabled(ready and self.current_model is not None)
        self.btn_load.setEnabled(ready)
        self.btn_stop.setEnabled(not ready)

    def log_message(self, msg_type, message):
        if msg_type == "error":
            self.log_area.append(f'<span style="color: red;">[ERROR] {message}</span>')
            QMessageBox.critical(self, "错误", message)
        elif msg_type == "status":
            self.status_label.setText(message)
            self.log_area.append(f'[STATUS] {message}')
        else:
            self.log_area.append(f'[INFO] {message}')

    def init_environment(self):
        try:
            self.log_message("status", "正在连接Carla服务器...")
            test_env = EnhancedPedestrianEnv()
            reset_environment(test_env)
            test_env.close()
            self.log_message("status", "环境初始化成功！")
            QMessageBox.information(self, "成功", "Carla连接成功！")
        except Exception as e:
            self.log_message("error", f"连接失败: {str(e)}")

    def start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            return

        train_params = {
            "learning_rate": self.lr_spin.value(),
            "n_steps": 1024,
            "batch_size": self.batch_spin.value(),
            "total_steps": self.steps_spin.value(),
            "policy_kwargs": {
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "activation_fn": torch.nn.ReLU,
                "ortho_init": True
            }
        }

        self.training_thread = TrainingThread({}, train_params)
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

        self.demo_thread = QThread()
        self.demo_thread.run = lambda: run_navigation_demo(
            self.current_model,
            gui_callback=self.log_message
        )
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
    sys.exit(app.exec_())