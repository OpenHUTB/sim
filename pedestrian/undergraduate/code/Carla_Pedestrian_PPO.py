import carla
import gymnasium as gym
import numpy as np
import random
import time
import threading
import cv2
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 动作空间定义
ACTION_DICT = {
    0: (0.0, 0.0),  # 停止
    1: (0.0, 1.0),  # 直行
    2: (-30.0, 0.8),  # 左转
    3: (30.0, 0.8),  # 右转
    4: (0.0, 2.0)  # 奔跑
}


class EnhancedPedestrianEnv(gym.Env):
    """强化学习行人环境完整实现"""

    def __init__(self, target_location=carla.Location(x=180, y=120, z=1)):
        super().__init__()

        # ==== 初始化关键属性 ====
        self.target_location = target_location
        self.last_reward = 0.0
        self.previous_speed = 0.0
        self.current_speed = 0.0
        self.collision_occurred = False
        self.min_obstacle_distance = 5.0
        self.previous_target_distance = 0.0
        self.episode_step = 0
        self.dynamic_obstacles = []
        self.sensors = []

        # ==== 线程安全锁 ====
        self.img_lock = threading.Lock()
        self.lidar_lock = threading.Lock()
        self.last_display = time.time()

        # ==== Carla连接配置 ====
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(30.0)
        self._connect_to_server()

        # ==== 空间定义 ====
        self.action_space = spaces.Discrete(len(ACTION_DICT))
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 8 + [0.0, -1.0]),
            high=np.array([1.0] * 8 + [3.0, 1.0]),
            dtype=np.float32
        )

        # ==== 初始化Carla组件 ====
        self._preload_assets()
        self._setup_spectator()

    def _connect_to_server(self):
        """强制使用Town01地图的连接方法"""
        for retry in range(5):
            try:
                # 强制加载Town01地图
                self.world = self.client.load_world("Town01")
                map_name = self.world.get_map().name
                if "Town01" in map_name:
                    print(f"成功加载Town01地图 (Carla v{self.client.get_server_version()})")
                    return
                else:
                    print(f"地图加载异常，获取到：{map_name}")
            except RuntimeError as e:
                print(f"地图加载失败（尝试 {retry + 1}/5）：{str(e)}")
                try:
                    # 如果无法加载，尝试启动离线服务器
                    if not self.client.get_client_version():  # 检查客户端连接
                        self.client.start()
                        print("已启动离线Carla服务器")
                    time.sleep(2)
                except Exception as inner_e:
                    print(f"服务器启动失败：{str(inner_e)}")
                    time.sleep(2)
            except Exception as e:
                print(f"意外错误：{str(e)}")
                time.sleep(2)

        # 最终检查
        try:
            final_map = self.client.get_world().get_map().name
            raise ConnectionError(f"最终连接失败，当前地图：{final_map}（应使用Town01）")
        except:
            raise ConnectionError("无法连接到Carla服务器，请检查：\n1. Carla服务器是否运行\n2. 端口2000是否可用")

    def _preload_assets(self):
        """预加载蓝图资产"""
        self.blueprint_library = self.world.get_blueprint_library()
        self.walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        self.controller_bp = self.blueprint_library.find('controller.ai.walker')
        self.vehicle_bps = self.blueprint_library.filter('vehicle.*')
        self.camera_bp = self._configure_camera()
        self.lidar_bp = self._configure_lidar()
        self.collision_bp = self.blueprint_library.find('sensor.other.collision')

    def _configure_camera(self):
        """配置RGB摄像头"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')  # 降低分辨率
        camera_bp.set_attribute('image_size_y', '360')
        camera_bp.set_attribute('fov', '90')
        return camera_bp

    def _configure_lidar(self):
        """配置激光雷达"""
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '10.0')
        lidar_bp.set_attribute('points_per_second', '10000')  # 降低点云密度
        return lidar_bp

    def _setup_spectator(self):
        """设置观察视角"""
        try:
            spectator = self.world.get_spectator()
            transform = carla.Transform(
                carla.Location(x=160, y=138, z=50),
                carla.Rotation(pitch=-90)
            )
            spectator.set_transform(transform)
        except Exception as e:
            print(f"视角设置失败: {str(e)}")

    def reset(self, **kwargs):
        """重置环境状态"""
        # 清理旧Actor
        self._cleanup_actors()
        time.sleep(0.5)  # 增加清理间隔

        # 初始化状态变量
        self.episode_step = 0
        self.collision_occurred = False
        self.last_reward = 0.0
        self.previous_speed = 0.0
        self.current_speed = 0.0
        self.min_obstacle_distance = 5.0
        self.previous_target_distance = 0.0

        # 生成新行人
        self._spawn_pedestrian()

        # 附加传感器
        self._attach_sensors()

        # 生成障碍物（初始数量较少）
        self._spawn_dynamic_obstacles(num_vehicles=2, num_walkers=1)

        return self._get_obs(), {}

    def _spawn_pedestrian(self):
        """生成受控行人（修复括号问题）"""
        for _ in range(3):
            try:
                spawn_point = carla.Transform(  # 修复此处括号
                    carla.Location(x=160, y=138, z=1.0),
                    carla.Rotation(yaw=random.randint(0, 360))
                )  # 补全括号
                self.pedestrian = self.world.spawn_actor(
                    random.choice(self.walker_bps),
                    spawn_point
                )
                break
            except Exception as e:
                print(f"行人生成失败: {str(e)}")
                time.sleep(0.5)

        # 添加AI控制器
        self.controller = self.world.spawn_actor(
            self.controller_bp,
            carla.Transform(),
            attach_to=self.pedestrian,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.controller.start()

    def _attach_sensors(self):
        """附加传感器（增加错误处理）"""
        try:
            # 碰撞传感器
            collision_sensor = self.world.spawn_actor(
                self.collision_bp,
                carla.Transform(),
                attach_to=self.pedestrian
            )
            collision_sensor.listen(lambda event: self._on_collision(event))

            # 激光雷达
            lidar_transform = carla.Transform(carla.Location(z=2.5))
            lidar_sensor = self.world.spawn_actor(
                self.lidar_bp,
                lidar_transform,
                attach_to=self.pedestrian
            )
            lidar_sensor.listen(lambda data: self._process_lidar(data))

            # 摄像头（降低帧率）
            camera_transform = carla.Transform(
                carla.Location(x=0.8, z=1.7),
                carla.Rotation(pitch=-10)
            )
            camera_sensor = self.world.spawn_actor(
                self.camera_bp,
                camera_transform,
                attach_to=self.pedestrian
            )
            camera_sensor.listen(lambda image: self._process_image(image))

            self.sensors = [collision_sensor, lidar_sensor, camera_sensor]
        except Exception as e:
            print(f"传感器初始化失败: {str(e)}")
            self._cleanup_actors()
            raise

    def _spawn_dynamic_obstacles(self, num_vehicles, num_walkers):
        """生成动态障碍物（安全版本）"""
        try:
            # 生成车辆
            vehicle_spawn_points = [
                p for p in self.world.get_map().get_spawn_points()
                if p.location.distance(self.pedestrian.get_location()) > 20.0
            ]
            for _ in range(num_vehicles):
                vehicle = self.world.spawn_actor(
                    random.choice(self.vehicle_bps),
                    random.choice(vehicle_spawn_points)
                )
                vehicle.set_autopilot(True)
                self.dynamic_obstacles.append(vehicle)

                # 生成行人障碍物
                for _ in range(num_walkers):
                    walker = self.world.spawn_actor(
                        random.choice(self.walker_bps),
                        carla.Transform(self._random_destination()))

                controller = self.world.spawn_actor(
                    self.controller_bp,
                    carla.Transform(),
                    attach_to=walker,
                    attachment_type=carla.AttachmentType.Rigid)
                controller.start()
                controller.go_to_location(self._random_destination())
                self.dynamic_obstacles.extend([walker, controller])
        except Exception as e:
            print(f"障碍物生成失败: {str(e)}")

    def _random_destination(self):
        """生成随机目的地"""
        return carla.Location(
            x=random.uniform(120, 200),
            y=random.uniform(100, 160),
            z=1.0)

    def _on_collision(self, event):
        """碰撞事件处理"""
        self.collision_occurred = True

    def _process_lidar(self, data):
        """处理激光雷达数据（安全版本）"""
        try:
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
            with self.lidar_lock:
                if len(points) > 0:
                    distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
                    self.min_obstacle_distance = np.min(distances)
                else:
                    self.min_obstacle_distance = 5.0
        except Exception as e:
            print(f"激光雷达处理错误: {str(e)}")

    def _process_image(self, image):
        """处理摄像头图像（优化性能）"""
        with self.img_lock:
            try:
                # 降低处理频率
                if time.time() - self.last_display < 0.1:  # 10FPS
                    return

                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                img_bgr = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)

                # 显示基本信息
                cv2.putText(img_bgr, f"Speed: {self.current_speed:.1f}m/s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(img_bgr, f"Reward: {self.last_reward:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # 显示处理后的图像
                cv2.imshow('Pedestrian View', cv2.resize(img_bgr, (960, 480)))
                cv2.waitKey(1)
                self.last_display = time.time()
            except Exception as e:
                print(f"图像处理错误: {str(e)}")

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
                [np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)]
            ])
            local_target = rotation_matrix @ np.array([target_dir.x, target_dir.y])

            # 获取障碍物距离
            with self.lidar_lock:
                obstacle_dist = self.min_obstacle_distance

            return np.array([
                current_loc.x / 200.0 - 1.0,
                current_loc.y / 200.0 - 1.0,
                local_target[0],
                local_target[1],
                np.clip(obstacle_dist / 5.0, 0.0, 1.0),
                self.current_speed / 3.0,
                (self.current_speed - self.previous_speed) / 3.0,
                np.sin(yaw),
                np.cos(yaw),
                target_distance / 100.0
            ], dtype=np.float32)
        except Exception as e:
            print(f"获取观测数据失败: {str(e)}")
            return np.zeros(self.observation_space.shape)

    def step(self, action_idx):
        """执行动作（安全版本）"""
        try:
            self.episode_step += 1

            # 解析动作
            yaw_offset, speed_ratio = ACTION_DICT[action_idx]

            # 平滑转向
            current_yaw = self.pedestrian.get_transform().rotation.yaw
            target_yaw = current_yaw + yaw_offset * 0.3
            self.pedestrian.set_transform(carla.Transform(
                self.pedestrian.get_location(),
                carla.Rotation(yaw=target_yaw)))

            # 速度控制
            base_speed = 1.5 + 1.5 * speed_ratio
            safe_speed = min(base_speed, 3.0) if self.min_obstacle_distance > 2.0 else 0.8
            self.previous_speed = self.current_speed
            self.current_speed = safe_speed

            # 应用控制
            control = carla.WalkerControl(
                direction=carla.Vector3D(1, 0, 0),
                speed=safe_speed)
            self.pedestrian.apply_control(control)

            # 推进仿真
            self.world.tick()

            # 计算奖励
            new_obs = self._get_obs()
            current_target_dist = new_obs[-1] * 100.0
            progress = self.previous_target_distance - current_target_dist
            speed_reward = min(safe_speed / 3.0, 0.5)
            collision_penalty = 10.0 if self.collision_occurred else 0.0
            time_penalty = 0.05
            reward = progress * 2.0 + speed_reward - time_penalty - collision_penalty
            self.last_reward = reward
            self.previous_target_distance = current_target_dist

            # 终止条件
            done = False
            if self.collision_occurred:
                done = True
                reward -= 20.0
            elif current_target_dist < 2.0:
                done = True
                reward += 20.0

            return new_obs, reward, done, False, {}
        except Exception as e:
            print(f"执行步骤时发生错误: {str(e)}")
            return np.zeros(self.observation_space.shape), 0, True, False, {}

    def _cleanup_actors(self):
        """增强版资源清理"""
        destroy_list = []

        # 主行人及控制器
        if hasattr(self, 'pedestrian') and self.pedestrian.is_alive:
            destroy_list.append(self.pedestrian)
        if hasattr(self, 'controller') and self.controller.is_alive:
            destroy_list.append(self.controller)

        # 传感器
        for sensor in self.sensors:
            if sensor.is_alive:
                destroy_list.append(sensor)

        # 动态障碍物
        for obstacle in self.dynamic_obstacles:
            if obstacle.is_alive:
                destroy_list.append(obstacle)

        # 批量销毁
        if destroy_list:
            try:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in destroy_list])
                time.sleep(0.5)  # 等待销毁完成
            except Exception as e:
                print(f"销毁Actor时发生错误: {str(e)}")

        self.sensors = []
        self.dynamic_obstacles = []

    def close(self):
        """关闭环境"""
        self._cleanup_actors()
        cv2.destroyAllWindows()


class TrainingWrapper(gym.Wrapper):
    """训练包装器（稳定版）"""

    def __init__(self, env):
        super().__init__(env)
        self.episode_count = 0
        self.success_history = []
        self.model = None

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            info['episode'] = {'r': reward, 'l': self.episode_count}
            return obs, reward, terminated, truncated, info
        except Exception as e:
            print(f"训练步骤错误: {str(e)}")
            return np.zeros(self.env.observation_space.shape), 0, True, False, {}

    def reset(self, **kwargs):
        self.episode_count += 1
        if self.episode_count % 50 == 0:  # 每50回合保存检查点
            self.save_checkpoint()
        return self.env.reset(**kwargs)

    def save_checkpoint(self):
        """安全保存检查点"""
        if self.model is not None:
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                self.model.save(f"pedestrian_model_{timestamp}")
                print(f"检查点已保存：pedestrian_model_{timestamp}")
            except Exception as e:
                print(f"保存检查点失败: {str(e)}")


if __name__ == "__main__":
    # 初始化环境
    env = EnhancedPedestrianEnv()
    wrapped_env = TrainingWrapper(env)
    vec_env = DummyVecEnv([lambda: wrapped_env])

    # 配置PPO模型（优化参数）
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        policy_kwargs={
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "activation_fn": torch.nn.ReLU
        },
        device='auto'
    )
    wrapped_env.model = model

    try:
        # 分阶段训练
        print("开始第一阶段训练（10k steps）...")
        model.learn(total_timesteps=10000)

        print("开始第二阶段训练（50k steps）...")
        model.learn(total_timesteps=50000, reset_num_timesteps=False)

    finally:
        model.save("pedestrian_ppo")
        vec_env.close()
        print("训练完成，最终模型已保存。")