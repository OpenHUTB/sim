import carla
import gymnasium as gym
import numpy as np
import random
import time
import threading
import torch
import gc
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ACTION_DICT = {
    0: (0.0, 0.0),  # 停止
    1: (0.0, 1.0),  # 直行
    2: (-30.0, 0.8),  # 左转
    3: (30.0, 0.8),  # 右转
    4: (0.0, 2.0)  # 奔跑
}

def reset_environment(env):
    """重置环境，确保每次运行时环境被清理并初始化"""
    print("正在重置环境...")
    try:
        env.close()  # 确保关闭上次的环境
        env.reset()  # 重置环境
        print("环境已重置")
    except Exception as e:
        print(f"重置环境时发生错误: {str(e)}")

class EnhancedPedestrianEnv(gym.Env):
    def __init__(self, target_location=carla.Location(x=202, y=65, z=1)):
        super().__init__()

        # 初始化关键属性
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
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 8 + [0.0, -1.0]),
            high=np.array([1.0] * 8 + [3.0, 1.0]),
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
            if hasattr(self, 'pedestrian') and self.pedestrian.is_alive:
                ped_loc = self.pedestrian.get_transform().location
                self.spectator.set_transform(carla.Transform(
                    carla.Location(x=ped_loc.x, y=ped_loc.y, z=20),
                    carla.Rotation(pitch=-90)
                ))
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

            # 显式停止控制器（防止残留控制信号）
            if hasattr(self, 'controller') and self.controller.is_alive:
                self.controller.stop()

            # 重置状态变量
            self.episode_step = 0
            self.collision_occurred = False
            self.last_reward = 0.0
            self.previous_speed = 0.0
            self.current_speed = 0.0
            self.min_obstacle_distance = 5.0
            self.previous_target_distance = 0.0

            # 生成新实例
            self._spawn_pedestrian()
            self._attach_sensors()
            self._spawn_target_marker()
            self._update_spectator_view()

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
                [np.cos(yaw), np.sin(yaw), 0],
                [-np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            local_target = rotation_matrix @ np.array([target_dir.x, target_dir.y, target_dir.z])

            return np.array([
                current_loc.x / 200 - 1,
                current_loc.y / 200 - 1,
                local_target[0],
                local_target[1],
                np.clip(self.min_obstacle_distance / 5, 0, 1),
                self.current_speed / 3,
                (self.current_speed - self.previous_speed) / 3,
                np.sin(yaw),
                np.cos(yaw),
                target_distance / 100
            ], dtype=np.float32)
        except Exception as e:
            print(f"观测获取失败: {str(e)}")
            return np.zeros(self.observation_space.shape)

    def step(self, action_idx):
        """执行动作"""
        try:
            # 获取行人当前朝向
            current_yaw = self.pedestrian.get_transform().rotation.yaw

            # 从 ACTION_DICT 中解析动作
            if isinstance(action_idx, (np.ndarray, list)):
                action_idx = int(action_idx[0])
            else:
                action_idx = int(action_idx)
            yaw_offset, speed_ratio = ACTION_DICT[action_idx]

            # 状态信息计算
            target_vector = self.target_location - self.pedestrian.get_location()
            target_dist = target_vector.length()
            target_yaw = np.degrees(np.arctan2(target_vector.y, target_vector.x))
            yaw_diff = np.arctan2(np.sin(np.radians(target_yaw - current_yaw)),
                                  np.cos(np.radians(target_yaw - current_yaw)))
            yaw_diff = np.degrees(yaw_diff)

            # 转向控制
            auto_steer = np.clip(yaw_diff / 30, -1, 1) * 15
            final_yaw = current_yaw + yaw_offset * 0.3 + auto_steer
            self.pedestrian.set_transform(carla.Transform(
                self.pedestrian.get_location(),
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

            # 获取新的观测数据
            new_obs = self._get_obs()

            # ==== 核心奖励计算 ====
            progress = self.previous_target_distance - target_dist
            alignment = np.cos(np.radians(yaw_diff))  # 使用修正后的 yaw_diff

            reward = (
                    progress * 10  # 进展奖励
                    + alignment * 2.0  # 方向对齐奖励
                    + (1 - target_dist / 10) ** 2 * 3  # 近距离指数奖励
                    + min(safe_speed / 3, 0.3)  # 降低速度奖励权重
                    - 0.05  # 固定惩罚
                    - self.episode_step * 0.01  # 时间惩罚
                    - (30 if self.collision_occurred else 0)  # 碰撞惩罚
            )

            # 新增远离惩罚
            if progress < 0:
                reward += progress * 20  # 远离时额外惩罚

            # 动态惩罚
            if target_dist > 15:
                reward -= target_dist * 0.01
            if target_dist > 10 and abs(yaw_diff) > 15:
                reward -= abs(yaw_diff) / 100

            # 更新状态变量
            self.last_reward = reward
            self.previous_target_distance = target_dist

            # 终止条件
            done = False
            if self.collision_occurred:
                done = True
                reward -= 30  # 确保碰撞时总惩罚为-30
            elif target_dist < 1:  # 到达目标
                done = True
                reward += 30  # 提高到达奖励
                print("到达目标！")

            return new_obs, reward, done, False, {}

        except Exception as e:
            print(f"执行步骤错误: {str(e)}")
            return np.zeros(self.observation_space.shape), 0, True, False, {}

    def _cleanup_actors(self):
        """清理所有Actor"""
        destroy_list = []
        try:
            if hasattr(self, 'pedestrian') and self.pedestrian.is_alive:
                destroy_list.append(self.pedestrian)
            if hasattr(self, 'controller') and self.controller.is_alive:
                destroy_list.append(self.controller)
            for sensor in self.sensors:
                if sensor.is_alive:
                    destroy_list.append(sensor)
            if self.target_actor and self.target_actor.is_alive:
                destroy_list.append(self.target_actor)

            if destroy_list:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in destroy_list])
                self.world.tick()
                time.sleep(0.5)

        except Exception as e:
            print(f"清理Actor时发生错误: {str(e)}")
        finally:
            self.sensors = []
            gc.collect()

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
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            info['episode'] = {'r': reward, 'l': self.episode_count}
            return obs, reward, terminated, truncated, info
        except Exception as e:
            print(f"训练步骤错误: {str(e)}")
            return np.zeros(self.env.observation_space.shape), 0, True, False, {}

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


def run_navigation_demo(model_path, episodes=1):
    """运行导航演示"""
    env = None
    try:
        env = EnhancedPedestrianEnv()
        model = PPO.load(model_path)

        for episode in range(episodes):
            # 调用 reset_environment 来确保每次导航前重置环境
            reset_environment(env)

            done = False
            total_reward = 0
            step_count = 0

            print(f"\n=== 第 {episode + 1}/{episodes} 次导航演示 ===")
            print("初始位置:", env.pedestrian.get_transform().location)
            print("目标位置:", env.target_location)

            # 转换初始观测值
            obs, _ = env.reset()
            if isinstance(obs, dict):
                obs = obs['observation']
            obs = np.array(obs).flatten().astype(np.float32)

            while not done and step_count < 10000:
                # 确保观测值维度正确
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)

                action, _ = model.predict(obs, deterministic=True)

                # 转换动作类型
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                else:
                    action = int(action)

                obs, reward, done, _, _ = env.step(action)

                # 更新观测值
                if isinstance(obs, dict):
                    obs = obs['observation']
                obs = np.array(obs).flatten().astype(np.float32)

                # 显示导航信息
                current_loc = env.pedestrian.get_transform().location
                target_dist = np.linalg.norm([current_loc.x - env.target_location.x,
                                              current_loc.y - env.target_location.y])
                print(f"步骤 {step_count}: 当前位置({current_loc.x:.1f}, {current_loc.y:.1f}) "
                      f"剩余距离: {target_dist:.1f}m 当前速度: {env.current_speed:.1f}m/s 当前奖励: {reward:.2f}")

                total_reward += reward
                step_count += 1
                time.sleep(0.05)

            print(f"演示结束！累计奖励: {total_reward:.2f}")
            print("最终位置:", current_loc)
            print("=" * 50)

    except Exception as e:
        print(f"导航演示发生严重错误: {str(e)}")
    finally:
        if env:
            env.close()
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)


if __name__ == "__main__":
    # 训练阶段
    env = EnhancedPedestrianEnv()
    reset_environment(env)  # 每次开始训练前重置环境

    wrapped_env = TrainingWrapper(env)
    vec_env = DummyVecEnv([lambda: wrapped_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        gamma=0.995,
        policy_kwargs={
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "activation_fn": torch.nn.ReLU
        },
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    wrapped_env.model = model

    try:
        print("=== 开始训练 ===")
        print("第一阶段训练（100k steps）...")
        model.learn(10000)
        print("第二阶段训练（200k steps）...")
        model.learn(10000, reset_num_timesteps=False)

        # 保存最终模型
        model.save("pedestrian_ppo")
        print("\n训练完成，模型已保存为 pedestrian_ppo.zip")

    finally:
        vec_env.close()

    # 导航演示阶段
    try:
        print("\n=== 开始导航演示 ===")
        run_navigation_demo("pedestrian_ppo", episodes=1)
    except Exception as e:
        print(f"导航演示发生错误: {str(e)}")
    finally:
        # 确保彻底关闭环境
        env.close()
        time.sleep(2)