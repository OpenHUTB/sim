import carla
import numpy as np
import cv2
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import os
import threading
from queue import Queue
class CarlaSyncMode(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=0.05))
        def make_queue(register_event):
            q = deque()
            register_event(q.append)
            self._queues.append(q)
        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.wait_for_tick(self.frame + 1, timeout)

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)
        for q in self._queues:
            q.clear()
# 初始化CARLA客户端
def init_carla_client():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    return client
def main():
    client = init_carla_client()
    world = setup_world(client)
    bp_lib = get_blueprint_library(world)
    vehicle = spawn_vehicle(world, bp_lib)
    camera = spawn_rgb_camera(world, bp_lib, vehicle)
    semantic_camera = spawn_semantic_camera(world, bp_lib, vehicle)
    lidar = spawn_lidar(world, bp_lib, vehicle)
    create_output_dir()
    state_size, action_size, batch_size, episodes = setup_experiment_parameters()
    agent = initialize_training(state_size, action_size)
    actions = get_action_space()
    try:
        with CarlaSyncMode(world, camera, semantic_camera, lidar) as sync_mode:
            spawn_points = world.get_map().get_spawn_points()
            for e in range(episodes):
                spawn_point = random.choice(spawn_points)
                vehicle.set_transform(spawn_point)
                sync_mode.tick(3.0)
                image = sync_mode._queues[1][-1]
                state = np.array(image.raw_data).reshape(200, 200, 3)
                state = np.expand_dims(state, axis=0)
                total_reward = 0
                done = False
                steps = 0
                while not done and steps < 1000:
                    action_idx = agent.act(state)
                    action = actions[action_idx]
                    vehicle.apply_control(carla.VehicleControl(
                        steer=action[0],
                        throttle=action[1],
                        brake=action[2]
                    ))
                    sync_mode.tick(3.0)
                    next_image = sync_mode._queues[1][-1]
                    next_state = np.array(next_image.raw_data).reshape(200, 200, 3)
                    next_state = np.expand_dims(next_state, axis=0)
                    reward = calculate_reward(vehicle, action)
                    total_reward += reward
                    steps += 1
                    done = steps >= 1000
                    agent.remember(state, action_idx, reward, next_state, done)
                    state = next_state
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                print(f"Episode: {e}, Total Reward: {total_reward}, Steps: {steps}")
    finally:
        save_model(agent)
        cleanup(vehicle, camera, semantic_camera, lidar)
# 加载地图并设置天气
def setup_world(client, map_name='Town01'):
    world = client.load_world(map_name)
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)
    return world

# 获取蓝图库
def get_blueprint_library(world):
    return world.get_blueprint_library()

# 添加车辆
def spawn_vehicle(world, bp_lib, vehicle_name='vehicle.tesla.model3'):
    vehicle_bp = bp_lib.find(vehicle_name)
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(False)
    return vehicle

# 添加RGB相机
def spawn_rgb_camera(world, bp_lib, vehicle):
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '200')
    camera_bp.set_attribute('image_size_y', '200')
    camera_bp.set_attribute('fov', '110')
    camera_location = carla.Location(2, 0, 1)
    camera_rotation = carla.Rotation(0, 0, 0)
    camera_transform = carla.Transform(camera_location, camera_rotation)
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

# 添加语义分割相机
def spawn_semantic_camera(world, bp_lib, vehicle):
    semantic_camera_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    semantic_camera_bp.set_attribute('image_size_x', '200')
    semantic_camera_bp.set_attribute('image_size_y', '200')
    semantic_camera_bp.set_attribute('fov', '110')
    semantic_camera_location = carla.Location(2, 0, 1)
    semantic_camera_rotation = carla.Rotation(0, 0, 0)
    semantic_camera_transform = carla.Transform(semantic_camera_location, semantic_camera_rotation)
    semantic_camera = world.spawn_actor(semantic_camera_bp, semantic_camera_transform, attach_to=vehicle)
    return semantic_camera

# 添加激光雷达
def spawn_lidar(world, bp_lib, vehicle):
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '5000')
    lidar_location = carla.Location(0, 0, 2.5)
    lidar_rotation = carla.Rotation(0, 0, 0)
    lidar_transform = carla.Transform(lidar_location, lidar_rotation)
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    return lidar

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DQN模型构建
def build_model(input_shape, action_size):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# PERDQN代理类
class PERDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 创建输出目录
def create_output_dir():
    if not os.path.exists('output'):
        os.makedirs('output')

# 定义奖励函数
def calculate_reward(vehicle, action):
    velocity = vehicle.get_velocity()
    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
    transform = vehicle.get_transform()
    lane_offset = transform.location.y
    reward = 0.0
    if abs(lane_offset) > 2.0:
        reward -= 10.0
    else:
        reward += 5.0
    if speed < 5 or speed > 15:
        reward -= 2.0
    else:
        reward += 1.0
    if transform.location.x > 100.0:
        reward += 10.0
    if abs(transform.rotation.yaw) > 15.0:
        reward -= 1.0
    return reward

# 设置实验参数
def setup_experiment_parameters():
    state_size = (200, 200, 3)
    action_size = 5
    batch_size = 32
    episodes = 300
    return state_size, action_size, batch_size, episodes

# 初始化训练过程
def initialize_training(state_size, action_size):
    agent = PERDQNAgent(state_size, action_size)
    return agent

# 定义动作空间
def get_action_space():
    steer_space = [-0.5, 0.0, 0.5]
    throttle_space = [0.3, 0.6]
    brake_space = [0.0, 0.3]
    actions = []
    action_idx = agent.act(state)
    action = actions[action_idx]
    for steer in steer_space:
        for throttle in throttle_space:
            for brake in brake_space:
                actions.append((steer, throttle, brake))
    return actions
def spawn_collision_sensor(world, bp_lib, vehicle):
    collision_bp = bp_lib.find('sensor.other.collision')
    collision_location = carla.Location(0, 0, 0)
    collision_rotation = carla.Rotation(0, 0, 0)
    collision_transform = carla.Transform(collision_location, collision_rotation)
    collision_sensor = world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle)
    return collision_sensor

def main():
    client = init_carla_client()
    world = setup_world(client)
    bp_lib = get_blueprint_library(world)
    vehicle = spawn_vehicle(world, bp_lib)
    camera = spawn_rgb_camera(world, bp_lib, vehicle)
    semantic_camera = spawn_semantic_camera(world, bp_lib, vehicle)
    lidar = spawn_lidar(world, bp_lib, vehicle)
    collision_sensor = spawn_collision_sensor(world, bp_lib, vehicle)
    create_output_dir()
    state_size, action_size, batch_size, episodes = setup_experiment_parameters()
    agent = initialize_training(state_size, action_size)
    actions = get_action_space()
    try:
        with CarlaSyncMode(world, camera, semantic_camera, lidar, collision_sensor) as sync_mode:
            spawn_points = world.get_map().get_spawn_points()
            for e in range(episodes):
                spawn_point = random.choice(spawn_points)
                vehicle.set_transform(spawn_point)
                sync_mode.tick(3.0)
                image = sync_mode._queues[1][-1]
                state = np.array(image.raw_data).reshape(200, 200, 3)
                state = np.expand_dims(state, axis=0)
                total_reward = 0
                done = False
                steps = 0
                while not done and steps < 1000:
                    action_idx = agent.act(state)
                    action = actions[action_idx]
                    vehicle.apply_control(carla.VehicleControl(
                        steer=action[0],
                        throttle=action[1],
                        brake=action[2]
                    ))
                    sync_mode.tick(3.0)
                    next_image = sync_mode._queues[1][-1]
                    next_state = np.array(next_image.raw_data).reshape(200, 200, 3)
                    next_state = np.expand_dims(next_state, axis=0)
                    reward = calculate_reward(vehicle, action)
                    total_reward += reward
                    steps += 1
                    # 检查碰撞
                    if len(sync_mode._queues[4]) > 0:
                        done = True
                    agent.remember(state, action_idx, reward, next_state, done)
                    state = next_state
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                print(f"Episode: {e}, Total Reward: {total_reward}, Steps: {steps}")
    finally:
        save_model(agent)
        cleanup(vehicle, camera, semantic_camera, lidar, collision_sensor)
# 训练循环
def training_loop(world, vehicle, camera, semantic_camera, lidar, agent, actions, batch_size, episodes):
    spawn_points = world.get_map().get_spawn_points()
    max_steps_per_episode = 1000
    for e in range(episodes):
        spawn_point = random.choice(spawn_points)
        vehicle.set_transform(spawn_point)
        image = camera.get()
        image = sync_mode._queues[1][-1]  # 假设第二个传感器是RGB相机
        state = np.array(image.raw_data).reshape(200, 200, 3)
        state = np.expand_dims(state, axis=0)
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            action_idx = agent.act(state)
            action = actions[action_idx]
            vehicle.apply_control(carla.VehicleControl(
                steer=action[0],
                throttle=action[1],
                brake=action[2]
            ))
            next_image = camera.get()
            next_state = np.array(next_image.raw_data).reshape(200, 200, 3)
            next_state = np.expand_dims(next_state, axis=0)
            reward = calculate_reward(vehicle, action)
            total_reward += reward
            steps += 1
            if steps >= max_steps_per_episode:
                done = True
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f"Episode: {e}, Total Reward: {total_reward}, Steps: {steps}")

# 保存模型
def save_model(agent, model_path='perdqn_model.h5'):
    agent.model.save(model_path)

# 清理环境
def cleanup(vehicle, camera, semantic_camera, lidar, collision_sensor):
    vehicle.destroy()
    camera.destroy()
    semantic_camera.destroy()
    lidar.destroy()
    collision_sensor.destroy()

# 同步模式辅助类
class CarlaSyncMode(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=0.05))
        def make_queue(register_event):
            q = deque()
            register_event(q.append)
            self._queues.append(q)
        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.wait_for_tick(self.frame + 1, timeout)

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)
        for q in self._queues:
            q.clear()

# 主函数
def main():
    client = init_carla_client()
    world = setup_world(client)
    bp_lib = get_blueprint_library(world)
    vehicle = spawn_vehicle(world, bp_lib)
    camera = spawn_rgb_camera(world, bp_lib, vehicle)
    semantic_camera = spawn_semantic_camera(world, bp_lib, vehicle)
    lidar = spawn_lidar(world, bp_lib, vehicle)
    create_output_dir()
    state_size, action_size, batch_size, episodes = setup_experiment_parameters()
    agent = initialize_training(state_size, action_size)
    actions = get_action_space()
    try:
        with CarlaSyncMode(world, camera, semantic_camera, lidar) as sync_mode:
            spawn_points = world.get_map().get_spawn_points()
            for e in range(episodes):
                spawn_point = random.choice(spawn_points)
                vehicle.set_transform(spawn_point)
                sync_mode.tick(3.0)
                image = sync_mode._queues[1][-1]
                state = np.array(image.raw_data).reshape(200, 200, 3)
                state = np.expand_dims(state, axis=0)
                total_reward = 0
                done = False
                steps = 0
                while not done and steps < 1000:
                    action_idx = agent.act(state)
                    action = actions[action_idx]
                    vehicle.apply_control(carla.VehicleControl(
                        steer=action[0],
                        throttle=action[1],
                        brake=action[2]
                    ))
                    sync_mode.tick(3.0)
                    next_image = sync_mode._queues[1][-1]
                    next_state = np.array(next_image.raw_data).reshape(200, 200, 3)
                    next_state = np.expand_dims(next_state, axis=0)
                    reward = calculate_reward(vehicle, action)
                    total_reward += reward
                    steps += 1
                    done = steps >= 1000
                    agent.remember(state, action_idx, reward, next_state, done)
                    state = next_state
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                print(f"Episode: {e}, Total Reward: {total_reward}, Steps: {steps}")
    finally:
        save_model(agent)
        cleanup(vehicle, camera, semantic_camera, lidar)

if __name__ == '__main__':
    main()
# 检查碰撞
if len(sync_mode._queues[4]) > 0:  # 假设第五个传感器是碰撞传感器
    done = True
