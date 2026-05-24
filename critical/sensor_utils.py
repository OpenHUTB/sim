# utils/sensor_utils.py
# 传感器工具：碰撞检测、RGB 相机、激光雷达、距离监控、车道入侵

import weakref
import numpy as np

import carla


# ================================================================
# 碰撞传感器
# ================================================================
class CollisionSensor:
    """挂载到车辆的碰撞事件传感器"""

    def __init__(self, vehicle):
        world = vehicle.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self._sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.collided = False
        self.collision_intensity = 0.0
        self._collision_frame = -1
        self._sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self.collided = True
        impulse = event.normal_impulse
        self.collision_intensity = np.sqrt(
            impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

    def reset(self):
        """每个 episode 开始时重置"""
        self.collided = False
        self.collision_intensity = 0.0
        self._collision_frame = -1

    def destroy(self):
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None


# ================================================================
# RGB 相机传感器
# ================================================================
class RGBCamera:
    """前置 RGB 相机，用于数据记录与可视化"""

    def __init__(self, vehicle, image_size_x=640, image_size_y=480, fov=90.0,
                 transform=None):
        world = vehicle.get_world()
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(image_size_x))
        bp.set_attribute("image_size_y", str(image_size_y))
        bp.set_attribute("fov", str(fov))

        if transform is None:
            transform = carla.Transform(
                carla.Location(x=1.5, y=0.0, z=2.0))

        self._sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.latest_image = None
        self._image_count = 0
        self._sensor.listen(lambda image: self._on_image(image))

    def _on_image(self, image):
        """接收相机原始数据，转为 numpy 数组"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_image = array[:, :, :3]  # 去掉 alpha 通道
        self._image_count += 1

    def save_image(self, path):
        """保存最新一帧到指定路径"""
        if self.latest_image is not None:
            from PIL import Image
            img = Image.fromarray(self.latest_image)
            img.save(path)

    def destroy(self):
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None


# ================================================================
# 激光雷达传感器
# ================================================================
class LidarSensor:
    """激光雷达，用于感知周围障碍物"""

    def __init__(self, vehicle, points_per_second=100000, range_m=50.0,
                 transform=None):
        world = vehicle.get_world()
        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        bp.set_attribute("points_per_second", str(points_per_second))
        bp.set_attribute("range", str(range_m))

        if transform is None:
            transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))

        self._sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.latest_points = None
        self._sensor.listen(lambda data: self._on_data(data))

    def _on_data(self, data):
        """将点云数据转为 numpy (N, 3)"""
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        self.latest_points = points.reshape((-1, 4))[:, :3]

    def destroy(self):
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None


# ================================================================
# 距离监控器
# ================================================================
class DistanceMonitor:
    """实时监控自车与目标之间的距离"""

    def __init__(self, ego_vehicle, target_vehicle):
        self.ego = ego_vehicle
        self.target = target_vehicle
        self.current_distance = 999.0
        self.min_distance = 999.0
        self.distance_history = []

    def update(self):
        """更新当前距离并将记录写入历史"""
        if self.ego is None or self.target is None:
            self.current_distance = 999.0
            return 999.0
        dist = self.ego.get_location().distance(self.target.get_location())
        self.current_distance = dist
        if dist < self.min_distance:
            self.min_distance = dist
        self.distance_history.append(dist)
        if len(self.distance_history) > 2000:
            self.distance_history.pop(0)
        return dist

    def get_current_distance(self):
        return self.current_distance

    def get_min_distance(self):
        return self.min_distance

    def reset(self):
        self.current_distance = 999.0
        self.min_distance = 999.0
        self.distance_history.clear()


# ================================================================
# 车道入侵传感器
# ================================================================
class LaneInvasionSensor:
    """检测车辆是否偏离车道"""

    def __init__(self, vehicle):
        world = vehicle.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self._sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.invaded = False
        self.invaded_lanes = []
        self._sensor.listen(lambda event: self._on_invasion(event))

    def _on_invasion(self, event):
        self.invaded = True
        self.invaded_lanes = list(event.crossed_lane_markings)

    def reset(self):
        self.invaded = False
        self.invaded_lanes.clear()

    def destroy(self):
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None
