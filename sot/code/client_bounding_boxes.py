#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import json
import cv2
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
import pathlib
pathlib.Path("data/images").mkdir(parents=True, exist_ok=True)
pathlib.Path("data/labels").mkdir(parents=True, exist_ok=True)

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


# 在 ClientSideBoundingBoxes 类中添加一个方法来获取车辆的速度并渲染速度文本

class ClientSideBoundingBoxes(object):
    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        bounding_boxes = []
        for vehicle in vehicles:
            bbox = ClientSideBoundingBoxes.get_bounding_box(vehicle, camera)
            speed = vehicle.get_velocity()
            speed_magnitude = np.sqrt(speed.x**2 + speed.y**2 + speed.z**2)  # 计算车辆的速度大小
            bounding_boxes.append((bbox, speed_magnitude))  # 返回边界框和速度
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[0][:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """
        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        chinese_font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 20)

        for bbox, speed in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # 绘制边界框
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
            
            # 绘制速度文本
            speed_text = f"{speed:.2f} m/s"  # 显示速度，保留两位小数
            text_surface = chinese_font.render(speed_text, True, (255, 255, 255))  # 白色文字
            text_rect = text_surface.get_rect(center=(int(bbox[0, 0]), int(bbox[0, 1]) - 10))  # 在边界框上方显示
            bb_surface.blit(text_surface, text_rect)
        
        display.blit(bb_surface, (0, 0))


    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """
    def save_frame_and_labels(self, array, bounding_boxes, frame_idx):
        """
        保存当前帧图像和目标边界框 + 速度信息
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"frame_{frame_idx}_{timestamp}.jpg"
        json_filename = img_filename.replace('.jpg', '.json')

        # 保存图像
        img_path = os.path.join("data/images", img_filename)
        cv2.imwrite(img_path, array)

        # 保存标注
        label_data = []
        for bbox, speed in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(4)]
            is_tracked = False
            if self.tracking_mode and self.target_id is not None:
                x_min = min([p[0] for p in points])
                y_min = min([p[1] for p in points])
                x_max = max([p[0] for p in points])
                y_max = max([p[1] for p in points])
                box_area = (x_max - x_min) * (y_max - y_min)
                # 简易判定是否与当前追踪目标相符（可改进为 IoU）
                is_tracked = (self.target_id is not None)


            label_data.append({
                "bbox": points,
                "speed_m_s": round(speed, 2),
                "tracked_id": self.target_id if is_tracked else None
            })


        with open(os.path.join("data/labels", json_filename), 'w') as f:
            json.dump(label_data, f, indent=2)

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None
        self.display = None
        self.image = None
        self.capture = True
        self.tracker = DeepSort(max_age=15)
        self.current_track_id = None
        self.tracked_lost_counter = 0  # 用于判断是否丢失目标
        self.tracking_mode = True
        self.prev_distance = None        # 上一帧距离
        self.intent_text = ""           # 当前分析结果
        self.speed_history = deque(maxlen=10)      # 用于平滑速度趋势
        self.distance_history = deque(maxlen=10)   # 用于平滑距离趋势


    def select_closest_target(self, bounding_boxes):
        if not bounding_boxes:
            print("[TRACK] 没有检测到目标")
            return None
        """
        在所有检测目标中选择最近一个（速度最快+框最近）
        """

        # 简单使用左上角点距离中心来估算“接近程度”
        center_x = VIEW_WIDTH / 2
        center_y = VIEW_HEIGHT / 2

        min_dist = float('inf')
        closest_bbox = None
        for bbox, speed in bounding_boxes:
            x_coords = bbox[:, 0]
            y_coords = bbox[:, 1]
            x_min, y_min = np.min(x_coords), np.min(y_coords)
            dist = np.sqrt((x_min - center_x) ** 2 + (y_min - center_y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_bbox = (bbox, speed)
        return closest_bbox


    def analyze_intention(self, bbox, speed):
        """
        基于滑动窗口的距离和速度趋势，分析意图
        """
        # 计算当前bbox的中心点
        x_coords = bbox[:, 0]
        y_coords = bbox[:, 1]
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        current_center = (x_center, y_center)

        # 自车在画面中间偏下
        car_center = (VIEW_WIDTH / 2, VIEW_HEIGHT)
        distance = np.linalg.norm(np.array(current_center) - np.array(car_center))

        # 保存历史
        self.speed_history.append(speed)
        self.distance_history.append(distance)

        # 必须积累一定帧数才判断
        if len(self.distance_history) < 5:
            self.intent_text = "目标初始化中"
            return

        # 平均速度和距离变化趋势
        avg_speed = np.mean(self.speed_history)
        delta_distance = self.distance_history[-1] - self.distance_history[0]

        # 阈值逻辑（根据实际调整）
        if delta_distance < -20 and avg_speed > 3.0:
            self.intent_text = "危险靠近"
        elif delta_distance < -5 and avg_speed > 1.0:
            self.intent_text = "目标靠近中"
        elif delta_distance > 5 and avg_speed > 1.0:
            self.intent_text = "目标远离中"
        else:
            self.intent_text = "目标稳定"



    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        渲染图像并返回 RGB 数组（用于保存）
        """
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            rgb_array = array[:, :, :3]  # RGB，不做反转
            bgr_array = rgb_array[:, :, ::-1]  # BGR for pygame

            surface = pygame.surfarray.make_surface(bgr_array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            return rgb_array  # 返回原始 RGB 图像用于保存
        return None



    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = [v for v in self.world.get_actors().filter('vehicle.*') if v.id != self.car.id]


            frame_count = 0  # 添加在循环开始前的计数器
            while True:
                self.world.tick()
                self.capture = True
                pygame_clock.tick_busy_loop(20)

                frame_count += 1
                rgb_array = self.render(self.display)

                bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
                bb_surface.set_colorkey((0, 0, 0))
                font = pygame.font.SysFont("Arial", 20)

                bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)
                # ---- 单目标追踪逻辑 ----
                # 选取最近目标作为 tracking 输入
                target_input = []
                if self.tracking_mode:
                    closest = self.select_closest_target(bounding_boxes)
                    if closest:
                        bbox, speed = closest
                        x_coords = bbox[:, 0]
                        y_coords = bbox[:, 1]
                        x_min, y_min = np.min(x_coords), np.min(y_coords)
                        x_max, y_max = np.max(x_coords), np.max(y_coords)
                        width = x_max - x_min
                        height = y_max - y_min
                        # ✅ 加入边界检查
                        if width >= 10 and height >= 10:
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            width = min(width, VIEW_WIDTH - x_min)
                            height = min(height, VIEW_HEIGHT - y_min)

                            if width >= 10 and height >= 10:
                                target_input = [([x_min, y_min, width, height], 0.9, "vehicle")]

                # 更新 DeepSort
                tracks = self.tracker.update_tracks(target_input, frame=rgb_array)
                self.target_id = None
                if tracks:
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        # 如果还没选目标，或者当前 track 是之前那个
                        if self.current_track_id is None or track.track_id == self.current_track_id:
                            self.current_track_id = track.track_id
                            self.target_id = track.track_id
                            self.tracked_lost_counter = 0  # reset
                            l, t, r, b = track.to_ltrb()
                            pygame.draw.rect(bb_surface, (255, 255, 0), pygame.Rect(l, t, r - l, b - t), 2)
                            text_surface = font.render(f"Tracked ID: {track.track_id}", True, (255, 255, 255))
                            bb_surface.blit(text_surface, (int(l), int(t) - 20))

                            if closest:
                                bbox, speed = closest
                                self.analyze_intention(bbox, speed)

                            chinese_font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 20)
                            intent_color = {
                                "危险靠近": (255, 0, 0),
                                "目标靠近中": (255, 128, 0),
                                "目标远离中": (0, 255, 0),
                                "目标稳定": (200, 200, 200),
                                "目标初始化中": (150, 150, 150)
                            }.get(self.intent_text, (255, 255, 255))
                            intent_surface = chinese_font.render(self.intent_text, True, intent_color)
                            bb_surface.blit(intent_surface, (int(l), int(t) - 40))
                            break
                    else:
                        # 没有匹配的 track id
                        self.tracked_lost_counter += 1
                        if self.tracked_lost_counter > 10:
                            self.current_track_id = None
                else:
                    self.tracked_lost_counter += 1
                    if self.tracked_lost_counter > 10:
                        self.current_track_id = None



                # 每 5 帧保存一次
                if frame_count % 5 == 0 and rgb_array is not None:
                    self.save_frame_and_labels(rgb_array, bounding_boxes, frame_count)

                self.display.blit(bb_surface, (0, 0))
                pygame.display.flip()
                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
