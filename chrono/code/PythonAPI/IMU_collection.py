#!/usr/bin/env python
# Copyright (c) 2019 Carnegie Mellon University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit

    F2           : 启动/停止 IMU数据录制
"""
from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import glob
import os
import sys
import csv
import time
import datetime
import collections
import random
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pygame
from pygame.locals import *


# ==============================================================================
# -- IMU 数据录制类 -------------------------------------------------------------
# ==============================================================================
class IMUDataCollector:
    def __init__(self, save_dir='imu_records'):
        self.imu_data = []
        self.save_dir = save_dir
        self.is_recording = False
        os.makedirs(self.save_dir, exist_ok=True)
        self.file_name = f"imu_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.file_path = os.path.join(self.save_dir, self.file_name)

    def add_imu_data(self, timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, compass, vehicle_id):
        if self.is_recording:
            self.imu_data.append({
                'timestamp': timestamp,
                'vehicle_id': vehicle_id,
                'accel_x': accel_x,
                'accel_y': accel_y,
                'accel_z': accel_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'compass': compass,
                'record_time': datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
            })

    def start_recording(self):
        self.is_recording = True
        self.imu_data.clear()
        print(f"[IMU] 开始录制: {self.file_path}")
        return "IMU 录制中..."

    def stop_recording(self):
        self.is_recording = False
        if not self.imu_data:
            print("[IMU] 无数据可保存")
            return "无IMU数据"
        with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'record_time', 'vehicle_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y',
                          'gyro_z', 'compass']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.imu_data)
        print(f"[IMU] 保存成功，共 {len(self.imu_data)} 条数据")
        return f"保存成功: {len(self.imu_data)} 条"


# ==============================================================================
# -- 全局工具函数 ----------------------------------------------------------------
# ==============================================================================
def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World 世界管理类 ------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world, hud, args, imu_collector):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self.imu_collector = imu_collector
        self._weather_index = 0
        self._actor_filter = args.filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player, self.imu_collector)
        self.camera_manager = CameraManager(self.player, self.hud)

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        if self.camera_manager:
            self.camera_manager.sensor.destroy()
        sensors = [self.collision_sensor, self.lane_invasion_sensor, self.gnss_sensor, self.imu_sensor]
        for s in sensors:
            if s and s.sensor:
                s.sensor.destroy()
        if self.player:
            self.player.destroy()


# ==============================================================================
# -- 键盘控制类 ------------------------------------------------------------------
# ==============================================================================
class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self.world = world
        self.hud = world.hud
        self._autopilot_enabled = start_in_autopilot
        self.imu_collector = world.imu_collector
        self._control = carla.VehicleControl()
        world.player.set_autopilot(self._autopilot_enabled)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == QUIT:
                return True
            elif event.type == KEYUP:
                if event.key == K_ESCAPE:
                    return True
                # F2 录制IMU数据
                elif event.key == K_F2:
                    if self.imu_collector.is_recording:
                        msg = self.imu_collector.stop_recording()
                    else:
                        msg = self.imu_collector.start_recording()
                    self.hud.notification(msg)
                # 其他快捷键保留原版功能
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)
                    self.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        # 基础驾驶控制
        keys = pygame.key.get_pressed()
        self._control.throttle = 1.0 if keys[K_w] else 0.0
        self._control.brake = 1.0 if keys[K_s] else 0.0
        self._control.steer = -0.5 if keys[K_a] else (0.5 if keys[K_d] else 0.0)
        world.player.apply_control(self._control)
        return False


# ==============================================================================
# -- HUD 显示类 -----------------------------------------------------------------
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        self._notifications = FadingText(pygame.font.Font(None, 20), (width, 40), (0, height - 40))
        self.server_fps = 0
        self.simulation_time = 0

    def on_world_tick(self, timestamp):
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def render(self, display):
        self._notifications.render(display)


# ==============================================================================
# -- 传感器类 -------------------------------------------------------------------
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        self.sensor.listen(lambda event: self._on_collision(event))

    def get_collision_history(self):
        # 核心修复：default → defaultdict
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    def _on_collision(self, event):
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=parent_actor)


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.lat = 0.0
        self.lon = 0.0
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=parent_actor)


class IMUSensor(object):
    def __init__(self, parent_actor, data_collector):
        self.sensor = None
        self._parent = parent_actor
        self.data_collector = data_collector
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        self.sensor.listen(lambda data: self._imu_callback(data))

    def _imu_callback(self, sensor_data):
        self.data_collector.add_imu_data(
            timestamp=sensor_data.timestamp,
            accel_x=sensor_data.accelerometer.x,
            accel_y=sensor_data.accelerometer.y,
            accel_z=sensor_data.accelerometer.z,
            gyro_x=sensor_data.gyroscope.x,
            gyro_y=sensor_data.gyroscope.y,
            gyro_z=sensor_data.gyroscope.z,
            compass=math.degrees(sensor_data.compass),
            vehicle_id=self._parent.id
        )


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=-5, z=2.5)), attach_to=self._parent)
        self.surface = None

    def render(self, display):
        if self.surface:
            display.blit(self.surface, (0, 0))


# ==============================================================================
# -- 辅助UI类 -------------------------------------------------------------------
# ==============================================================================
class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, seconds=2.0):
        self.text = text
        self.seconds_left = seconds

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)

    def render(self, display):
        if self.seconds_left > 0:
            text_surf = self.font.render(self.text, True, (255, 255, 255))
            display.blit(text_surf, self.pos)


# ==============================================================================
# -- 主循环 ---------------------------------------------------------------------
# ==============================================================================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    imu_collector = IMUDataCollector()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args, imu_collector)
        controller = KeyboardControl(world, args.autopilot)
        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if imu_collector.is_recording:
            imu_collector.stop_recording()
        if world:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- 主函数 ---------------------------------------------------------------------
# ==============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--res', default='1280x720')
    parser.add_argument('--autopilot', action='store_true')
    parser.add_argument('--filter', default='vehicle.*')
    args = parser.parse_args()
    args.width, args.height = map(int, args.res.split('x'))
    game_loop(args)


if __name__ == '__main__':
    main()