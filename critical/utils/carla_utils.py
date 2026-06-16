# utils/carla_utils.py
# CARLA 仿真器连接、车辆生成、同步模式等基础工具
# 不包含场景差异化逻辑，仅提供通用操作

import random
import carla

from config.carla_config import (
    CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT,
    EGO_VEHICLE_BLUEPRINT, ADV_VEHICLE_BLUEPRINT,
)


# ================================================================
# 连接管理
# ================================================================

def connect_to_carla(host=None, port=None, timeout=None):
    """连接 CARLA 服务器，返回 (client, world)"""
    host = host or CARLA_HOST
    port = port or CARLA_PORT
    timeout = timeout or CARLA_TIMEOUT
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    return client, world


def enable_sync_mode(world, fps=20):
    """开启同步模式"""
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)


def disable_sync_mode(world):
    """关闭同步模式"""
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)


# ================================================================
# 车辆操作
# ================================================================

def spawn_ego_vehicle(world, spawn_point=None):
    """生成自车"""
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find(EGO_VEHICLE_BLUEPRINT)
    if bp is None:
        available = bp_lib.filter("vehicle.*")
        bp = available[0] if available else None
        if bp is None:
            raise RuntimeError("没有可用的车辆蓝图")
    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
    return world.spawn_actor(bp, spawn_point)


def spawn_adv_vehicle(world, spawn_point=None):
    """生成对抗车辆"""
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find(ADV_VEHICLE_BLUEPRINT)
    if bp is None:
        available = bp_lib.filter("vehicle.*")
        bp = available[1] if len(available) > 1 else available[0]
    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[1] if len(spawn_points) > 1 else spawn_points[0]
    return world.try_spawn_actor(bp, spawn_point)


def spawn_npc_vehicles(world, count=20):
    """批量生成 NPC 车辆并开启自动驾驶"""
    vehicles = []
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    for _ in range(count):
        bp = random.choice(vehicle_bps)
        sp = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(bp, sp)
        if vehicle is not None:
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
    return vehicles


# ================================================================
# 行人操作
# ================================================================

def spawn_pedestrians(world, count=10):
    """批量生成行人"""
    pedestrians = []
    bp_lib = world.get_blueprint_library()
    ped_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")
    for _ in range(count):
        spawn_loc = world.get_random_location_from_navigation()
        if spawn_loc is None:
            continue
        spawn_transform = carla.Transform(location=spawn_loc)
        ped = world.try_spawn_actor(random.choice(ped_bps), spawn_transform)
        if ped is None:
            continue
        controller = world.spawn_actor(controller_bp, carla.Transform(),
                                       attach_to=ped)
        controller.start()
        walk_target = world.get_random_location_from_navigation()
        if walk_target:
            controller.go_to_location(walk_target)
        controller.set_max_speed(1.2 + random.random() * 0.8)
        pedestrians.append((ped, controller))
    return pedestrians


def spawn_pedestrian_at(world, location, speed_ms=1.5):
    """
    在指定位置附近生成行人并返回 (pedestrian, controller)。
    自动查找最近的可导航点生成，确保 AI 控制器能正常寻路。
    """
    bp_lib = world.get_blueprint_library()
    ped_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")

    # 找附近可导航点作为生成位置
    spawn_loc = _find_nearby_navigable(world, location)
    if spawn_loc is None:
        spawn_loc = location
    spawn_loc.z += 0.2

    ped = None
    for dx in [0, 1, -1, 2, -2, 3, -3]:
        for dy in [0, 1, -1, 2, -2]:
            sp = carla.Transform(carla.Location(
                x=spawn_loc.x + dx, y=spawn_loc.y + dy, z=spawn_loc.z))
            ped = world.try_spawn_actor(random.choice(ped_bps), sp)
            if ped is not None:
                break
        if ped is not None:
            break

    if ped is None:
        raise RuntimeError("行人生成失败: (%.1f, %.1f)" % (location.x, location.y))

    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=ped)
    controller.start()
    controller.set_max_speed(speed_ms)
    return ped, controller


def _find_nearby_navigable(world, location, max_distance=50.0):
    """在目标位置附近查找可导航点"""
    try:
        nav_map = world.get_map()
        # 尝试在目标附近找一个 waypoint
        waypoint = nav_map.get_waypoint(location, project_to_road=False,
                                         lane_type=carla.LaneType.Sidewalk)
        if waypoint is not None:
            return waypoint.transform.location
        # 回退：尝试 Any 类型
        waypoint = nav_map.get_waypoint(location)
        if waypoint is not None:
            return waypoint.transform.location
    except Exception:
        pass
    return None


def walk_to_location(controller, world, target_location, speed_ms=None):
    """
    让行人走向目标位置（自动查找可导航目标点）。
    调用此函数而非直接 go_to_location，确保终点可达。
    """
    target = _find_nearby_navigable(world, target_location)
    if target is None:
        target = target_location
    if speed_ms is not None:
        controller.set_max_speed(speed_ms)
    controller.go_to_location(target)
    return target



# ================================================================
# 车辆控制
# ================================================================

def set_vehicle_speed(vehicle, speed_ms):
    """设置车辆目标速度 (m/s)"""
    forward = vehicle.get_transform().get_forward_vector()
    vehicle.set_target_velocity(carla.Vector3D(
        float(forward.x * speed_ms),
        float(forward.y * speed_ms),
        0.0))


def apply_brake(vehicle, brake_amount=1.0):
    """制动"""
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=brake_amount))


# ================================================================
# 地图工具
# ================================================================

def get_spawn_points(world):
    return world.get_map().get_spawn_points()


def get_random_spawn_point(world):
    points = world.get_map().get_spawn_points()
    return random.choice(points)


# ================================================================
# 清理
# ================================================================

def destroy_actors(actors):
    """批量销毁 actor"""
    for actor in actors:
        if actor is not None:
            try:
                actor.destroy()
            except RuntimeError:
                pass


def cleanup_all(world, *actor_lists):
    """批量清理"""
    destroy_actors(actor_lists)
