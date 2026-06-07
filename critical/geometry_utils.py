# utils/geometry_utils.py
# 几何计算工具：距离、角度、车道位置、轨迹偏差、包围盒重叠

import math
import numpy as np

import carla


# ================================================================
# 距离
# ================================================================

def distance_2d(loc1, loc2):
    """两点二维欧氏距离（忽略 z 轴）"""
    return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)


def distance_3d(loc1, loc2):
    """两点三维欧氏距离"""
    return math.sqrt(
        (loc1.x - loc2.x) ** 2
        + (loc1.y - loc2.y) ** 2
        + (loc1.z - loc2.z) ** 2
    )


def distance_between_vehicles(ego_vehicle, adv_vehicle):
    """两车之间的 3D 距离"""
    ego_loc = ego_vehicle.get_location()
    adv_loc = adv_vehicle.get_location()
    return ego_loc.distance(adv_loc)


# ================================================================
# 速度
# ================================================================

def speed_ms(vehicle):
    """获取车辆速度 (m/s)"""
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def speed_kmh(vehicle):
    """获取车辆速度 (km/h)"""
    return speed_ms(vehicle) * 3.6


def relative_speed(ego, adv):
    """相对速度 (m/s)，ego 速度 - adv 速度"""
    return speed_ms(ego) - speed_ms(adv)


# ================================================================
# 角度与方向
# ================================================================

def heading_vector(vehicle):
    """获取车辆前进方向的单位向量"""
    t = vehicle.get_transform()
    forward = t.get_forward_vector()
    return np.array([forward.x, forward.y])


def heading_angle(vehicle):
    """获取车辆朝向角 (弧度, 相对于 x 轴正方向)"""
    hv = heading_vector(vehicle)
    return math.atan2(hv[1], hv[0])


def angle_between(v1, v2):
    """两个二维向量之间的夹角 (弧度)"""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-10:
        return 0.0
    cos = max(-1.0, min(1.0, dot / norm))
    return math.acos(cos)


def is_facing(vehicle, target_location, angle_threshold_deg=30):
    """
    判断车辆是否大致朝向目标点。

    angle_threshold_deg: 角度阈值 (度)
    """
    hv = heading_vector(vehicle)
    v_loc = vehicle.get_location()
    to_target = np.array([
        target_location.x - v_loc.x,
        target_location.y - v_loc.y,
    ])
    ang = angle_between(hv, to_target)
    return ang < math.radians(angle_threshold_deg)


# ================================================================
# 车道相关
# ================================================================

def get_lane_offset(vehicle, world):
    """
    计算车辆偏离当前车道中心线的横向距离 (m)。

    返回: float，正值表示右偏，负值表示左偏（取决于路点方向）。
    """
    waypoint = world.get_map().get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return 0.0

    wp_loc = waypoint.transform.location
    ego_loc = vehicle.get_location()

    # 横向偏移：计算 ego 到 waypoint 在垂直车道方向上的投影
    wp_forward = waypoint.transform.get_forward_vector()
    dx = ego_loc.x - wp_loc.x
    dy = ego_loc.y - wp_loc.y

    # 叉积得到侧向分量
    lateral = dx * (-wp_forward.y) + dy * wp_forward.x
    return lateral


def get_current_lane_id(vehicle, world):
    """获取当前车道 ID"""
    waypoint = world.get_map().get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return -1
    return waypoint.lane_id


def is_same_lane(ego_vehicle, adv_vehicle, world):
    """判断两车是否在同一车道"""
    return (get_current_lane_id(ego_vehicle, world)
            == get_current_lane_id(adv_vehicle, world))


# ================================================================
# 轨迹偏差
# ================================================================

def trajectory_deviation(actual_path, reference_path):
    """
    计算实际轨迹与参考轨迹之间的均方根误差 (RMSE)。

    actual_path: list[carla.Location]  实际路径
    reference_path: list[carla.Location]  参考路径
    """
    n = min(len(actual_path), len(reference_path))
    if n == 0:
        return 0.0

    errors = [
        distance_2d(actual_path[i], reference_path[i])
        for i in range(n)
    ]
    return math.sqrt(np.mean(np.square(errors)))


# ================================================================
# 包围盒
# ================================================================

def get_vehicle_bbox(vehicle):
    """获取车辆 2D 包围盒（简化表示为四个角点）"""
    transform = vehicle.get_transform()
    bbox = vehicle.bounding_box
    corners = [
        carla.Location(x=bbox.extent.x, y=bbox.extent.y),
        carla.Location(x=bbox.extent.x, y=-bbox.extent.y),
        carla.Location(x=-bbox.extent.x, y=-bbox.extent.y),
        carla.Location(x=-bbox.extent.x, y=bbox.extent.y),
    ]
    world_corners = []
    for corner in corners:
        world_corners.append(transform.transform(corner))
    return world_corners


def bbox_overlap(ego_vehicle, adv_vehicle):
    """
    简化包围盒重叠检测（SAT 分离轴定理，2D）。

    返回 True 表示两车包围盒有重叠。
    """
    bbox1 = get_vehicle_bbox(ego_vehicle)
    bbox2 = get_vehicle_bbox(adv_vehicle)

    # 获取两个包围盒在各轴上的投影
    def project(corners, axis):
        dots = [c.x * axis[0] + c.y * axis[1] for c in corners]
        return min(dots), max(dots)

    def get_axes(corners):
        axes = []
        n = len(corners)
        for i in range(n):
            e1 = corners[i]
            e2 = corners[(i + 1) % n]
            edge = (e2.x - e1.x, e2.y - e1.y)
            # 法向量
            axes.append((-edge[1], edge[0]))
        return axes

    axes = get_axes(bbox1) + get_axes(bbox2)
    for axis in axes:
        min1, max1 = project(bbox1, axis)
        min2, max2 = project(bbox2, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True


# ================================================================
# 坐标变换
# ================================================================

def to_local_coordinates(world_loc, reference_transform):
    """将世界坐标转为参考坐标系下的局部坐标"""
    dx = world_loc.x - reference_transform.location.x
    dy = world_loc.y - reference_transform.location.y

    rot = reference_transform.rotation.yaw
    rad = math.radians(rot)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    local_x = dx * cos_r + dy * sin_r
    local_y = -dx * sin_r + dy * cos_r
    return local_x, local_y


def to_world_coordinates(local_x, local_y, reference_transform):
    """将局部坐标转回世界坐标"""
    rot = reference_transform.rotation.yaw
    rad = math.radians(rot)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    dx = local_x * cos_r - local_y * sin_r
    dy = local_x * sin_r + local_y * cos_r

    return carla.Location(
        x=reference_transform.location.x + dx,
        y=reference_transform.location.y + dy,
        z=reference_transform.location.z,
    )
