import carla
import random
import time

def parse_description(desc):
    vehicle_color = "0,0,255"  # 默认蓝色
    if "红色" in desc:
        vehicle_color = "255,0,0"
    elif "白色" in desc:
        vehicle_color = "255,255,255"
    elif "绿色" in desc:
        vehicle_color = "0,255,0"

    need_pedestrian = "行人" in desc or "过马路" in desc or "人" in desc

    return vehicle_color, need_pedestrian

def spawn_vehicle(world, color):
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    bp.set_attribute('color', color)

    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(bp, spawn_point)
    if vehicle:
        print("✅ Vehicle spawned at:", spawn_point.location)
    return vehicle

def spawn_pedestrian(world):
    blueprint_library = world.get_blueprint_library()
    walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))

    # 尝试从人行道上生成
    sidewalks = [wp for wp in world.get_map().generate_waypoints(2.0)
                 if wp.lane_type == carla.LaneType.Sidewalk]

    pedestrian = None
    if sidewalks:
        sidewalk_wp = random.choice(sidewalks)
        pedestrian = world.try_spawn_actor(walker_bp, sidewalk_wp.transform)
        if pedestrian:
            print(f"✅ Pedestrian spawned on sidewalk at: {sidewalk_wp.transform.location}")
            return pedestrian

    # 如果失败，尝试使用导航点
    nav_point = world.get_random_location_from_navigation()
    if nav_point:
        pedestrian = world.try_spawn_actor(walker_bp, carla.Transform(nav_point))
        if pedestrian:
            print(f"✅ Pedestrian spawned at random nav location: {nav_point}")
            return pedestrian

    print("❌ Failed to spawn pedestrian.")
    return None

def clear_actors(world, actor_list):
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    print("✅ 清理完成")

def main():
    desc = "一辆蓝色的车在城镇中间的道路上行驶，前面有一个行人过马路"
    print(f"使用描述：{desc}")

    vehicle_color, need_pedestrian = parse_description(desc)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    vehicle = spawn_vehicle(world, vehicle_color)

    pedestrian = None
    if need_pedestrian:
        pedestrian = spawn_pedestrian(world)

    time.sleep(10)
    clear_actors(world, [vehicle, pedestrian])

if __name__ == '__main__':
    main()
