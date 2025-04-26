import json
import carla

# 读取生成的场景数据
with open("D:/sceneMain/chatScene/demo/generated_scenes/generated_dynamic_scene.json", "r") as f:
    scene_data = json.load(f)

# 连接到 Carla
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# 获取蓝图库
blueprints = world.get_blueprint_library()

# 根据场景数据生成车辆
for vehicle in scene_data["vehicles"]:
    vehicle_type = vehicle["type"]

    # 查找匹配的蓝图
    blueprint = blueprints.filter(vehicle_type)

    if not blueprint:
        # 如果没有匹配的蓝图，打印警告并设置默认蓝图
        print(f"Warning: No blueprint found for vehicle type '{vehicle_type}', using default.")
        blueprint = blueprints.filter('vehicle.audi.a2')  # 使用一个常见的蓝图作为默认

    blueprint = blueprint[0]  # 获取蓝图并生成车辆

    # 设置生成车辆的坐标
    spawn_point = carla.Transform(
        carla.Location(x=vehicle["x"], y=vehicle["y"], z=0.5),
        carla.Rotation(yaw=vehicle["yaw"])
    )
    
    # 生成车辆
    world.spawn_actor(blueprint, spawn_point)
