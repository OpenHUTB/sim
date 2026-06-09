# 【终极版：任何环境都能跑，专门给你的新CARLA】
import sys
import os
import time

# ======================
# 1. 强制添加路径（核心！）
# ======================
sys.path.insert(0, r"D:\hutb\hutb\PythonAPI")
sys.path.insert(0, r"D:\hutb\hutb\PythonAPI\carla\dist")

# ======================
# 2. 导入CARLA（现在一定能找到）
# ======================
try:
    import carla
    print("✅ 成功导入 carla")
except:
    print("❌ 导入失败")
    exit()

# ======================
# 3. 连接模拟器
# ======================
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

print(f"✅ 当前地图: {world.get_map().name}")

# ======================
# 4. 生成车辆 + 跟随视角
# ======================
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

vehicle = world.spawn_actor(
    bp_lib.filter("vehicle.tesla.model3")[0],
    spawn_points[0]
)

# 视角跟随
spectator = world.get_spectator()
def follow():
    trans = vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        trans.location + carla.Location(z=4),
        carla.Rotation(pitch=-25)
    ))

follow()
vehicle.set_autopilot(True)

print("✅ 车辆开始自动行驶")

# ======================
# 5. 运行
# ======================
try:
    for _ in range(20):
        follow()
        time.sleep(1)
finally:
    vehicle.destroy()
    print("\n✅ 测试完成")