# 【CARLA 连接测试 —— 通过环境变量 CARLA_ROOT 定位引擎】
import sys
import os
import time

# ======================
# 1. 通过环境变量添加 CARLA 路径
# ======================
carla_root = os.environ.get("CARLA_ROOT")
if not carla_root:
    print("❌ 未设置 CARLA_ROOT 环境变量")
    print("   请设置: set CARLA_ROOT=D:\\hutb\\hutb")
    exit(1)

sys.path.insert(0, os.path.join(carla_root, "PythonAPI"))
sys.path.insert(0, os.path.join(carla_root, "PythonAPI", "carla", "dist"))

# ======================
# 2. 导入CARLA
# ======================
try:
    import carla
    print("✅ 成功导入 carla")
except ImportError:
    print("❌ 导入失败，请检查 CARLA_ROOT 路径是否正确: %s" % carla_root)
    exit(1)

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