# coding:utf-8 
import carla
import random

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
world = client.load_world('Town05')
pedestrain_blueprints = world.get_blueprint_library().filter("walker.pedestrian.0001")
# 设置行人起点
pedestrain = world.try_spawn_actor(random.choice(pedestrain_blueprints),
                                   carla.Transform(carla.Location(x=19, y=9, z=2), carla.Rotation(yaw=-90)))
pedestrain_control = carla.WalkerControl()
# 设置行人速度
pedestrain_control.speed = 2.0
pedestrain_rotation = carla.Rotation(0, -90, 0)
pedestrain_control.direction = pedestrain_rotation.get_forward_vector()
pedestrain.apply_control(pedestrain_control)

while True:
    # 设置终点条件
    if (pedestrain.get_location().y < -6.8):
        control = carla.WalkerControl()
        control.direction.x = 0
        control.direction.z = 0
        control.direction.y = 0
        pedestrain.apply_control(control)
        print("finish")
        break