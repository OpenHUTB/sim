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
revert_flag = False

while True:
    world.wait_for_tick()
    control = carla.WalkerControl()
    control.direction.x = 0
    control.direction.z = 0
    control.speed = 2.0
    # 往返路径
    if (pedestrain.get_location().y > 7.7):
        revert_flag = True
    if (pedestrain.get_location().y < -7.0):
        revert_flag = False
    if (revert_flag):
        control.direction.y = -1
    else:
        control.direction.y = 1
    pedestrain.apply_control(control)