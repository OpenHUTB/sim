import carla
import pygame
import sys
import numpy as np
import time
import random
import math

# 初始化 pygame
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CARLA 行人控制 - 完整版")
clock = pygame.time.Clock()


class ObstacleDetector:
    def __init__(self, world, parent_actor, detection_range=5.0):
        self.world = world
        self.parent = parent_actor
        self.detection_range = detection_range
        self.obstacles = []

        # 碰撞传感器
        self.collision_sensor = self._setup_collision_sensor()

    def _setup_collision_sensor(self):
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        transform = carla.Transform(carla.Location(x=0.5, z=0.5))
        sensor = self.world.spawn_actor(blueprint, transform, attach_to=self.parent)
        sensor.listen(self._on_collision)
        return sensor

    def _on_collision(self, event):
        print(f"! 碰撞发生: {event.other_actor.type_id}")

    def update(self):
        self.obstacles = []
        parent_location = self.parent.get_location()

        # 检测车辆
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.id != self.parent.id:
                distance = self._calculate_distance(parent_location, vehicle.get_location())
                if distance < self.detection_range:
                    self.obstacles.append({
                        'type': '车辆',
                        'distance': distance,
                        'actor': vehicle
                    })

        # 检测行人
        for walker in self.world.get_actors().filter('walker.*'):
            if walker.id != self.parent.id:
                distance = self._calculate_distance(parent_location, walker.get_location())
                if distance < self.detection_range:
                    self.obstacles.append({
                        'type': '行人',
                        'distance': distance,
                        'actor': walker
                    })

    def _calculate_distance(self, loc1, loc2):
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        dz = loc1.z - loc2.z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def get_nearest_obstacle(self):
        return min(self.obstacles, key=lambda x: x['distance']) if self.obstacles else None


class CarlaSimulation:
    def __init__(self):
        # 连接CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town01')
        self.original_settings = self.world.get_settings()

        # 同步模式设置
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)

        # 初始化对象
        self.walker = None
        self.camera = None
        self.ai_walkers = []
        self.vehicles = []

        # 初始化系统
        self._init_main_walker()
        self._init_camera()
        self._spawn_ai_walkers(10)  # 减少AI行人数量
        self._spawn_vehicles(10)  # 减少车辆数量
        self.obstacle_detector = ObstacleDetector(self.world, self.walker)

        # 使用中文字体
        self.font = pygame.font.Font("SimHei.ttf", 36)  # 请确保 SimHei.ttf 字体文件存在

    def _init_main_walker(self):
        walker_bp = self.world.get_blueprint_library().find('walker.pedestrian.0001')
        spawn_point = carla.Transform(carla.Location(x=30, y=3, z=0.5))

        for _ in range(5):
            self.walker = self.world.try_spawn_actor(walker_bp, spawn_point)
            if self.walker:
                break
            time.sleep(0.5)

        if not self.walker:
            raise RuntimeError("主行人生成失败")

    def _init_camera(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WIDTH))
        camera_bp.set_attribute('image_size_y', str(HEIGHT))

        # 修正的摄像头位置和旋转参数
        self.camera = self.world.spawn_actor(
            camera_bp,
            carla.Transform(
                carla.Location(x=-5, z=2.5),  # 位置
                carla.Rotation(pitch=-10)  # 旋转
            ),
            attach_to=self.walker
        )

        # 图像处理回调
        self.image_surface = None

        def process_image(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((HEIGHT, WIDTH, 4))[:, :, :3]
            array = array[:, :, ::-1]  # BGR转RGB
            self.image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        self.camera.listen(process_image)

    def _spawn_ai_walkers(self, count):
        for _ in range(count):
            actor = self._create_ai_walker()
            if actor:
                self.ai_walkers.append(actor)

    def _create_ai_walker(self):
        try:
            spawn_loc = self.world.get_random_location_from_navigation()
            if not spawn_loc:
                return None

            walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.pedestrian.*'))
            walker = self.world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc))
            if not walker:
                return None

            controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            return (controller, walker)
        except:
            return None

    def _spawn_vehicles(self, count):
        spawn_points = [p for p in self.world.get_map().get_spawn_points() if p.location.z > 0]
        vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')

        for _ in range(count):
            if not spawn_points:
                break
            spawn_point = random.choice(spawn_points)
            vehicle_bp = random.choice(vehicle_bps)

            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                self.vehicles.append(vehicle)
                spawn_points.remove(spawn_point)

    def run(self):
        print("控制说明：WASD移动 | 空格跳跃 | Shift加速 | ESC退出")
        try:
            while True:
                self.world.tick()

                # 处理退出事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        return

                # 获取控制输入
                keys = pygame.key.get_pressed()
                control = carla.WalkerControl()
                direction = carla.Vector3D()

                # 移动控制
                if keys[pygame.K_w]: direction.x += 1
                if keys[pygame.K_s]: direction.x -= 1
                if keys[pygame.K_a]: direction.y -= 1
                if keys[pygame.K_d]: direction.y += 1

                # 配置参数
                control.jump = keys[pygame.K_SPACE]
                control.speed = 3.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.5
                if direction.length() > 0:
                    control.direction = direction.make_unit_vector()

                # 障碍物检测
                self.obstacle_detector.update()
                nearest = self.obstacle_detector.get_nearest_obstacle()

                # 自动制动
                if nearest and nearest['distance'] < 2.0:
                    control.speed = 0
                    print(f"! 自动停止：检测到 {nearest['type']} ({nearest['distance']:.1f}m)")

                self.walker.apply_control(control)

                # 更新显示
                screen.fill((0, 0, 0))
                if self.image_surface:
                    screen.blit(self.image_surface, (0, 0))
                self._draw_hud()
                pygame.display.flip()

                clock.tick(30)

        finally:
            self.cleanup()

    def _draw_hud(self):
        y_pos = 20
        nearest = self.obstacle_detector.get_nearest_obstacle()

        if nearest:
            text = self.font.render(f"最近障碍物: {nearest['type']} 距离: {nearest['distance']:.2f}m", True,
                                    (255, 0, 0))
            screen.blit(text, (20, y_pos))
            y_pos += 40

            if nearest['distance'] < 3.0:
                warning = self.font.render("警告：前方有障碍物！", True, (255, 0, 0))
                screen.blit(warning, (WIDTH // 2 - 100, HEIGHT - 60))

    def cleanup(self):
        print("\n正在清理资源...")
        self.world.apply_settings(self.original_settings)

        # 清理主摄像头
        if self.camera and self.camera.is_alive:
            print("清理主摄像头...")
            self.camera.stop()
            self.camera.destroy()

        # 清理主行人
        if self.walker and self.walker.is_alive:
            print("清理主行人...")
            self.walker.destroy()

        # 清理AI行人
        for controller, walker in self.ai_walkers:
            try:
                if controller.is_alive:
                    print(f"清理AI行人控制器: {controller.id}")
                    controller.stop()
                    controller.destroy()
                if walker.is_alive:
                    print(f"清理AI行人: {walker.id}")
                    walker.destroy()
            except Exception as e:
                print(f"清理AI行人错误: {e}")

        # 清理车辆
        for vehicle in self.vehicles:
            try:
                if vehicle.is_alive:
                    print(f"清理车辆: {vehicle.id}")
                    vehicle.destroy()
            except Exception as e:
                print(f"清理车辆错误: {e}")

        # 清理传感器
        if hasattr(self.obstacle_detector, 'collision_sensor'):
            if self.obstacle_detector.collision_sensor.is_alive:
                print("清理碰撞传感器...")
                self.obstacle_detector.collision_sensor.destroy()

        # 等待资源释放
        for _ in range(10):
            self.world.tick()
            time.sleep(0.1)

        pygame.quit()


if __name__ == "__main__":
    try:
        sim = CarlaSimulation()
        sim.run()
    except Exception as e:
        print(f"运行错误: {str(e)}")
        sys.exit(1)
