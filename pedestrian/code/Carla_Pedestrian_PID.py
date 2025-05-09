import carla
import numpy as np
import random
import time
import threading
import cv2


class PIDController:
    """PID控制器实现"""

    def __init__(self, kp, ki, kd, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, -self.max_output, self.max_output)
        self.previous_error = error
        return output


class PIDPedestrianEnv:
    """基于PID控制的行人环境"""

    def __init__(self, target_location=carla.Location(x=180, y=120, z=1)):
        # ==== 初始化关键属性 ====
        self.last_lidar_points = np.array([])  # 初始化为空数组
        self.min_obstacle_distance = 10.0  # 初始默认值更大
        self.target_location = target_location
        self.collision_occurred = False
        self.obstacle_clusters = []
        self.current_speed = 0.0
        self.sensors = []
        self.dynamic_obstacles = []
        self.last_display = time.time()
        self.img_lock = threading.Lock()
        self.lidar_lock = threading.Lock()
        # 初始化方向向量
        self.target_direction = carla.Vector3D(0, 0, 0)
        self.avoidance_direction = carla.Vector3D(0, 0, 0)
        self.combined_direction = carla.Vector3D(0, 0, 0)
        self.obstacle_detected = False

        # ==== PID控制器 ====
        self.steering_pid = PIDController(kp=0.8, ki=0.001, kd=0.2, max_output=30)
        self.speed_pid = PIDController(kp=0.5, ki=0.01, kd=0.1, max_output=3.0)

        # ==== Carla连接配置 ====
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(30.0)
        self._connect_to_server()

        # ==== 初始化Carla组件 ====
        self._preload_assets()
        self._setup_spectator()
        self._spawn_pedestrian()
        self._attach_sensors()
        self._spawn_dynamic_obstacles(num_vehicles=2, num_walkers=1)

    def _connect_to_server(self):
        """强制使用Town01地图并设置良好的天气"""
        for retry in range(5):
            try:
                self.world = self.client.load_world("Town01")
                if "Town01" in self.world.get_map().name:
                    # 设置更好的天气
                    weather = carla.WeatherParameters(
                        cloudiness=10.0,
                        precipitation=0.0,
                        sun_altitude_angle=70.0,  # 较高的太阳角度
                        fog_density=0.0,
                        wetness=0.0
                    )
                    self.world.set_weather(weather)
                    print("成功连接Carla服务器")
                    return
            except Exception as e:
                print(f"连接失败（尝试 {retry + 1}/5）: {str(e)}")
                time.sleep(2)
        raise ConnectionError("无法连接到Carla服务器")

    def _preload_assets(self):
        """预加载蓝图"""
        self.blueprint_library = self.world.get_blueprint_library()
        self.walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        self.controller_bp = self.blueprint_library.find('controller.ai.walker')
        self.vehicle_bps = self.blueprint_library.filter('vehicle.*')
        self.collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.lidar_bp = self._configure_lidar()
        self.camera_bp = self._configure_camera()

    def _configure_lidar(self):
        lidar = self.blueprint_library.find('sensor.lidar.ray_cast')
        # 增加检测范围和精度
        lidar.set_attribute('range', '15.0')  # 增加检测范围
        lidar.set_attribute('points_per_second', '50000')  # 增加点云密度
        lidar.set_attribute('rotation_frequency', '20')  # 增加旋转频率
        lidar.set_attribute('upper_fov', '10')  # 上视场角
        lidar.set_attribute('lower_fov', '-30')  # 下视场角，增加对地面和低障碍物的检测
        lidar.set_attribute('channels', '64')  # 增加通道数提高分辨率
        return lidar

    def _configure_camera(self):
        camera = self.blueprint_library.find('sensor.camera.rgb')
        # 提高分辨率到1280x720（高清）
        camera.set_attribute('image_size_x', '1280')
        camera.set_attribute('image_size_y', '720')
        # 增加视野范围
        camera.set_attribute('fov', '100')
        return camera

    def _setup_spectator(self):
        try:
            spectator = self.world.get_spectator()
            transform = carla.Transform(
                carla.Location(x=160, y=138, z=50),
                carla.Rotation(pitch=-90)
            )
            spectator.set_transform(transform)
        except Exception as e:
            print(f"视角设置失败: {str(e)}")

    def _spawn_pedestrian(self):
        """生成受控行人"""
        for _ in range(3):
            try:
                spawn_point = carla.Transform(
                    carla.Location(x=160, y=138, z=1.0),
                    carla.Rotation(yaw=random.randint(0, 360))
                )
                self.pedestrian = self.world.spawn_actor(
                    random.choice(self.walker_bps),
                    spawn_point)
                break
            except Exception as e:
                print(f"行人生成失败: {str(e)}")
                time.sleep(0.5)

    def _attach_sensors(self):
        """附加传感器"""
        try:
            # 碰撞传感器
            collision_sensor = self.world.spawn_actor(
                self.collision_bp,
                carla.Transform(),
                attach_to=self.pedestrian)
            collision_sensor.listen(lambda event: self._on_collision(event))

            # 激光雷达
            lidar_transform = carla.Transform(carla.Location(z=2.5))
            lidar_sensor = self.world.spawn_actor(
                self.lidar_bp,
                lidar_transform,
                attach_to=self.pedestrian)
            lidar_sensor.listen(lambda data: self._process_lidar(data))

            # 摄像头 - 调整位置和角度
            camera_transform = carla.Transform(
                carla.Location(x=0.5, z=1.6),  # 更接近真实人眼位置
                carla.Rotation(pitch=-5)  # 稍微向下的角度，更自然
            )
            camera_sensor = self.world.spawn_actor(
                self.camera_bp,
                camera_transform,
                attach_to=self.pedestrian)
            camera_sensor.listen(lambda image: self._process_image(image))

            self.sensors = [collision_sensor, lidar_sensor, camera_sensor]
        except Exception as e:
            print(f"传感器初始化失败: {str(e)}")
            self._cleanup_actors()
            raise

    def _spawn_dynamic_obstacles(self, num_vehicles, num_walkers):
        """生成动态障碍物"""
        vehicle_spawn_points = [
            p for p in self.world.get_map().get_spawn_points()
            if p.location.distance(self.pedestrian.get_location()) > 20.0
        ]
        for _ in range(num_vehicles):
            try:
                vehicle = self.world.spawn_actor(
                    random.choice(self.vehicle_bps),
                    random.choice(vehicle_spawn_points))
                vehicle.set_autopilot(True)
                self.dynamic_obstacles.append(vehicle)
            except:
                print("生成车辆失败，继续尝试")

        for _ in range(num_walkers):
            try:
                walker = self.world.spawn_actor(
                    random.choice(self.walker_bps),
                    carla.Transform(self._random_destination()))
                controller = self.world.spawn_actor(
                    self.controller_bp,
                    carla.Transform(),
                    attach_to=walker)
                controller.start()
                controller.go_to_location(self._random_destination())
                self.dynamic_obstacles.extend([walker, controller])
            except:
                print("生成行人失败，继续尝试")

    def _random_destination(self):
        return carla.Location(
            x=random.uniform(120, 200),
            y=random.uniform(100, 160),
            z=1.0)

    def _on_collision(self, event):
        self.collision_occurred = True

    def _process_lidar(self, data):
        try:
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
            with self.lidar_lock:
                if len(points) > 0:
                    # 过滤出前方扇区的点 (±60度范围内)
                    front_points = []
                    for point in points:
                        # 只考虑xy平面
                        x, y = point[0], point[1]
                        distance = np.sqrt(x ** 2 + y ** 2)
                        # 计算方位角 (弧度)
                        angle = np.arctan2(y, x)
                        # 转换为度数并取绝对值
                        angle_deg = np.abs(np.degrees(angle))
                        # 如果在前方扇区内，加入到前方点集
                        if angle_deg <= 60 and distance <= 8.0:
                            front_points.append(point)

                    front_points = np.array(front_points)

                    if len(front_points) > 0:
                        # 计算最小距离
                        distances = np.sqrt(front_points[:, 0] ** 2 + front_points[:, 1] ** 2)
                        self.min_obstacle_distance = np.min(distances)

                        # 记录点云位置
                        self.last_lidar_points = front_points

                        # 尝试对点云进行简单聚类，识别障碍物
                        self.obstacle_clusters = self._cluster_points(front_points[:, :2])
                    else:
                        self.min_obstacle_distance = 10.0
                        self.obstacle_clusters = []
                else:
                    self.min_obstacle_distance = 10.0
                    self.obstacle_clusters = []
        except Exception as e:
            print(f"激光雷达处理错误: {str(e)}")

    def _cluster_points(self, points, eps=0.5, min_samples=5):
        """简单聚类算法，返回障碍物聚类中心点列表"""
        if len(points) < min_samples:
            return []

        clusters = []
        # 简化版聚类 - 实际应用中可以使用DBSCAN等算法
        visited = set()

        for i, point in enumerate(points):
            if i in visited:
                continue

            cluster = []
            self._expand_cluster(points, i, cluster, visited, eps)

            if len(cluster) >= min_samples:
                # 计算聚类中心
                cluster_center = np.mean(np.array([points[j] for j in cluster]), axis=0)
                cluster_size = len(cluster)
                clusters.append((cluster_center, cluster_size))

        return clusters

    def _expand_cluster(self, points, point_idx, cluster, visited, eps):
        """辅助聚类函数"""
        visited.add(point_idx)
        cluster.append(point_idx)

        for i, other_point in enumerate(points):
            if i in visited:
                continue

            if np.linalg.norm(points[point_idx] - other_point) <= eps:
                self._expand_cluster(points, i, cluster, visited, eps)

    def _process_image(self, image):
        with self.img_lock:
            try:
                if time.time() - self.last_display < 0.1:
                    return

                # 解析图像数据
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))

                # 提取RGB图像并转换为BGR（OpenCV格式）
                img_bgr = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)

                # 应用图像增强
                # 1. 锐化图像
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                img_bgr = cv2.filter2D(img_bgr, -1, kernel)

                # 2. 增加亮度和对比度（如果画面太暗）
                alpha = 1.1  # 对比度因子
                beta = 5  # 亮度增加值
                img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

                # 3. 添加信息文本
                cv2.putText(img_bgr, f"Speed: {self.current_speed:.1f}m/s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # 显示障碍物距离，颜色随距离变化
                if self.min_obstacle_distance < 1.0:
                    obstacle_color = (0, 0, 255)  # 红色 - 危险
                elif self.min_obstacle_distance < 3.0:
                    obstacle_color = (0, 165, 255)  # 橙色 - 警告
                else:
                    obstacle_color = (0, 255, 0)  # 绿色 - 安全

                cv2.putText(img_bgr, f"Obstacle: {self.min_obstacle_distance:.1f}m",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, obstacle_color, 2)

                # 绘制避障指示
                h, w = img_bgr.shape[:2]
                center_x, center_y = w // 2, h // 2

                # 绘制目标方向和避障方向向量
                if hasattr(self, 'target_direction') and hasattr(self, 'avoidance_direction'):
                    # 目标方向 (绿色)
                    target_dir = self.target_direction
                    target_x = int(center_x + target_dir.x * 100)
                    target_y = int(center_y - target_dir.y * 100)  # 注意Y轴翻转
                    cv2.arrowedLine(img_bgr, (center_x, center_y), (target_x, target_y), (0, 255, 0), 2)

                    # 避障方向 (红色) - 仅当检测到障碍物时绘制
                    if hasattr(self, 'obstacle_detected') and self.obstacle_detected:
                        avoid_dir = self.avoidance_direction
                        avoid_x = int(center_x + avoid_dir.x * 100)
                        avoid_y = int(center_y - avoid_dir.y * 100)
                        cv2.arrowedLine(img_bgr, (center_x, center_y), (avoid_x, avoid_y), (0, 0, 255), 2)

                    # 实际前进方向 (蓝色)
                    if hasattr(self, 'combined_direction'):
                        combined_dir = self.combined_direction
                        combined_x = int(center_x + combined_dir.x * 100)
                        combined_y = int(center_y - combined_dir.y * 100)
                        cv2.arrowedLine(img_bgr, (center_x, center_y), (combined_x, combined_y), (255, 0, 0), 3)

                # 绘制障碍物警告
                if self.min_obstacle_distance < 5.0:
                    warning_radius = int(80 * (1 - self.min_obstacle_distance / 5.0))
                    cv2.circle(img_bgr, (center_x, center_y), warning_radius, obstacle_color, 2)

                    if self.min_obstacle_distance < 1.5:
                        # 添加危险警告文本
                        cv2.putText(img_bgr, "OBSTACLE WARNING!",
                                    (center_x - 120, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # 4. 显示无抗锯齿缩放图像
                display_img = cv2.resize(img_bgr, (960, 540), interpolation=cv2.INTER_LANCZOS4)
                cv2.imshow('Pedestrian View (HD)', display_img)
                cv2.waitKey(1)
                self.last_display = time.time()
            except Exception as e:
                print(f"图像处理错误: {str(e)}")

    def calculate_errors(self):
        """计算控制误差，加入高级避障逻辑"""
        transform = self.pedestrian.get_transform()
        current_loc = transform.location

        # 基本目标方向
        target_vector = self.target_location - current_loc
        target_distance = target_vector.length()

        # 目标单位向量
        if target_distance > 0:
            target_dir = target_vector / target_distance
        else:
            target_dir = carla.Vector3D(0, 0, 0)

        # ==== 更强大的避障逻辑 ====
        avoidance_vector = carla.Vector3D(0, 0, 0)

        # 记录是否检测到障碍物
        obstacle_detected = False

        # 如果检测到障碍物，尝试找到最佳绕行方向
        with self.lidar_lock:
            if self.min_obstacle_distance < 5.0:
                obstacle_detected = True

                # 1. 先检查是否有聚类障碍物
                if hasattr(self, 'obstacle_clusters') and len(self.obstacle_clusters) > 0:
                    # 使用障碍物聚类信息
                    avoid_vectors = []
                    weights = []

                    for cluster_center, cluster_size in self.obstacle_clusters:
                        # 转换为carla向量
                        obstacle_vector = carla.Vector3D(float(cluster_center[0]),
                                                         float(cluster_center[1]), 0)

                        # 计算障碍物距离
                        distance = obstacle_vector.length()

                        if distance < 0.1:  # 避免除零
                            continue

                        # 方向为远离障碍物
                        avoid_dir = obstacle_vector / -distance  # 负号表示远离

                        # 权重与障碍物大小和距离成反比
                        weight = cluster_size / (distance ** 2)

                        avoid_vectors.append(avoid_dir)
                        weights.append(weight)

                    # 如果有避障向量，计算加权平均
                    if avoid_vectors:
                        total_weight = sum(weights)
                        if total_weight > 0:
                            avoidance_vector = carla.Vector3D(0, 0, 0)
                            for vec, weight in zip(avoid_vectors, weights):
                                avoidance_vector.x += vec.x * weight / total_weight
                                avoidance_vector.y += vec.y * weight / total_weight

                            # 正则化向量
                            length = avoidance_vector.length()
                            if length > 0:
                                avoidance_vector = avoidance_vector / length

                # 2. 如果没有聚类或聚类处理失败，使用原始点云
                if avoidance_vector.length() < 0.1 and hasattr(self, 'last_lidar_points'):
                    lidar_points = self.last_lidar_points

                    if len(lidar_points) > 0:
                        # 根据点云计算避障向量
                        avoid_x, avoid_y = 0, 0
                        total_weight = 0

                        for point in lidar_points:
                            x, y = point[0], point[1]
                            dist = np.sqrt(x * x + y * y)

                            if dist < 0.1:  # 避免除零
                                continue

                            # 权重与距离成反比
                            weight = 1.0 / (dist * dist)

                            # 方向为远离障碍物
                            avoid_x -= x * weight / dist
                            avoid_y -= y * weight / dist
                            total_weight += weight

                        if total_weight > 0:
                            avoidance_vector = carla.Vector3D(
                                avoid_x / total_weight,
                                avoid_y / total_weight,
                                0
                            )

                            # 正则化向量
                            length = avoidance_vector.length()
                            if length > 0:
                                avoidance_vector = avoidance_vector / length

        # ==== 避障与目标向量融合 ====
        combined_dir = carla.Vector3D(0, 0, 0)

        if obstacle_detected:
            # 1. 检查目标方向与避障方向的一致性
            dot_product = target_dir.x * avoidance_vector.x + target_dir.y * avoidance_vector.y

            # 2. 设置避障权重 - 离障碍物越近，避障权重越大
            # 在5米内线性增加避障权重
            avoidance_weight = min(0.95, max(0.5, 2.0 * (1.0 - self.min_obstacle_distance / 5.0)))

            # 如果方向一致性低，增加避障权重
            if dot_product < 0:
                avoidance_weight = min(0.98, avoidance_weight + 0.3)

            # 3. 设置目标权重
            target_weight = 1.0 - avoidance_weight

            # 4. 如果极近障碍物，几乎完全依赖避障向量
            if self.min_obstacle_distance < 1.0:
                avoidance_weight = 0.95
                target_weight = 0.05

            # 5. 组合向量
            combined_dir.x = target_dir.x * target_weight + avoidance_vector.x * avoidance_weight
            combined_dir.y = target_dir.y * target_weight + avoidance_vector.y * avoidance_weight
        else:
            # 没有障碍物，使用目标方向
            combined_dir = target_dir

        # 正则化向量
        combined_length = combined_dir.length()
        if combined_length > 0:
            combined_dir = combined_dir / combined_length

        # 计算局部坐标系下的误差
        yaw = np.radians(transform.rotation.yaw)
        rotation_matrix = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])

        local_target = rotation_matrix @ np.array([combined_dir.x, combined_dir.y])

        # 保存诊断信息以显示
        self.target_direction = target_dir
        self.avoidance_direction = avoidance_vector
        self.combined_direction = combined_dir
        self.obstacle_detected = obstacle_detected

        return {
            'distance': target_distance,
            'lateral_error': local_target[1],  # 横向误差
            'longitudinal_error': local_target[0],  # 纵向误差
            'obstacle_detected': obstacle_detected
        }

    def run_control_loop(self):
        """主控制循环"""
        last_time = time.time()
        try:
            while True:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

                # 计算控制误差
                errors = self.calculate_errors()

                # 横向控制（转向）- 添加更激进的避障转向
                steering = self.steering_pid.compute(errors['lateral_error'], dt)

                # 如果检测到障碍物，增强转向反应
                if errors.get('obstacle_detected', False) and self.min_obstacle_distance < 2.0:
                    # 在非常接近障碍物时增强转向
                    steering *= 1.5

                # 纵向控制（速度）
                # 平滑的速度控制，确保在接近障碍物时明显减速
                if self.min_obstacle_distance < 5.0:
                    # 使用平方曲线使速度更平滑地降低
                    safe_speed = max(0.01, (self.min_obstacle_distance / 5.0) ** 1.5 * 3.0)

                    # 在极近障碍物时几乎停止
                    if self.min_obstacle_distance < 1.0:
                        safe_speed = max(0.01, self.min_obstacle_distance / 4.0)
                else:
                    safe_speed = 3.0

                # 如果发生碰撞，后退并尝试绕行
                if self.collision_occurred:
                    print("检测到碰撞，准备后退并绕行！")
                    # 后退
                    self._backup_and_turn()
                    self.collision_occurred = False
                    continue

                speed_error = safe_speed - self.current_speed
                throttle = self.speed_pid.compute(speed_error, dt)

                # 应用控制
                self.apply_control(steering, throttle)
                self.world.tick()

                # 检查终止条件 - 仅在到达目标时结束
                if errors['distance'] < 2.0:
                    print("成功到达目标！")
                    break

                # 添加一个小延时，减轻CPU负担
                time.sleep(0.01)

        finally:
            self._cleanup_actors()
            cv2.destroyAllWindows()

    def _backup_and_turn(self):
        """碰撞后后退并转向"""
        print("执行后退和转向...")

        # 后退
        for i in range(10):
            # 获取当前朝向的反方向
            forward_vector = self.pedestrian.get_transform().get_forward_vector()
            backward_vector = carla.Vector3D(-forward_vector.x, -forward_vector.y, 0)

            # 应用后退控制
            control = carla.WalkerControl(
                direction=backward_vector,
                speed=1.0)
            self.pedestrian.apply_control(control)
            self.world.tick()
            time.sleep(0.05)

        # 随机选择一个新的转向角度 (左或右)
        turn_angle = random.choice([-90, 90])
        current_yaw = self.pedestrian.get_transform().rotation.yaw

        # 应用转向
        for i in range(5):
            self.pedestrian.set_transform(carla.Transform(
                self.pedestrian.get_location(),
                carla.Rotation(yaw=current_yaw + turn_angle * (i + 1) / 5)
            ))
            self.world.tick()
            time.sleep(0.05)

        # 重置PID控制器，避免积分项影响
        self.steering_pid.reset()
        self.speed_pid.reset()

    def apply_control(self, steering, throttle):
        """应用控制指令"""
        try:
            steering = np.clip(steering, -30, 30)
            speed = np.clip(throttle, 0, 3.0)
            self.current_speed = speed

            # 更新方向 - 使用更安全的方式
            current_transform = self.pedestrian.get_transform()
            new_yaw = current_transform.rotation.yaw + steering * 0.1  # 平滑转向

            self.pedestrian.set_transform(carla.Transform(
                current_transform.location,
                carla.Rotation(yaw=new_yaw)
            ))

            # 应用速度 - 使用前进方向向量
            forward_vector = self.pedestrian.get_transform().get_forward_vector()
            control = carla.WalkerControl(
                direction=forward_vector,
                speed=speed)
            self.pedestrian.apply_control(control)
        except Exception as e:
            print(f"控制应用错误: {str(e)}")

    def _cleanup_actors(self):
        """清理资源"""
        destroy_list = []
        if hasattr(self, 'pedestrian') and self.pedestrian.is_alive:
            destroy_list.append(self.pedestrian)
        for sensor in self.sensors:
            if sensor.is_alive:
                destroy_list.append(sensor)
        for obstacle in self.dynamic_obstacles:
            if obstacle.is_alive:
                destroy_list.append(obstacle)
        if destroy_list:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in destroy_list])
            time.sleep(0.5)

if __name__ == "__main__":
    try:
        env = PIDPedestrianEnv()
        env.run_control_loop()
    except Exception as e:
        print(f"程序运行错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序已结束")
        cv2.destroyAllWindows()