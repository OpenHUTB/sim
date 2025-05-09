import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import pygame
from pathlib import Path
from queue import Queue
from torch.utils.data import Dataset, DataLoader
import carla

# 创建YOLOv5工作目录并克隆仓库
os.system('git clone https://github.com/ultralytics/yolov5')
os.chdir('yolov5')

# 安装依赖
os.system('pip install -r requirements.txt')

# 创建虚拟环境并安装核心依赖
os.system('python -m venv carla_env')
os.system('source carla_env/bin/activate')
os.system('pip install carla pygame numpy matplotlib')
os.system('pip install torch torchvision tensorboard')

# 启动Carla服务器
def launch_carla_server():
    os.system('./CarlaUE4.sh Town01 -windowed -ResX=800 -ResY=600')

# 连接Carla客户端
def connect_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return world

# 生成车辆
def spawn_vehicle(world):
    blueprint = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(blueprint, spawn_point)
    return vehicle

# 加载YOLOv5模型
def load_yolov5_model():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 根目录
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 加载预训练模型
    return model

# 处理图像并进行目标检测
def process_image(image, model):
    img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img_array = np.reshape(img_array, (image.height, image.width, 4))
    img_array = img_array[:, :, :3]  # 仅保留RGB通道
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 使用YOLOv5进行目标检测
    results = model(img)
    detections = results.pandas().xyxy[0]  # 结果转换为pandas DataFrame
    
    # 在 Carla 中绘制检测结果
    for _, detection in detections.iterrows():
        xmin, ymin, xmax, ymax = int(detection.xmin), int(detection.ymin), int(detection.xmax), int(detection.ymax)
        label = f"{detection.name} {detection.confidence:.2f}"
        color = (0, 255, 0)  # 绿色边界框
        img_array = cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), color, 2)
        img_array = cv2.putText(img_array, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 显示检测结果
    cv2.imshow("YOLOv5 Detection", img_array)
    cv2.waitKey(1)

# 附加传感器
def attach_sensors(world, vehicle):
    blueprint_library = world.get_blueprint_library()
    
    # RGB相机配置
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '110')
    
    # 生成RGB相机
    cam = world.spawn_actor(cam_bp, carla.Transform(), attach_to=vehicle)
    
    # 监听相机数据
    cam.listen(lambda data: process_image(data, model))
    
    return cam

# 数据记录器
class SensorDataRecorder:
    def __init__(self):
        self.image_queue = Queue(maxsize=100)
        self.control_queue = Queue(maxsize=100)
        self.sync_counter = 0
 
    def record_image(self, image):
        self.image_queue.put(image)
        self.sync_counter += 1
 
    def record_control(self, control):
        self.control_queue.put(control)
 
    def save_episode(self, episode_id):
        images = []
        controls = []
        while not self.image_queue.empty():
            images.append(self.image_queue.get())
        while not self.control_queue.empty():
            controls.append(self.control_queue.get())
        
        np.savez(f'expert_data/episode_{episode_id}.npz',
                 images=np.array(images),
                 controls=np.array(controls))

# 手动控制车辆
def manual_control(vehicle):
    keys = pygame.key.get_pressed()
    control = carla.VehicleControl()
    control.throttle = 0.5 * keys[pygame.K_UP]
    control.brake = 1.0 * keys[pygame.K_DOWN]
    control.steer = 2.0 * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
    vehicle.apply_control(control)
    return control

# 图像增强
def augment_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
    M = cv2.getRotationMatrix2D((400, 300), np.random.uniform(-5, 5), 1)
    augmented = cv2.warpAffine(hsv, M, (800, 600))
    return cv2.cvtColor(augmented, cv2.COLOR_HSV2BGR)

# 自动驾驶模型
class AutonomousDriver(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 94 * 70, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # throttle, brake, steer
        )
 
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# 训练模型
def train_model(model, dataloader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0
# 数据集类
class DrivingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = glob.glob(f'{data_dir}/*.npz')
        self.transform = transform

    def __len__(self):
        return len(self.files) * 100  # 假设每个episode有100帧

    def __getitem__(self, idx):
        file_idx = idx // 100
        frame_idx = idx % 100
        data = np.load(self.files[file_idx])
        image = data['images'][frame_idx].transpose(2, 0, 1).astype(np.float32) / 255.0  # 转换为CHW格式
        control = data['controls'][frame_idx].astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return torch.tensor(image), torch.tensor(control)

# 评估模型
def evaluate_model(model, world, episodes=10):
    metrics = {
        'collision_rate': 0,
        'route_completion': 0,
        'traffic_violations': 0,
        'control_smoothness': 0
    }
    
    control_filter = ControlFilter()
    
    for _ in range(episodes):
        vehicle = spawn_vehicle(world)
        recorder = SensorDataRecorder()
        
        # 附加传感器
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '110')
        cam = world.spawn_actor(cam_bp, carla.Transform(), attach_to=vehicle)
        cam.listen(lambda data: recorder.record_image(data))
        
        while True:
            # 获取控制输入（示例：手动控制）
            manual_control(vehicle)
            
            # 检查碰撞
            if vehicle.get_transform().location.distance(vehicle.get_world().get_map().get_spawn_points()[0].location) < 5.0:
                metrics['route_completion'] += 1
                break
                
            # 检查交通违规（示例）
            if random.random() < 0.01:
                metrics['traffic_violations'] += 1
            
            # 控制平滑度
            control = vehicle.get_control()
            smoothed_control = control_filter.smooth(control)
            vehicle.apply_control(smoothed_control)
            
            # 终止条件
            if metrics['collision_rate'] >= episodes:
                break
                
    return calculate_safety_scores(metrics)

# 模型量化
def quantize_model(model):
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    torch.ao.quantization.prepare(model, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)
    return model

# 控制信号平滑
class ControlFilter:
    def __init__(self, alpha=0.8):
        self.prev_control = None
        self.alpha = alpha
        
    def smooth(self, current_control):
        if self.prev_control is None:
            self.prev_control = current_control
            return current_control
        
        smoothed = self.alpha * self.prev_control + (1 - self.alpha) * current_control
        self.prev_control = smoothed
        return smoothed

# 导出模型
def export_model(model, output_path):
    model.eval()
    traced_model = torch.jit.trace(model, torch.randn(1, 3, 600, 800))
    traced_model.save(output_path)

# 加载模型
def load_deployed_model(model_path):
    model = AutonomousDriver()
    model.load_state_dict(torch.load(model_path))
    return model

# 自动驾驶主循环
def autonomous_driving_loop(world):
    model = load_deployed_model('deployed_model.pt')
    vehicle = spawn_vehicle(world)
    control_filter = ControlFilter()
    
    while True:
        # 获取相机图像
        image = get_camera_image(world, vehicle)
        preprocessed = preprocess_image(image)
        
        # 模型推理
        with torch.no_grad():
            control = model(preprocessed)
        
        # 控制信号平滑
        smoothed_control = control_filter.smooth(control)
        
        # 执行控制
        vehicle.apply_control(smoothed_control)
        
        # 安全监控
        if detect_critical_situation(vehicle):
            trigger_emergency_stop(vehicle)

# 辅助函数：获取相机图像
def get_camera_image(world, vehicle):
    # 这里需要根据你的传感器设置获取最新的相机图像
    # 示例代码：
    # for actor in world.get_actors():
    #     if actor.type_id == 'sensor.camera.rgb' and actor.parent.id == vehicle.id:
    #         return actor.queue.get()
    pass

# 辅助函数：预处理图像
def preprocess_image(image):
    # 图像预处理逻辑
    img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img_array = np.reshape(img_array, (image.height, image.width, 4))
    img_array = img_array[:, :, :3]  # 仅保留RGB通道
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = cv2.resize(img_array, (800, 600))
    return torch.from_numpy(img_array.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)

# 辅助函数：检测紧急情况
def detect_critical_situation(vehicle):
    # 紧急情况检测逻辑
    # 示例：检测车辆是否静止超过一定时间
    velocity = vehicle.get_velocity()
    if velocity.x == 0 and velocity.y == 0 and velocity.z == 0:
        return True
    return False

# 辅助函数：触发紧急停止
def trigger_emergency_stop(vehicle):
    # 紧急停止逻辑
    control = carla.VehicleControl()
    control.brake = 1.0
    control.hand_brake = True
    vehicle.apply_control(control)

# 辅助函数：计算安全分数
def calculate_safety_scores(metrics):
    # 安全分数计算逻辑
    total_score = 0
    total_score -= metrics['collision_rate'] * 10
    total_score += metrics['route_completion'] * 5
    total_score -= metrics['traffic_violations'] * 3
    total_score += metrics['control_smoothness'] * 2
    return total_score

if __name__ == '__main__':
    # 启动Carla服务器
    server_thread = Thread(target=launch_carla_server)
    server_thread.start()
    time.sleep(5)  # 等待服务器启动
    
    # 连接Carla客户端
    world = connect_carla()
    
    # 加载YOLOv5模型
    model = load_yolov5_model()
    
    # 训练模型
    dataset = DrivingDataset('expert_data')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    autodrive_model = AutonomousDriver()
    train_model(autodrive_model, dataloader, epochs=50)
    
    # 导出模型
    export_model(autodrive_model, 'deployed_model.pt')
    
    # 量化模型
    quantized_model = quantize_model(autodrive_model)
    export_model(quantized_model, 'quantized_model.pt')
    
    # 启动自动驾驶
    autonomous_driving_loop(world)
    
    # 评估模型
    metrics = evaluate_model(autodrive_model, world)
    print(f'Model Safety Score: {metrics}')