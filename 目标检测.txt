git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# 创建虚拟环境
python -m venv carla_env
source carla_env/bin/activate
 
# 安装核心依赖
pip install carla pygame numpy matplotlib
pip install torch torchvision tensorboard
# server_launcher.py
import os
os.system('./CarlaUE4.sh Town01 -windowed -ResX=800 -ResY=600')
# client_connector.py
import carla
 
def connect_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return world
 
def spawn_vehicle(world):
    blueprint = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(blueprint, spawn_point)
    return vehicle
 
# 使用示例
world = connect_carla()
vehicle = spawn_vehicle(world)
# sensor_setup.py
# sensor_setup.py
from pathlib import Path
import torch

# 加载 YOLOv5 模型
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 根目录
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 加载预训练模型

def process_image(image):
    img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img_array = np.reshape(img_array, (image.height, image.width, 4))
    img_array = img_array[:,:,:3]  # 仅保留RGB通道
    img = Image.fromarray(img_array)
    
    # 使用 YOLOv5 进行目标检测
    results = model(img)
    detections = results.pandas().xyxy[0]  # 转换为 pandas DataFrame
    
    # 在 Carla 中绘制检测结果（示例：绘制边界框）
    for _, detection in detections.iterrows():
        xmin, ymin, xmax, ymax = int(detection.xmin), int(detection.ymin), int(detection.xmax), int(detection.ymax)
        label = f"{detection.name} {detection.confidence:.2f}"
        color = (0, 255, 0)  # 绿色边界框
        img_array = cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), color, 2)
        img_array = cv2.putText(img_array, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 可选：显示检测结果
    cv2.imshow("YOLOv5 Detection", img_array)
    cv2.waitKey(1)
def attach_sensors(vehicle):
    # RGB相机配置
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '110')
    
    # IMU配置
    imu_bp = world.get_blueprint_library().find('sensor.other.imu')
    
    # 生成传感器
    cam = world.spawn_actor(cam_bp, carla.Transform(), attach_to=vehicle)
    imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    
    # 监听传感器数据
    cam.listen(lambda data: process_image(data))
    imu.listen(lambda data: process_imu(data))
    return cam, imu
# data_recorder.py
import numpy as np
from queue import Queue
 
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
# expert_controller.py
def manual_control(vehicle):
    while True:
        control = vehicle.get_control()
        # 添加专家控制逻辑（示例：键盘控制）
        keys = pygame.key.get_pressed()
        control.throttle = 0.5 * keys[K_UP]
        control.brake = 1.0 * keys[K_DOWN]
        control.steer = 2.0 * (keys[K_RIGHT] - keys[K_LEFT])
        return control
# data_augmentation.py
def augment_image(image):
    # 随机亮度调整
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2]*np.random.uniform(0.8,1.2),0,255)
    
    # 随机旋转（±5度）
    M = cv2.getRotationMatrix2D((400,300), np.random.uniform(-5,5), 1)
    augmented = cv2.warpAffine(hsv, M, (800,600))
    
    return cv2.cvtColor(augmented, cv2.COLOR_HSV2BGR)
# model.py
import torch
import torch.nn as nn
 
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
            nn.Linear(64*94*70, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # throttle, brake, steer
        )
 
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)
# train.py
def train_model(model, dataloader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch['images'].to(device)
            targets = batch['controls'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
        torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pth')
# dataset.py
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
        image = data['images'][frame_idx].transpose(1,2,0)  # HWC to CHW
        control = data['controls'][frame_idx]
        
        if self.transform:
            image = self.transform(image)
            
        return torch.tensor(image, dtype=torch.float32)/255.0, \
               torch.tensor(control, dtype=torch.float32)
# evaluator.py
def evaluate_model(model, episodes=10):
    metrics = {
        'collision_rate': 0,
        'route_completion': 0,
        'traffic_violations': 0,
        'control_smoothness': 0
    }
    
    for _ in range(episodes):
        vehicle = spawn_vehicle(world)
        while True:
            # 获取传感器数据
            image = get_camera_image()
            control = model.predict(image)
            
            # 执行控制
            vehicle.apply_control(control)
            
            # 安全检测
            check_collisions(vehicle, metrics)
            check_traffic_lights(vehicle, metrics)
            
            # 终止条件
            if has_reached_destination(vehicle):
                metrics['route_completion'] += 1
                break
                
    return calculate_safety_scores(metrics)
# quantization.py
model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
torch.ao.quantization.prepare(model, inplace=True)
torch.ao.quantization.convert(model, inplace=True)
# control_smoothing.py
class ControlFilter:
    def __init__(self, alpha=0.8):
        self.prev_control = None
        self.alpha = alpha
        
    def smooth(self, current_control):
        if self.prev_control is None:
            self.prev_control = current_control
            return current_control
        
        smoothed = self.alpha * self.prev_control + (1-self.alpha) * current_control
        self.prev_control = smoothed
        return smoothed
# model_export.py
def export_model(model, output_path):
    traced_model = torch.jit.trace(model, torch.randn(1,3,600,800))
    traced_model.save(output_path)
 
# 加载示例
loaded_model = torch.jit.load('deployed_model.pt')
# deploy.py
def autonomous_driving_loop():
    model = load_deployed_model()
    vehicle = spawn_vehicle(world)
    
    while True:
        # 传感器数据获取
        image_data = get_camera_image()
        preprocessed = preprocess_image(image_data)
        
        # 模型推理
        with torch.no_grad():
            control = model(preprocessed).numpy()
        
        # 控制信号后处理
        smoothed_control = control_filter.smooth(control)
        
        # 执行控制
        vehicle.apply_control(smoothed_control)
        
        # 安全监控
        if detect_critical_situation():
            trigger_emergency_stop()

