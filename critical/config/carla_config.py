# config/carla_config.py
# 全局固定参数：CARLA 连接、地图、车型、传感器通用配置、全局路径
# 按 config/CLAUDE.md 规范 —— 不写入任何场景差异化内容

# ============================================================
# 1. CARLA 连接公共参数
# ============================================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 30.0

# 同步模式（强化学习训练必须）
SYNC_MODE = True
SIMULATION_FPS = 20
FIXED_DELTA_SECONDS = 1.0 / SIMULATION_FPS

# ============================================================
# 2. 全局车辆公共参数
# ============================================================
EGO_VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
ADV_VEHICLE_BLUEPRINT = "vehicle.audi.a2"

# 全局默认生成点索引（场景可覆盖）
DEFAULT_SPAWN_INDEX = 0

# 碰撞检测阈值 & 控制频率（所有场景共用）
COLLISION_THRESHOLD = 1.5            # 碰撞判定距离 (m)
MIN_SAFE_DISTANCE = 3.0              # 最小安全距离 (m)
VEHICLE_MAX_SPEED = 80.0             # 最高速度 (km/h)

# ============================================================
# 3. 全局传感器公共参数
# ============================================================
RGB_CAMERA_CONFIG = {
    "image_size_x": "640",
    "image_size_y": "480",
    "fov": "90.0",
}

LIDAR_CONFIG = {
    "points_per_second": "100000",
    "range": "50.0",
}

SENSOR_UPDATE_FPS = 20               # 传感器采集频率（全局统一）

# ============================================================
# 4. 训练全局参数（所有算法共用）
# ============================================================
MAX_EPISODE_STEPS = 500
EPISODE_TIMEOUT = 60.0               # 单集超时 (s)
COLLISION_PENALTY = -10.0            # 碰撞惩罚
SUCCESS_REWARD = 10.0                # 安全完成奖励

# ============================================================
# 5. 全局路径
# ============================================================
RESULTS_DIR = "results"
MODELS_DIR = "models"
LOGS_DIR = "logs"
SCENARIOS_OUTPUT_DIR = "results/scenarios"
EVALUATION_OUTPUT_DIR = "results/evaluation"
COMPARISON_OUTPUT_DIR = "results/comparison"
PLOTS_OUTPUT_DIR = "results/plots"
