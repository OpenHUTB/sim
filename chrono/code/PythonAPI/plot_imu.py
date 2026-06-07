import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 自动找最新录制的 IMU 文件
csv_files = glob.glob("imu_records/imu_recording_*.csv")
latest_csv = max(csv_files, key=os.path.getctime)
print("正在绘图：" + latest_csv)

# 读取数据
df = pd.read_csv(latest_csv)

# 时间归一化（从0开始）
t = df['timestamp'] - df['timestamp'].iloc[0]

# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(14, 10))

# 1. 三轴加速度
plt.subplot(3,1,1)
plt.plot(t, df['accel_x'], label='X 加速度', linewidth=1.2)
plt.plot(t, df['accel_y'], label='Y 加速度', linewidth=1.2)
plt.plot(t, df['accel_z'], label='Z 加速度', linewidth=1.2)
plt.title('IMU 三轴加速度', fontsize=14)
plt.ylabel('m/s²')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 三轴角速度
plt.subplot(3,1,2)
plt.plot(t, df['gyro_x'], label='X 角速度', linewidth=1.2)
plt.plot(t, df['gyro_y'], label='Y 角速度', linewidth=1.2)
plt.plot(t, df['gyro_z'], label='Z 角速度', linewidth=1.2)
plt.title('IMU 三轴角速度', fontsize=14)
plt.ylabel('rad/s')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. 航向角
plt.subplot(3,1,3)
plt.plot(t, df['compass'], label='航向角 Compass', linewidth=1.2, color='orange')
plt.title('IMU 航向角', fontsize=14)
plt.xlabel('时间 (s)')
plt.ylabel('度 (°)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("imu_result.png", dpi=300)
plt.show()
print("✅ 折线图已保存：imu_result.png")