# -*- coding: utf-8 -*-
import carla
import time

# ==========================
# 在这里直接编辑地图名字！！
TARGET_MAP = 'Town01'    # 改成 'Town10HD_Opt'和'Town01' 等其他地图名字即可
HOST = '127.0.0.1'       # Carla服务器IP
PORT = 2000              # Carla服务器端口
# ==========================

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    print(f"[INFO] Connecting to Carla server at {HOST}:{PORT}")
    print(f"[INFO] Loading map: {TARGET_MAP} ...")
    
    client.load_world(TARGET_MAP)

    time.sleep(2.0)  # 等待地图加载完成
    world = client.get_world()
    current_map = world.get_map().name

    print(f"[INFO] Successfully loaded map: {current_map}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted.")
    except Exception as e:
        print(f"[ERROR] {e}")
