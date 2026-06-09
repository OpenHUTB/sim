"""
gen_settings.py

根据无人机数量自动生成 AirSim settings.json
主要是坐标计算麻烦，手写容易出错，所以写了这个脚本

用法:
    python scripts/gen_settings.py --num 10 --output settings.json
    python scripts/gen_settings.py --num 30 --layout circle --radius 15

2024/10
"""

import json
import math
import argparse
import os


def gen_circle_positions(n, radius=10.0, height=0.5):
    """
    把 n 架无人机摆成圆形
    radius: 圆半径，单位米
    height: 离地高度，别太高不然掉下来那个距离不好看
    """
    positions = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions.append((x, y, height))
    return positions


def gen_grid_positions(n, spacing=5.0, height=0.5):
    """
    网格排列，适合数量多的时候
    spacing: 间距，单位米
    """
    cols = math.ceil(math.sqrt(n))
    positions = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = col * spacing - (cols * spacing / 2)
        y = row * spacing - (cols * spacing / 2)
        positions.append((x, y, height))
    return positions


def build_settings(n, layout='circle', radius=10.0, spacing=5.0):
    if layout == 'circle':
        positions = gen_circle_positions(n, radius)
    else:
        positions = gen_grid_positions(n, spacing)

    vehicles = {}
    for i in range(n):
        name = f"Drone{i+1}"
        x, y, z = positions[i]
        vehicles[name] = {
            "VehicleType": "SimpleFlight",
            "X": round(x, 2),
            "Y": round(y, 2),
            "Z": round(-z, 2),   # AirSim Z轴朝下
            "Yaw": 0,
            "EnableCollisions": True,
            "EnableTrace": False,  # 多机时开 trace 会很卡
            "Sensors": {
                "Barometer": {
                    "SensorType": 1,
                    "Enabled": True
                }
            }
        }

    settings = {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ClockSpeed": 1.0,
        "Vehicles": vehicles,
        "CameraDefaults": {
            "CaptureSettings": [
                {
                    "ImageType": 0,
                    "Width": 256,
                    "Height": 144,
                    "FOV_Degrees": 90
                }
            ]
        }
    }
    return settings


def main():
    parser = argparse.ArgumentParser(description='生成 AirSim settings.json')
    parser.add_argument('--num', type=int, default=10, help='无人机数量')
    parser.add_argument('--layout', choices=['circle', 'grid'], default='circle')
    parser.add_argument('--radius', type=float, default=10.0, help='圆形排列半径(m)')
    parser.add_argument('--spacing', type=float, default=5.0, help='网格间距(m)')
    parser.add_argument('--output', type=str, default='settings.json')
    args = parser.parse_args()

    settings = build_settings(args.num, args.layout, args.radius, args.spacing)

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True) if os.path.dirname(out_path) else None

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

    print(f"生成完成: {out_path}")
    print(f"  无人机数量: {args.num}")
    print(f"  排列方式: {args.layout}")
    print(f"  输出路径: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
