"""
batch_run.py

批量运行实验，自动生成 settings.json 并调用 swarm_test.py
2024/11

用法:
    python scripts/batch_run.py --max_drones 30 --output results/data
    python scripts/batch_run.py --write_settings   # 自动写 AirSim 配置
"""

import os
import json
import subprocess
import argparse
import time
import math


SETTINGS_PATH = os.path.expanduser("~/Documents/AirSim/settings.json")


def make_settings(n: int) -> dict:
    """生成 n 架无人机的 AirSim settings.json，圆形排布"""
    vehicles = {}
    for i in range(n):
        angle = 2 * math.pi * i / max(n, 1)
        r = 3.0 * math.ceil(n / 8)
        vehicles[f"Drone{i}"] = {
            "VehicleType": "SimpleFlight",
            "X": round(r * math.cos(angle), 2),
            "Y": round(r * math.sin(angle), 2),
            "Z": 0,
            "Yaw": 0,
        }
    return {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ClockType": "SteppableClock",
        "Vehicles": vehicles,
    }


def write_settings(n: int):
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(make_settings(n), f, indent=2)
    print(f"[info] settings.json 已写入 ({n} 架)")


def run_test(n: int, model: str, dur: int, out: str):
    cmd = [
        "python", "scripts/swarm_test.py",
        "--num_drones", str(n),
        "--model", model,
        "--duration", str(dur),
        "--output", out,
    ]
    print(f"\n[run] {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[warn] 退出码={ret.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_drones", type=int, default=30)
    ap.add_argument("--models", type=str, default="M1,M2,M4")
    ap.add_argument("--duration", type=int, default=20)
    ap.add_argument("--output", type=str, default="results/data")
    ap.add_argument("--write_settings", action="store_true",
                    help="自动写 settings.json（每次需重启 AirSim）")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    ns = sorted(set(n for n in [1, 5, 10, 15, 20, 25, args.max_drones]
                    if n <= args.max_drones))

    print(f"计划: models={models}  ns={ns}")
    print(f"共 {len(models) * len(ns)} 组实验")

    for n in ns:
        if args.write_settings:
            write_settings(n)
            print(f"[info] 请重启 AirSim 加载 {n} 架配置，按回车继续...")
            input()

        for m in models:
            run_test(n, m, args.duration, args.output)
            time.sleep(3)

    print("\n[done] 全部完成")


if __name__ == "__main__":
    main()
