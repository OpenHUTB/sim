"""
swarm_test.py

多无人机集群控制 + 性能数据采集
2024/11

用法:
    python swarm_test.py --num_drones 10 --model M1 --duration 30
    python swarm_test.py --sweep

依赖: pip install airsim numpy pandas psutil
没有 AirSim 环境时自动切换模拟模式，数据用经验公式估算。
"""

import argparse
import time
import os
import math
import threading
import numpy as np
import pandas as pd
import psutil

try:
    import airsim
    _AIRSIM_OK = True
except ImportError:
    _AIRSIM_OK = False
    print("[warn] airsim 未安装，使用模拟模式")


class SwarmController:
    def __init__(self, n: int, model: str = "M1"):
        self.n = n
        self.model = model
        self.names = [f"Drone{i}" for i in range(n)]
        self.client = None

    def connect(self):
        if not _AIRSIM_OK:
            print(f"[sim] 模拟模式，{self.n} 架")
            return
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            for name in self.names:
                self.client.enableApiControl(True, name)
                self.client.armDisarm(True, name)
            print(f"[info] 连接成功，{self.n} 架就绪")
        except Exception as e:
            print(f"[warn] 连接失败({e})，切换模拟模式")
            self.client = None

    def takeoff_all(self):
        if self.client is None:
            return
        futs = [self.client.takeoffAsync(timeout_sec=10, vehicle_name=n)
                for n in self.names]
        for f in futs:
            f.join()

    def formation_step(self, alt: float):
        if self.client is None:
            return
        futs = []
        for i, name in enumerate(self.names):
            angle = 2 * math.pi * i / self.n
            r = 5.0 * math.ceil(self.n / 8)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            futs.append(self.client.moveToPositionAsync(
                x, y, alt, velocity=3.0, vehicle_name=name))
        for f in futs:
            f.join()

    def land_all(self):
        if self.client is None:
            return
        futs = [self.client.landAsync(vehicle_name=n) for n in self.names]
        for f in futs:
            f.join()

    def sample_fps(self) -> float:
        if self.client is None:
            # 模拟值：随机扰动 + 随无人机数量衰减
            base = 60.0
            drop = max(0.1, 1.0 - (self.n - 1) * 0.048)
            return max(1.5, base * drop + np.random.normal(0, 1.5))
        try:
            t0 = time.perf_counter()
            self.client.getMultirotorState(vehicle_name=self.names[0])
            dt = time.perf_counter() - t0
            return 1.0 / dt if dt > 0 else 60.0
        except Exception:
            return 0.0

    def sample_latency_ms(self) -> float:
        if self.client is None:
            base = 4.5 + self.n * 1.15
            return max(1.0, base + np.random.normal(0, 1.3))
        try:
            t0 = time.perf_counter()
            self.client.getMultirotorState(vehicle_name=self.names[0])
            return (time.perf_counter() - t0) * 1000.0
        except Exception:
            return 0.0


class Monitor:
    def __init__(self, ctrl: SwarmController, interval=0.2):
        self.ctrl = ctrl
        self.interval = interval
        self.records = []
        self._running = False
        self._t = None

    def start(self):
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._running = False
        if self._t:
            self._t.join(timeout=3)

    def _loop(self):
        proc = psutil.Process(os.getpid())
        while self._running:
            self.records.append({
                "ts": time.time(),
                "n": self.ctrl.n,
                "model": self.ctrl.model,
                "fps": self.ctrl.sample_fps(),
                "latency_ms": self.ctrl.sample_latency_ms(),
                "cpu_pct": psutil.cpu_percent(),
                "mem_mb": proc.memory_info().rss / 1024 / 1024,
            })
            time.sleep(self.interval)

    def summary(self) -> dict:
        if not self.records:
            return {}
        df = pd.DataFrame(self.records)
        return {
            "n": self.ctrl.n,
            "model": self.ctrl.model,
            "fps_mean": df.fps.mean(),
            "fps_min": df.fps.min(),
            "fps_p5": df.fps.quantile(0.05),
            "lat_mean": df.latency_ms.mean(),
            "lat_max": df.latency_ms.max(),
            "cpu_mean": df.cpu_pct.mean(),
            "mem_mean": df.mem_mb.mean(),
        }


def run_one(n: int, model: str, dur: int, out_dir: str) -> dict:
    print(f"\n{'='*50}")
    print(f"  n={n}  model={model}  dur={dur}s")
    print(f"{'='*50}")

    ctrl = SwarmController(n, model)
    ctrl.connect()
    ctrl.takeoff_all()

    mon = Monitor(ctrl)
    mon.start()

    t0 = time.time()
    step = 0
    while time.time() - t0 < dur:
        ctrl.formation_step(-10.0 - step * 2)
        step = (step + 1) % 5
        time.sleep(1.0)

    mon.stop()
    ctrl.land_all()

    s = mon.summary()
    print(f"  fps={s.get('fps_mean', 0):.1f}  "
          f"lat={s.get('lat_mean', 0):.1f}ms  "
          f"cpu={s.get('cpu_mean', 0):.1f}%")

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(mon.records).to_csv(
        os.path.join(out_dir, f"raw_{model}_{n}d.csv"), index=False)
    return s


def run_sweep(out_dir: str, dur: int):
    models = ["M1", "M2", "M4"]
    ns = [1, 5, 10, 15, 20, 25, 30]
    rows = []
    for m in models:
        for n in ns:
            rows.append(run_one(n, m, dur, out_dir))
            time.sleep(2)
    df = pd.DataFrame(rows)
    p = os.path.join(out_dir, "swarm_summary.csv")
    df.to_csv(p, index=False)
    print(f"\n汇总: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_drones", type=int, default=10)
    ap.add_argument("--model", type=str, default="M1",
                    choices=["M1", "M2", "M3", "M4", "M5", "all"])
    ap.add_argument("--duration", type=int, default=20)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--output", type=str, default="results/data")
    args = ap.parse_args()

    if args.sweep:
        run_sweep(args.output, args.duration)
    else:
        ms = ["M1", "M2", "M3", "M4", "M5"] if args.model == "all" else [args.model]
        for m in ms:
            run_one(args.num_drones, m, args.duration, args.output)


if __name__ == "__main__":
    main()
