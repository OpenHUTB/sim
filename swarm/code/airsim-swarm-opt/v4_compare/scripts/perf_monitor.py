"""
perf_monitor.py

监控进程的 CPU 内核态/用户态占比

用法:
    python perf_monitor.py --pid 12345 --duration 30
    python perf_monitor.py --duration 30   # 自动找 AirSim 进程

说明:
    第一版，暂时只支持 Windows，Linux 版本后面补
"""

import argparse
import time
import os
import psutil
import pandas as pd


def find_airsim_pid() -> int:
    targets = ["AirSim", "UE4Editor", "UnrealEditor"]
    for proc in psutil.process_iter(["pid", "name"]):
        for t in targets:
            if t.lower() in proc.info["name"].lower():
                print(f"[info] 找到: {proc.info['name']}  pid={proc.info['pid']}")
                return proc.info["pid"]
    return 0


def sample_process(pid: int):
    try:
        p = psutil.Process(pid)
        ct = p.cpu_times()
        total = ct.user + ct.system
        user_pct = ct.user / total * 100 if total > 0 else 0.0
        kernel_pct = ct.system / total * 100 if total > 0 else 0.0
        return {
            "total_cpu": p.cpu_percent(interval=0.1),
            "user_pct": round(user_pct, 1),
            "kernel_pct": round(kernel_pct, 1),
            "threads": p.num_threads(),
            "mem_mb": round(p.memory_info().rss / 1024 / 1024, 1),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def monitor(pid: int, dur: int, interval: float, out: str):
    print(f"[info] 监控 pid={pid}，时长={dur}s")
    records = []
    t0 = time.time()

    while time.time() - t0 < dur:
        t = round(time.time() - t0, 2)
        d = sample_process(pid)
        if d is None:
            print("[warn] 进程消失，停止")
            break
        d["time_sec"] = t
        records.append(d)
        print(f"  [{t:6.1f}s]  cpu={d['total_cpu']:5.1f}%  "
              f"kernel={d['kernel_pct']:5.1f}%  thr={d['threads']}")
        time.sleep(interval)

    if not records:
        return

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[info] 保存: {out}")
    print(f"平均内核态: {df.kernel_pct.mean():.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--output", type=str, default="results/data/perf_monitor.csv")
    args = ap.parse_args()

    pid = args.pid or find_airsim_pid()
    if pid == 0:
        print("[error] 找不到进程，用 --pid 手动指定")
        return

    monitor(pid, args.duration, args.interval, args.output)


if __name__ == "__main__":
    main()
