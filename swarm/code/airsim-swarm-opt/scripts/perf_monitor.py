"""
perf_monitor.py

监控 AirSim 进程的 CPU 内核态/用户态占比

用法:
    python perf_monitor.py --pid 12345 --duration 60
    python perf_monitor.py --duration 60   # 自动找 AirSim 进程

输出 CSV 包含: time_sec, total_cpu, user_pct, kernel_pct, threads, mem_mb
"""

import argparse
import time
import os
import psutil
import pandas as pd


def find_airsim_pid() -> int:
    """找 AirSim/UE4 进程，找不到返回0"""
    targets = ["AirSim", "UE4Editor", "UnrealEditor"]
    for proc in psutil.process_iter(["pid", "name"]):
        for t in targets:
            if t.lower() in proc.info["name"].lower():
                print(f"[info] 找到: {proc.info['name']}  pid={proc.info['pid']}")
                return proc.info["pid"]
    return 0


def sample_process(pid: int):
    """采一次，返回 dict，进程不存在返回 None"""
    try:
        p = psutil.Process(pid)
        ct = p.cpu_times()
        total_cpu_time = ct.user + ct.system
        if total_cpu_time == 0:
            user_pct = kernel_pct = 0.0
        else:
            user_pct = ct.user / total_cpu_time * 100
            kernel_pct = ct.system / total_cpu_time * 100
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
    print(f"[info] 开始监控 pid={pid}，时长={dur}s，间隔={interval}s")
    records = []
    t0 = time.time()

    while time.time() - t0 < dur:
        t = round(time.time() - t0, 2)
        d = sample_process(pid)
        if d is None:
            print("[warn] 进程消失了，停止")
            break
        d["time_sec"] = t
        records.append(d)
        print(f"  [{t:6.1f}s]  cpu={d['total_cpu']:5.1f}%  "
              f"user={d['user_pct']:5.1f}%  kernel={d['kernel_pct']:5.1f}%  "
              f"thr={d['threads']}")
        time.sleep(interval)

    if not records:
        print("[warn] 没有数据")
        return

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[info] 已保存: {out}")

    print("\n--- 统计 ---")
    print(f"平均 CPU:        {df.total_cpu.mean():.1f}%")
    print(f"平均内核态占比:  {df.kernel_pct.mean():.1f}%")
    print(f"平均用户态占比:  {df.user_pct.mean():.1f}%")
    print(f"最大线程数:      {df.threads.max()}")
    print(f"平均内存:        {df.mem_mb.mean():.0f} MB")


def main():
    ap = argparse.ArgumentParser(description="AirSim 进程 CPU 监控")
    ap.add_argument("--pid", type=int, default=0, help="进程PID，0=自动查找")
    ap.add_argument("--duration", type=int, default=60, help="监控时长(秒)")
    ap.add_argument("--interval", type=float, default=1.0, help="采样间隔(秒)")
    ap.add_argument("--output", type=str, default="results/data/perf_monitor.csv")
    args = ap.parse_args()

    pid = args.pid
    if pid == 0:
        pid = find_airsim_pid()
    if pid == 0:
        print("[error] 找不到 AirSim 进程，用 --pid 手动指定")
        return

    monitor(pid, args.duration, args.interval, args.output)


if __name__ == "__main__":
    main()
