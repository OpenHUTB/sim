"""
perf_monitor.py

监控 AirSim 进程的 CPU 内核态/用户态占比

修复: Linux 下 cpu_times 没有 iowait 字段导致的 AttributeError
修复: 进程名匹配大小写问题

用法:
    python perf_monitor.py --pid 12345 --duration 60
    python perf_monitor.py --duration 60
"""

import argparse
import time
import os
import sys
import psutil
import pandas as pd


def find_airsim_pid() -> int:
    targets = ["airsim", "ue4editor", "unrealeditor"]
    for proc in psutil.process_iter(["pid", "name"]):
        name = proc.info["name"].lower()
        for t in targets:
            if t in name:
                print(f"[info] 找到: {proc.info['name']}  pid={proc.info['pid']}")
                return proc.info["pid"]
    return 0


def sample_process(pid: int):
    try:
        p = psutil.Process(pid)
        ct = p.cpu_times()
        # Linux 下 ct 有 iowait，Windows 下没有，统一只取 user + system
        total = ct.user + ct.system
        user_pct   = ct.user   / total * 100 if total > 0 else 0.0
        kernel_pct = ct.system / total * 100 if total > 0 else 0.0
        return {
            "total_cpu":   p.cpu_percent(interval=0.1),
            "user_pct":    round(user_pct, 1),
            "kernel_pct":  round(kernel_pct, 1),
            "threads":     p.num_threads(),
            "mem_mb":      round(p.memory_info().rss / 1024 / 1024, 1),
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
            print("[warn] 进程消失，停止")
            break
        d["time_sec"] = t
        records.append(d)
        print(f"  [{t:6.1f}s]  cpu={d['total_cpu']:5.1f}%  "
              f"user={d['user_pct']:5.1f}%  kernel={d['kernel_pct']:5.1f}%  "
              f"thr={d['threads']}")
        time.sleep(interval)

    if not records:
        print("[warn] 无数据")
        return

    df = pd.DataFrame(records)
    out_dir = os.path.dirname(os.path.abspath(out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[info] 已保存: {out}")
    print(f"平均 CPU:       {df.total_cpu.mean():.1f}%")
    print(f"平均内核态占比: {df.kernel_pct.mean():.1f}%")
    print(f"平均用户态占比: {df.user_pct.mean():.1f}%")
    print(f"最大线程数:     {df.threads.max()}")
    print(f"平均内存:       {df.mem_mb.mean():.0f} MB")


def main():
    ap = argparse.ArgumentParser(description="AirSim 进程 CPU 监控")
    ap.add_argument("--pid",      type=int,   default=0,
                    help="进程PID，0=自动查找")
    ap.add_argument("--duration", type=int,   default=60,
                    help="监控时长(秒)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="采样间隔(秒)")
    ap.add_argument("--output",   type=str,
                    default="results/data/perf_monitor.csv")
    args = ap.parse_args()

    pid = args.pid or find_airsim_pid()
    if pid == 0:
        print("[error] 找不到 AirSim 进程，请用 --pid 手动指定")
        sys.exit(1)

    monitor(pid, args.duration, args.interval, args.output)


if __name__ == "__main__":
    main()
