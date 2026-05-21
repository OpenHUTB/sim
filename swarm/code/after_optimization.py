# ============================================================
# 优化后：并发控制无人机集群
# 优化点：
#   1. 多线程并发起飞/飞行（时间缩短）
#   2. 速度自适应调整（根据距离动态设速）
#   3. 平滑路径插值（减少位置误差）
#   4. 异步指令批量下发（降低响应延迟）
# ============================================================

import airsim
import time
import json
import math
import os
import threading

# -------- 性能数据收集 --------
metrics = {
    "mode": "优化后（并发+路径优化）",
    "takeoff_times": [],
    "task_times": [],
    "response_latencies": [],
    "position_errors": [],
    "total_time": 0
}
metrics_lock = threading.Lock()

DRONES = ["Drone1", "Drone2", "Drone3"]

WAYPOINTS = {
    "Drone1": [(0, 0, -10), (10, 0, -10), (10, 10, -10), (0, 10, -10)],
    "Drone2": [(4, 0, -12), (14, 0, -12), (14, 10, -12), (4, 10, -12)],
    "Drone3": [(8, 0, -8),  (18, 0, -8),  (18, 10, -8),  (8, 10, -8)],
}


def connect():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("已连接到 AirSim")
    return client


def setup_drone(client, name):
    client.enableApiControl(True, name)
    client.armDisarm(True, name)


# -------- 优化1：并发起飞 --------
def takeoff_all_concurrent(client):
    """所有无人机同时起飞，不互相等待"""
    print("\n[阶段1] 并发起飞（所有无人机同时）...")
    futures = []
    t0 = time.time()

    for name in DRONES:
        f = client.takeoffAsync(vehicle_name=name)
        futures.append((name, f, time.time()))

    for name, f, t_start in futures:
        f.join()
        elapsed = time.time() - t_start
        with metrics_lock:
            metrics["takeoff_times"].append(elapsed)
        print(f"  {name} 起飞完成，耗时 {elapsed:.2f}s")

    print(f"  全部起飞总耗时: {time.time()-t0:.2f}s（串行需逐架等待）")


# -------- 优化2：自适应速度 --------
def calc_speed(p1, p2):
    """根据距离自适应调整飞行速度"""
    dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
    if dist > 15:
        return 8   # 远距离：快速
    elif dist > 8:
        return 6   # 中距离：正常
    else:
        return 4   # 近距离：精准


# -------- 优化3：平滑路径插值 --------
def interpolate_path(waypoints, steps=2):
    """在航点之间插入中间点，使路径更平滑"""
    smooth = []
    for i in range(len(waypoints) - 1):
        p1, p2 = waypoints[i], waypoints[i+1]
        smooth.append(p1)
        for s in range(1, steps):
            t = s / steps
            mid = (
                p1[0] + (p2[0]-p1[0]) * t,
                p1[1] + (p2[1]-p1[1]) * t,
                p1[2] + (p2[2]-p1[2]) * t
            )
            smooth.append(mid)
    smooth.append(waypoints[-1])
    return smooth


# -------- 优化4：单架无人机并发飞行线程 --------
def fly_drone_thread(client, name):
    """每架无人机在独立线程中飞行"""
    t0 = time.time()
    raw_waypoints = WAYPOINTS[name]
    waypoints = interpolate_path(raw_waypoints)  # 平滑路径
    errors = []
    prev_wp = waypoints[0]

    for wp in waypoints:
        speed = calc_speed(prev_wp, wp)   # 自适应速度
        t_cmd = time.time()
        client.moveToPositionAsync(wp[0], wp[1], wp[2], speed,
                                   vehicle_name=name).join()
        latency = time.time() - t_cmd

        with metrics_lock:
            metrics["response_latencies"].append(latency)

        pos = client.getMultirotorState(vehicle_name=name).kinematics_estimated.position
        error = math.sqrt((pos.x_val - wp[0])**2 +
                          (pos.y_val - wp[1])**2 +
                          (pos.z_val - wp[2])**2)
        errors.append(error)
        prev_wp = wp

    task_time = time.time() - t0
    with metrics_lock:
        metrics["task_times"].append(task_time)
        metrics["position_errors"].extend(errors)

    print(f"  {name} 任务完成，耗时 {task_time:.2f}s，平均误差 {sum(errors)/len(errors):.2f}m")


def fly_all_concurrent(client):
    """所有无人机并发执行飞行任务"""
    print("\n[阶段2] 并发执行飞行任务（多线程）...")
    threads = []
    for name in DRONES:
        t = threading.Thread(target=fly_drone_thread, args=(client, name))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()


def land_all_concurrent(client):
    """并发降落"""
    print("\n[阶段3] 并发降落...")
    futures = [(name, client.landAsync(vehicle_name=name)) for name in DRONES]
    for name, f in futures:
        f.join()
        client.armDisarm(False, name)
        client.enableApiControl(False, name)
        print(f"  {name} 已降落")


def main():
    print("=" * 50)
    print("  优化后：并发控制无人机集群")
    print("=" * 50)

    client = connect()

    for name in DRONES:
        setup_drone(client, name)

    total_start = time.time()

    takeoff_all_concurrent(client)
    fly_all_concurrent(client)
    land_all_concurrent(client)

    metrics["total_time"] = time.time() - total_start

    save_metrics()
    print_summary()


def save_metrics():
    os.makedirs("results", exist_ok=True)
    with open("results/after_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n数据已保存到 results/after_metrics.json")


def print_summary():
    print("\n" + "=" * 50)
    print("  性能统计（优化后）")
    print("=" * 50)
    print(f"  总耗时:        {metrics['total_time']:.2f} 秒")
    print(f"  平均起飞时间:  {sum(metrics['takeoff_times'])/len(metrics['takeoff_times']):.2f} 秒")
    print(f"  平均任务时间:  {sum(metrics['task_times'])/len(metrics['task_times']):.2f} 秒")
    print(f"  平均响应延迟:  {sum(metrics['response_latencies'])/len(metrics['response_latencies'])*1000:.0f} ms")
    print(f"  平均位置误差:  {sum(metrics['position_errors'])/len(metrics['position_errors']):.2f} 米")
    print("=" * 50)


if __name__ == "__main__":
    main()
