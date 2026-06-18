# 方案设计记录

2024/10

---

## 1. 问题在哪

### 现象

在 AirSim 里跑多机实验的时候，无人机数量到20架以上会出现明显卡顿：

| 无人机数量 | 实测 FPS | 指令延迟 | 内核态 CPU |
|:---:|:---:|:---:|:---:|
| 1   | ~60  | ~3ms   | ~8%  |
| 10  | ~45  | ~8ms   | ~20% |
| 20  | ~18  | ~22ms  | ~48% |
| 30  | ~8   | ~45ms  | ~62% |

数据在测试机（16线程）上采集，仅供参考，不同硬件环境结果有差异。

### 根因

问题在 `AirLib/include/common/ClockBase.hpp`，`SteppableClock` 的 `sleep_for` 实现：

```cpp
virtual void sleep_for(TTimeDelta dt) {
    TTimePoint start = now();
    TTimePoint end   = start + dt;
    while (now() < end) {
        // 每次循环都调用 sleep_for(0)，触发一次内核调用
        std::this_thread::sleep_for(
            std::chrono::duration<int64_t, std::nano>(0));
    }
}
```

每架无人机对应一个仿真线程，物理步长是 10ms（100Hz）。每个线程在等下一个时间步的时候，会不停地调 `sleep_for(0)`。

`sleep_for(0)` 会触发 `NtDelayExecution`（Windows）或 `nanosleep`（Linux）系统调用，每次调用都有内核态/用户态切换的开销。

30架无人机：30个线程 × ~100次/秒 × 内核切换 = 调度压力很大。超过物理核心数之后操作系统调度器开始频繁切换线程上下文，内核态时间占比飙升。

用 Windows Performance Analyzer 抓到的火焰图能很清楚看到 `NtDelayExecution` 占了大量 CPU 时间。

---

## 2. 五种替代方案

### M1：单线程分发器（主要优化方案）

思路：把所有的"等待"集中到一个 dispatcher 线程。dispatcher 用 `sleep_until` 等到下一个 tick，然后 `notify_all` 唤醒所有 worker 线程。worker 线程通过 `condition_variable::wait` 阻塞，不做任何等待操作。

```
dispatcher (1个线程)
    sleep_until(next_tick)  ← 唯一发出系统调用的地方
    notify_all()

worker_0, worker_1, ..., worker_N
    cv.wait(...)            ← 纯阻塞，不占 CPU
```

优势：N个无人机，系统调用从 O(N) 降到 O(1)  
缺点：所有 worker 同时被唤醒，短时间内有 N 个线程竞争 CPU，可能出现短暂的调度峰值

### M2：标准 sleep_until

每个线程各自算好下一个 tick 时刻，直接 `sleep_until`，不再自旋。

比 M4 原始方案好，因为 `sleep_until` 只调用一次系统调用。  
问题：Windows 默认计时器分辨率是 15.6ms，`sleep_until` 精度不够，可能会导致仿真 tick 不均匀。可以用 `timeBeginPeriod(1)` 提高到 1ms，但这是全局设置，影响系统其他程序。

### M3：yield 自旋

```cpp
while (Clock::now() < nxt)
    std::this_thread::yield();
```

`yield` 不进内核，只是提示调度器"可以切换了"，比 `sleep(0)` 开销小一点。但线程多的时候仍然会占满 CPU，实测线程数超 10 之后系统已经很卡了，不适合实际使用。

### M4：sleep(0) 自旋（AirSim 原始方式）

就是现在的实现，留在这里作为对照组。

### M5：纯自旋

```cpp
while (Clock::now() < nxt) {}
```

精度最高，但每个线程都会把一个 CPU 核心跑满。实验中只在线程数 ≤ 5 的时候测，避免死机。

---

## 3. 实验方案

### 3.1 微观基准测试（`src/thread_benchmark/benchmark.cpp`）

不依赖 AirSim，单独测各模型的定时精度。

- 目标间隔：10ms（100Hz）
- 并发线程数：1, 5, 10, 20, 30
- 每组跑 5 秒
- 指标：平均误差、最大误差、标准差

### 3.2 AirSim 集成测试（`scripts/swarm_test.py`）

在真实 AirSim 环境里跑，对每种模型 × 每个无人机数量组合做测试。

- 无人机数量：1, 5, 10, 15, 20, 25, 30
- 飞行场景：圆形编队
- 每组 30 秒
- 指标：FPS、指令延迟、CPU占用

### 3.3 CPU 内核态分析（`scripts/perf_monitor.py`）

专门监控 AirSim 进程的内核态/用户态时间占比，这是判断系统调用压力的关键指标。

---

## 4. 预期结论

理论上 M1 应该是综合表现最好的：系统调用只有 dispatcher 一个线程产生，worker 纯阻塞，内核态 CPU 占比应该最低，FPS 最高。

但实际结果还需要跑完实验才知道，也有可能 M2 在某些情况下更好（实现更简单，且 Windows 10/11 上计时器分辨率改进了很多）。

---

## 5. 已知问题/TODO

- [ ] AirSim settings.json 里多机配置还没搞定，先用模拟模式跑通了流程
- [ ] Windows 上编译 C++ 部分需要 MSVC 或者 MinGW，CMake 配置还没测
- [ ] M1 方案里 dispatcher notify_all 之后所有 worker 同时醒来，可能有"惊群"问题，需要验证
- [ ] perf_monitor 在 Linux 上没测过

---

## 6. 开发环境

- AirSim：基于 [Midas75/AirSim](https://github.com/Midas75/AirSim)
- Unreal Engine：4.27
- 编译：MSVC 2022 / CMake 3.20+
- Python：3.9+
- 操作系统：Windows 11
