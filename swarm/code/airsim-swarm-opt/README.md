# 基于 AirSim 的无人机集群仿真性能优化

---

## 这个项目是干什么的

AirSim 是微软开源的无人机/无人车仿真平台，基于 Unreal Engine，可以比较真实地模拟物理环境。

但我在跑多机集群实验的时候发现，无人机一旦超过大概 20 架，帧率会掉得很严重，指令延迟也会变大，导致集群算法根本没法正常验证。

翻了一下源码，发现问题出在 `AirLib/include/common/ClockBase.hpp` 里的线程等待逻辑：

```cpp
// AirSim 原始实现
virtual void sleep_for(TTimeDelta dt) {
    TTimePoint end = now() + dt;
    while (now() < end) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(0));  // 问题在这
    }
}
```

每架无人机是一个线程，每个线程以 100Hz 的频率调用 `sleep(0)`。`sleep(0)` 这个调用虽然语义上是"立刻返回"，但每次都会触发内核态/用户态切换。30 架无人机就是 30 × 100 = 3000 次/秒的系统调用，超过物理核心数之后调度器开始出问题，内核态 CPU 占比会飙到 60% 以上。

本项目的目标：设计并对比几种替代方案，找到一个既能保证仿真精度又不会拖垮 CPU 的线程等待模型。

## 参考

- AirSim 仿真平台（Midas75 fork）：https://github.com/Midas75/AirSim
- 环境搭建说明：https://github.com/Midas75/AirSim/issues/1
- OpenHUTB Engine：https://github.com/OpenHUTB/engine
- OpenHUTB Air：https://github.com/OpenHUTB/air

## 项目结构

```
airsim-swarm-opt/
├── src/
│   ├── thread_benchmark/     # 独立 C++ 程序，测各线程模型的定时精度和开销
│   │   ├── benchmark.cpp
│   │   └── CMakeLists.txt
│   └── airsim_patch/         # 对 AirSim 源码的改动（diff patch 格式）
├── scripts/
│   ├── swarm_test.py         # 在 AirSim 里跑集群，采 FPS/延迟数据
│   ├── perf_monitor.py       # 监控 AirSim 进程的内核态CPU占比
│   └── batch_run.py          # 批量运行实验
├── analysis/
│   └── plot_results.py       # 画图
├── results/
│   ├── data/                 # 实验数据 CSV
│   └── figures/              # 生成的图表
└── docs/
    └── design.md             # 方案设计
```

## 五种线程模型

| 编号 | 实现方式 | 特点 |
|------|---------|------|
| M1 | 单 dispatcher + condition_variable | 系统调用只有1个线程触发，其余纯阻塞 |
| M2 | 每线程独立 `sleep_until` | 简单，比原始好，但精度受系统计时器影响 |
| M3 | `yield` 自旋 | 精度高，CPU开销随线程数线性增长 |
| M4 | `sleep(0)` 自旋（原始 AirSim 方式） | 优化对象 |
| M5 | 纯空循环自旋 | 精度最高，CPU满载，实际不可用 |

## 快速跑起来

### 依赖

```bash
pip install airsim numpy pandas matplotlib psutil
```

### 编译 C++ 基准测试

```bash
cd src/thread_benchmark
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./thread_benchmark
```

### 在 AirSim 里跑集群测试

先把 AirSim 仿真器启动，settings.json 里配好无人机数量，然后：

```bash
python scripts/swarm_test.py --num_drones 10 --model M1 --duration 30
```

批量跑所有组合：

```bash
python scripts/swarm_test.py --sweep
```

### 画图

```bash
# 有数据的情况
python analysis/plot_results.py --input results/data/swarm_summary.csv

# 先看看图表效果（用预设数据）
python analysis/plot_results.py --demo
```

## 进度

- [x] 问题定位，分析 AirSim 源码
- [x] 五种方案设计与独立基准测试
- [x] Python 集群控制与数据采集脚本
- [x] AirSim 源码修改 patch
- [ ] 完整实验数据（等环境搭好）
