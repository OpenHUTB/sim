# 方案设计记录

2024/10

---

## 1. 问题在哪

翻了 AirSim 源码 `AirLib/include/common/ClockBase.hpp`，找到了核心问题：

```cpp
virtual void sleep_for(TTimeDelta dt) {
    TTimePoint end = now() + dt;
    while (now() < end) {
        std::this_thread::sleep_for(
            std::chrono::duration<int64_t, std::nano>(0));
    }
}
```

每架无人机是一个仿真线程，每个线程在等待下一个物理步长（10ms）的时候，会在 while 里不停地调 `sleep_for(0)`。

`sleep_for(0)` 语义上是立刻返回，但实际上每次调用都要经过内核，触发一次用户态→内核态→用户态的切换。

30 架无人机 × 100次/秒 = 3000 次/秒的系统调用，超过 CPU 核心数之后调度器开始频繁切换线程上下文，内核态 CPU 时间暴涨。

## 2. 接下来打算怎么做

设计几种替代方案，先独立测定时精度和 CPU 开销，再放到 AirSim 里验证实际效果。

具体方案后面补充。
