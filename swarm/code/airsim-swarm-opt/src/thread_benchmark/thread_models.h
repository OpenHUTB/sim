// thread_models.h
// 各种线程等待模型的封装，供 benchmark 和 AirSim patch 复用
// 2024/10

#pragma once
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

// ---- M1: 原始忙等（对照组）----
// AirSim ClockBase 的原始实现方式
// CPU 占用极高，多机时会互相抢占
struct BusyWaitModel {
    static void sleep_for(double seconds) {
        auto end = std::chrono::high_resolution_clock::now()
                   + std::chrono::duration<double>(seconds);
        while (std::chrono::high_resolution_clock::now() < end) {
            // 什么都不做，就是死循环
        }
    }
    static const char* name() { return "M1_BusyWait"; }
};

// ---- M2: sleep_for ----
// 最简单的替换，直接睡。精度差，但 CPU 几乎不占
// 低速无人机可以用，高频控制环不行
struct SleepModel {
    static void sleep_for(double seconds) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(seconds)
        );
    }
    static const char* name() { return "M2_Sleep"; }
};

// ---- M3: 条件变量超时等待 ----
// 推荐方案。可以被外部唤醒（比如紧急停机），
// 同时 CPU 占用比 M1 低很多
struct CondVarModel {
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> interrupted{false};

    void sleep_for(double seconds) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait_for(lock,
            std::chrono::duration<double>(seconds),
            [this]{ return interrupted.load(); }
        );
    }

    void interrupt() {
        interrupted = true;
        cv.notify_all();
    }

    void reset() {
        interrupted = false;
    }

    static const char* name() { return "M3_CondVar"; }
};

// ---- M4: 自适应混合 ----
// 先睡大部分时间，剩最后一小段再忙等补精度
// 权衡：CPU 比 M1 低，精度比 M2/M3 高
// 在 AirSim 多机场景中综合表现较好
struct AdaptiveModel {
    // busyThreshold: 最后多少秒改用忙等补齐
    // 经验值 0.0005s 够了，再大就浪费 CPU
    static constexpr double busyThreshold = 0.0005;

    static void sleep_for(double seconds) {
        if (seconds <= 0) return;

        double sleepPart = seconds - busyThreshold;
        if (sleepPart > 0) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(sleepPart)
            );
        }

        // 剩余时间忙等
        auto end = std::chrono::high_resolution_clock::now()
                   + std::chrono::duration<double>(
                       seconds - sleepPart > 0 ? seconds - sleepPart : 0
                   );
        while (std::chrono::high_resolution_clock::now() < end) {}
    }

    static const char* name() { return "M4_Adaptive"; }
};

// ---- M5: 系统定时器回调 ----
// 用 timerfd（Linux）或 CreateWaitableTimer（Windows）
// 精度依赖 OS 时钟分辨率，通常 ~1ms
// 实现略复杂，这里先留个接口占位
// TODO: Windows 版用 timeBeginPeriod(1) 提升精度
struct TimerModel {
    static void sleep_for(double seconds) {
        // 暂时退化到 sleep，后面补 platform-specific 实现
        std::this_thread::sleep_for(
            std::chrono::duration<double>(seconds)
        );
    }
    static const char* name() { return "M5_Timer"; }
};
