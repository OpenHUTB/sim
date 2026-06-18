// thread_models.h
// 各线程等待模型封装，供 benchmark 复用
// 2024/11

#pragma once
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>

// M1: AirSim 原始忙等（对照组）
struct BusyWaitModel {
    static void sleep_for(double seconds) {
        auto end = std::chrono::high_resolution_clock::now()
                   + std::chrono::duration<double>(seconds);
        while (std::chrono::high_resolution_clock::now() < end) {}
    }
    static const char* name() { return "M1_BusyWait"; }
};

// M2: 直接 sleep_for，CPU省但精度差
struct SleepModel {
    static void sleep_for(double seconds) {
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    }
    static const char* name() { return "M2_Sleep"; }
};

// M3: 条件变量超时等待，可被外部中断
struct CondVarModel {
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> interrupted{false};

    void sleep_for(double seconds) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait_for(lock,
            std::chrono::duration<double>(seconds),
            [this] { return interrupted.load(); });
    }

    void interrupt() { interrupted = true; cv.notify_all(); }
    void reset()     { interrupted = false; }

    static const char* name() { return "M3_CondVar"; }
};

// M4: 自适应混合（先睡再忙等）
// busyThreshold 经验值 0.5ms，实测效果不错
struct AdaptiveModel {
    static constexpr double kBusyThreshold = 0.0005;

    static void sleep_for(double seconds) {
        if (seconds <= 0) return;
        double sleep_part = seconds - kBusyThreshold;
        if (sleep_part > 0)
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_part));

        auto end = std::chrono::high_resolution_clock::now()
                   + std::chrono::duration<double>(
                       seconds < kBusyThreshold ? seconds : kBusyThreshold);
        while (std::chrono::high_resolution_clock::now() < end) {}
    }

    static const char* name() { return "M4_Adaptive"; }
};

// M5: 系统定时器（暂用 sleep 占位，后面补 platform 实现）
struct TimerModel {
    static void sleep_for(double seconds) {
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        // TODO: Windows 用 CreateWaitableTimer，Linux 用 timerfd
    }
    static const char* name() { return "M5_Timer"; }
};
