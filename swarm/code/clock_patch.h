// clock_patch.h
// 替换 AirSim ClockBase::sleep_for 的优化版本
// 对应 AirLib/include/common/ClockBase.hpp
// 2024/11

#pragma once
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <atomic>

// 用法：把 AirSim 里 SteppableClock::sleep_for 的实现换成这里的
// 或者用 thread_optimization.patch 直接打补丁

namespace airsim_patch {

// 自适应方案：先 sleep 大部分，最后 500us 忙等补精度
// 实测 i7-12700H 上误差 < 0.3ms，内核态 CPU 从 ~35% 降到 ~9%
class OptimizedClock {
public:
    static void sleep_for_us(long long us) {
        if (us <= 0) return;

        auto target = std::chrono::high_resolution_clock::now()
                      + std::chrono::microseconds(us);

        // 留 500us 给最后的忙等
        constexpr long long kBusyUs = 500;
        long long sleep_us = us - kBusyUs;

        if (sleep_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }

        // 最后一段忙等，保证定时精度
        while (std::chrono::high_resolution_clock::now() < target) {
            std::this_thread::yield();
        }
    }

    static void sleep_for(double seconds) {
        sleep_for_us(static_cast<long long>(seconds * 1e6));
    }
};

// 可中断版本，支持外部 stop 信号（比如紧急停机）
class InterruptibleClock {
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};

public:
    void sleep_for(double seconds) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait_for(lk,
            std::chrono::duration<double>(seconds),
            [this] { return stop_.load(); });
    }

    void stop() {
        stop_ = true;
        cv_.notify_all();
    }

    void reset() { stop_ = false; }
};

} // namespace airsim_patch
