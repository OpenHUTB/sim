// thread_benchmark.cpp
// 线程等待模型对比测试
// 2024/10

// 独立测试程序，不依赖 AirSim 环境
// 测各种线程模型的定时精度和 CPU 开销，结果用于和 AirSim 集成测试数据对比

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cassert>

using namespace std::chrono;
using Clock = high_resolution_clock;

// 测试结果结构体
struct Result {
    std::string name;
    int n_threads;
    double avg_err_us;
    double max_err_us;
    double std_us;
    long long total_iters;
};

std::mutex g_mtx;

static void print_res(const Result& r) {
    std::lock_guard<std::mutex> lk(g_mtx);
    printf("[%-22s] n=%2d  avg=%.1fus  max=%.1fus  std=%.1fus  iters=%lld\n",
           r.name.c_str(), r.n_threads,
           r.avg_err_us, r.max_err_us, r.std_us,
           r.total_iters);
}

static Result calc_stats(const std::string& name, int n,
                          const std::vector<double>& errs) {
    Result r;
    r.name = name;
    r.n_threads = n;
    r.total_iters = (long long)errs.size();
    if (errs.empty()) return r;

    double sum = 0;
    for (double e : errs) sum += e;
    r.avg_err_us = sum / errs.size();
    r.max_err_us = *std::max_element(errs.begin(), errs.end());

    double var = 0;
    for (double e : errs)
        var += (e - r.avg_err_us) * (e - r.avg_err_us);
    r.std_us = std::sqrt(var / errs.size());
    return r;
}


// ----------------------------------------------------------------
// M1: 单线程分发器
//   一个 dispatcher 用 sleep_until 等到时间，然后 notify_all 唤醒所有 worker
//   worker 本身不自旋，纯阻塞等 cv
//
// 这个是我认为最好的方案，因为系统调用只有 dispatcher 一个线程在做
// ----------------------------------------------------------------

Result bench_m1(int n_workers, microseconds interval, int dur_sec) {
    std::atomic<bool> running{true};
    std::vector<std::thread> workers;
    std::vector<double> all_errs;
    std::mutex errs_mtx;

    std::condition_variable cv;
    std::mutex cv_mtx;
    std::atomic<int> tick{0};

    std::thread dispatcher([&] {
        auto nxt = Clock::now() + interval;
        while (running.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_until(nxt);
            {
                std::lock_guard<std::mutex> lk(cv_mtx);
                tick.fetch_add(1, std::memory_order_release);
            }
            cv.notify_all();
            nxt += interval;
        }
    });

    for (int i = 0; i < n_workers; ++i) {
        workers.emplace_back([&] {
            std::vector<double> local;
            int last = 0;
            auto expected = Clock::now() + interval;
            while (running.load(std::memory_order_relaxed)) {
                std::unique_lock<std::mutex> lk(cv_mtx);
                cv.wait(lk, [&] {
                    return tick.load(std::memory_order_acquire) > last
                           || !running.load(std::memory_order_relaxed);
                });
                auto now = Clock::now();
                last = tick.load(std::memory_order_relaxed);
                lk.unlock();

                double err = std::abs((double)duration_cast<microseconds>(
                    now - expected).count());
                local.push_back(err);
                expected = now + interval;
            }
            std::lock_guard<std::mutex> lk(errs_mtx);
            all_errs.insert(all_errs.end(), local.begin(), local.end());
        });
    }

    std::this_thread::sleep_for(seconds(dur_sec));
    running.store(false, std::memory_order_release);
    cv.notify_all();

    dispatcher.join();
    for (auto& w : workers) w.join();

    return calc_stats("M1_dispatcher", n_workers, all_errs);
}


// ----------------------------------------------------------------
// M2: 标准 sleep_until
//   每个线程各自算好下一个时刻，直接 sleep_until
//   比 sleep(0) 好，但 Windows 上计时器分辨率问题会导致误差较大
// ----------------------------------------------------------------

Result bench_m2(int n, microseconds interval, int dur_sec) {
    std::atomic<bool> running{true};
    std::vector<std::thread> threads;
    std::vector<double> all_errs;
    std::mutex mtx;

    for (int i = 0; i < n; ++i) {
        threads.emplace_back([&] {
            std::vector<double> local;
            auto nxt = Clock::now() + interval;
            while (running.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_until(nxt);
                auto now = Clock::now();
                double err = std::abs((double)duration_cast<microseconds>(
                    now - nxt).count());
                local.push_back(err);
                nxt += interval;
            }
            std::lock_guard<std::mutex> lk(mtx);
            all_errs.insert(all_errs.end(), local.begin(), local.end());
        });
    }

    std::this_thread::sleep_for(seconds(dur_sec));
    running = false;
    for (auto& t : threads) t.join();

    return calc_stats("M2_sleep_until", n, all_errs);
}


// ----------------------------------------------------------------
// M3: yield 自旋
//   不睡眠，一直 yield 直到时间到
//   定时精度比较好，但线程多的时候 CPU 会被打满
//   线程数 > 10 的时候我就不跑了，太卡
// ----------------------------------------------------------------

Result bench_m3(int n, microseconds interval, int dur_sec) {
    std::atomic<bool> running{true};
    std::vector<std::thread> threads;
    std::vector<double> all_errs;
    std::mutex mtx;

    for (int i = 0; i < n; ++i) {
        threads.emplace_back([&] {
            std::vector<double> local;
            auto nxt = Clock::now() + interval;
            while (running.load(std::memory_order_relaxed)) {
                while (Clock::now() < nxt
                       && running.load(std::memory_order_relaxed))
                    std::this_thread::yield();
                auto now = Clock::now();
                double err = std::abs((double)duration_cast<microseconds>(
                    now - nxt).count());
                local.push_back(err);
                nxt += interval;
            }
            std::lock_guard<std::mutex> lk(mtx);
            all_errs.insert(all_errs.end(), local.begin(), local.end());
        });
    }

    std::this_thread::sleep_for(seconds(dur_sec));
    running = false;
    for (auto& t : threads) t.join();

    return calc_stats("M3_yield_spin", n, all_errs);
}


// ----------------------------------------------------------------
// M4: sleep(0) 自旋 —— AirSim 原始实现
//   AirSim 的 ClockBase.hpp 里就是这么写的
//   sleep_for(0) 其实还是会触发一次系统调用，N个线程就是N*100次/秒的syscall
//   无人机一多就炸了
// ----------------------------------------------------------------

Result bench_m4(int n, microseconds interval, int dur_sec) {
    std::atomic<bool> running{true};
    std::vector<std::thread> threads;
    std::vector<double> all_errs;
    std::mutex mtx;

    for (int i = 0; i < n; ++i) {
        threads.emplace_back([&] {
            std::vector<double> local;
            auto nxt = Clock::now() + interval;
            while (running.load(std::memory_order_relaxed)) {
                while (Clock::now() < nxt
                       && running.load(std::memory_order_relaxed))
                    std::this_thread::sleep_for(microseconds(0));  // AirSim原始方式
                auto now = Clock::now();
                double err = std::abs((double)duration_cast<microseconds>(
                    now - nxt).count());
                local.push_back(err);
                nxt += interval;
            }
            std::lock_guard<std::mutex> lk(mtx);
            all_errs.insert(all_errs.end(), local.begin(), local.end());
        });
    }

    std::this_thread::sleep_for(seconds(dur_sec));
    running = false;
    for (auto& t : threads) t.join();

    return calc_stats("M4_sleep0_spin(orig)", n, all_errs);
}


// ----------------------------------------------------------------
// M5: 纯自旋
//   空循环等时间到，精度最高但CPU会跑满
//   线程少的时候测一下，线程多直接跳过
// ----------------------------------------------------------------

Result bench_m5(int n, microseconds interval, int dur_sec) {
    std::atomic<bool> running{true};
    std::vector<std::thread> threads;
    std::vector<double> all_errs;
    std::mutex mtx;

    for (int i = 0; i < n; ++i) {
        threads.emplace_back([&] {
            std::vector<double> local;
            auto nxt = Clock::now() + interval;
            while (running.load(std::memory_order_relaxed)) {
                while (Clock::now() < nxt
                       && running.load(std::memory_order_relaxed)) {}
                auto now = Clock::now();
                double err = std::abs((double)duration_cast<microseconds>(
                    now - nxt).count());
                local.push_back(err);
                nxt += interval;
            }
            std::lock_guard<std::mutex> lk(mtx);
            all_errs.insert(all_errs.end(), local.begin(), local.end());
        });
    }

    std::this_thread::sleep_for(seconds(dur_sec));
    running = false;
    for (auto& t : threads) t.join();

    return calc_stats("M5_pure_spin", n, all_errs);
}


// ----------------------------------------------------------------
// 保存 CSV
// ----------------------------------------------------------------

void save_csv(const std::vector<Result>& results, const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "无法写入文件: %s\n", path.c_str());
        return;
    }
    f << "model,n_threads,avg_err_us,max_err_us,std_us,total_iters\n";
    for (const auto& r : results) {
        f << r.name << "," << r.n_threads << ","
          << std::fixed << std::setprecision(2)
          << r.avg_err_us << "," << r.max_err_us << ","
          << r.std_us << "," << r.total_iters << "\n";
    }
    printf("已保存: %s\n", path.c_str());
}


int main(int argc, char* argv[]) {
    printf("Thread Benchmark - AirSim 线程模型对比\n");
    printf("========================================\n\n");

    // 测试配置
    // interval 10ms = 100Hz，对应 AirSim 物理步长
    microseconds interval(10000);
    int dur = 5;  // 每组跑5秒，时间太长了不行

    // 线程数序列
    std::vector<int> ns = {1, 5, 10, 20, 30};

    std::vector<Result> all;

    for (int n : ns) {
        printf("\n>>> n_threads = %d\n", n);

        all.push_back(bench_m1(n, interval, dur));
        print_res(all.back());

        all.push_back(bench_m2(n, interval, dur));
        print_res(all.back());

        // M3/M5 线程多了太卡，跳过
        if (n <= 10) {
            all.push_back(bench_m3(n, interval, dur));
            print_res(all.back());
        }

        all.push_back(bench_m4(n, interval, dur));
        print_res(all.back());

        if (n <= 5) {
            all.push_back(bench_m5(n, interval, dur));
            print_res(all.back());
        }
    }

    // 存结果
    save_csv(all, "../../results/data/thread_benchmark.csv");

    printf("\n全部跑完。\n");
    return 0;
}
