
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

using namespace std::chrono;
using Clock = high_resolution_clock;

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
    printf("[%-22s] n=%2d  avg=%.1fus  max=%.1fus  std=%.1fus\n",
           r.name.c_str(), r.n_threads,
           r.avg_err_us, r.max_err_us, r.std_us);
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
                    std::this_thread::sleep_for(microseconds(0));
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

    return calc_stats("M4_sleep0(baseline)", n, all_errs);
}

int main() {
    printf("Thread Benchmark - 对照组测试\n");
    printf("========================================\n\n");

    microseconds interval(10000);  // 10ms = 100Hz
    int dur = 5;

    
    std::vector<int> ns = {1, 5, 10, 20, 30};

    for (int n : ns) {
        printf("\n>>> n_threads = %d\n", n);
        auto r = bench_m4(n, interval, dur);
        print_res(r);

       
        if (n >= 20) {
            printf("    (注意: n=%d 时系统明显卡顿，这就是需要优化的原因)\n", n);
        }
    }

    printf("\n基准测试完成，后续加入对比方案。\n");
    return 0;
}
