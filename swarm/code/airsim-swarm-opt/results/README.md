# results/

存放实验结果数据和图表。

## 目录结构

```
results/
├── data/          # 原始数据 CSV，由 swarm_test.py 生成
│   ├── swarm_summary.csv      # 各组实验汇总
│   └── raw/                   # 每次实验的原始帧数据
└── figures/       # plot_results.py 生成的图表
    ├── fps_comparison.png
    ├── cpu_comparison.png
    └── latency_comparison.png
```

## 数据格式

`swarm_summary.csv` 列说明：

| 列名 | 说明 |
|---|---|
| num_drones | 无人机数量 |
| model | 线程模型（M1~M5） |
| avg_fps | 平均帧率 |
| min_fps | 最低帧率（卡顿时） |
| avg_latency_ms | 平均指令延迟(ms) |
| cpu_kernel_pct | CPU 内核态占比(%) |
| cpu_user_pct | CPU 用户态占比(%) |
| duration_s | 测试时长(s) |

## 说明

`data/` 目录下的 `.gitkeep` 是占位文件，实际数据因为文件太大不上传。
要复现数据，跑 `scripts/batch_run.py` 就会自动生成。
