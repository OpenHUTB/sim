# 基于 CARLA 的极端驾驶仿真场景生成算法研究
## 项目简介

研究并实现极端驾驶场景自动生成算法，定义 10 种典型危险驾驶场景，支持手动构建与强化学习（DQN / PPO 系列）自动生成对抗场景，输出标准 OpenSCENARIO (.xosc) 文件。

## 技术栈

- 仿真平台：hutb (CARLA 2.9.16)
- 语言：Python 3.7
- 强化学习：DQN、Attention-DQN、PPO、Smooth-PPO
- 场景标准：ASAM OpenSCENARIO 1.2

## 项目结构

```
├── config/              全局固定参数
├── env/                 仿真交互环境 + 通用奖励
├── utils/               工具函数（传感器/评估/几何）
├── scenarios/           10 种危险场景定义
├── rl_algorithms/       强化学习算法
├── experiments/         训练/评估/对比入口
├── osc_exporter/        OpenSCENARIO 导出      
├── tests/               单元测试
└── main.py              统一入口
```

## 10 种危险场景

| # | 场景 | 类别 | 核心危险 |
|---|------|------|----------|
| 1 | 暴雨跟车 | 极端天气 | 湿滑路面 + 低能见度 |
| 2 | 浓雾巡航 | 极端天气 | 能见度 < 20m |
| 3 | 夜间黑暗行驶 | 极端天气 | 光照 5% |
| 4 | 前车急刹 | 车辆对抗 | 前车 -8m/s² 急刹 |
| 5 | 旁车加塞 | 车辆对抗 | 强行切入自车车道 |
| 6 | 行人横穿 | 行人危险 | 人行道突然横穿 |
| 7 | 鬼探头 | 行人危险 | 货车盲区突然冲出 |
| 8 | 行人闯红灯 | 行人危险 | 违规横穿 |
| 9 | 夜间行人横穿 | 多因素耦合 | 黑暗 + 行人 |
| 10 | 雾天鬼探头 | 多因素耦合 | 浓雾 + 盲区 |

## 快速开始

```bash
# 1. 启动 CARLA 仿真器
双击 CarlaUE4.exe

# 2. 运行项目
python main.py

# 或命令行模式
python main.py train --algo dqn --scenario rain_storm --episodes 500
python main.py evaluate --algo smooth_ppo --scenario emergency_brake --model models/xxx.pth
python main.py export --scenario ghost_peek
python main.py test --suite all
```

## 环境要求

- Windows 10 / Ubuntu 18.04+
- CARLA 0.9.16 (hutb 定制版)
- Python 3.7
- PyTorch + NumPy + PyYAML + matplotlib

## 百度网盘链接
- 通过网盘分享的文件：carla
- 链接: https://pan.baidu.com/s/1PIzndyc4LrLxvKNImyH9CA?pwd=shju 提取码: shju 