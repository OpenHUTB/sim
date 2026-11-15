# comm —— 基于水下工作环境的传感器与通信链路仿真系统

本模块为本科毕业设计（论文）代码与论文目录，对应题目：**基于水下工作环境的传感器与通信链路仿真系统**。

- GitHub 账号：thirtyone-o
- 搭档：joe-justin、heshuyun9
- 选题关键词：噪声、延迟、漂移、遮挡、低能见度、声呐成像；支撑算法和操作训练

## 目录结构
```
comm/
├── code/                  # 仿真代码
│   ├── assets/             # 模型、配置等资源文件
│   ├── tests/             # 测试代码（功能测试、性能测试）
│   └── comm_sim.py        # 模块唯一入口程序（占位，待实现）
└── undergraduate/         # 毕业论文（LaTeX）
    ├── hutbthesis_main.tex # 论文编译主文件
    ├── content/            # 章节、摘要、参考文献
    └── images/             # 论文插图
```

## 环境与规范
- 仿真平台、依赖版本以仓库根目录 `requirements.txt`、`config.ini` 与导师要求为准。
- 代码中统一使用相对路径，不写绝对路径；依赖写入 `code/requirements.txt` 并指定版本。
- 所有功能须在 Town01、Town10 场景下测试通过，并经另一名同学在其电脑上运行验证、截图留档后，才可合并到主分支。
- 论文中只允许写入本人已测试通过的功能与数据；未经验证的内容不得写入。

## 参考链接（选题表 H 列）
- 示例·水域载具：https://openhutb.github.io/ros2/#water_vehicle
- 运动控制：https://mp.weixin.qq.com/s/RLZHx11olWrYezoFQ4E9lw
- 水下方向文档：https://openhutb.github.io/mujoco_plugin/underwater/
- 水下机器人仿真参考代码：https://github.com/open-airlab/UNav-Sim
- 模拟器主仓库：https://github.com/OpenHUTB/mcp
- 平台文档主页：https://openhutb.github.io/
