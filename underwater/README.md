# underwater —— 基于水下工作环境的传感器与通信链路仿真系统

本模块为本科毕业设计（论文）代码与论文目录，对应题目：**基于水下工作环境的传感器与通信链路仿真系统**。

## 目录结构
```
underwater/
├── code/                  # 仿真代码
│   ├── assets/             # 模型、配置等资源文件
│   ├── tests/             # 测试代码（功能测试、性能测试）
│   └── underwater_sim.py   # 模块唯一入口程序（占位，待实现）
└── undergraduate/         # 毕业论文（LaTeX）
    ├── hutbthesis_main.tex # 论文编译主文件
    ├── content/            # 章节、摘要、参考文献
    └── images/             # 论文插图
```

## 环境与规范
- 仿真平台、依赖版本以仓库根目录 `requirements.txt`、`config.ini` 与导师要求为准。
- 代码中统一使用相对路径，不写绝对路径；依赖写入根目录 `requirements.txt` 并指定版本。
- 所有功能须在指定仿真场景下测试通过，并经另一名同学在其电脑上运行验证、截图留档后，才可合并到主分支。
- 论文中只允许写入本人已测试通过的功能与数据；未经验证的内容不得写入。

## 参考链接
- 仿真平台文档主页：https://openhub.github.io/
- 水下方向参考文档：https://openhub.github.io/muoco_plugin/underwater/
- 水下机器人仿真参考代码：https://github.com/open-airlab/UNav-Sim
