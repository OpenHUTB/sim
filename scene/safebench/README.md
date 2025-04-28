# SafeBench (简化版)

本目录仅保留了本项目 (ChatScene) 中使用到的 **SafeBench** 相关功能模块，以保证项目小巧、清晰。

如果需要完整体验 SafeBench，请参考官方仓库：[SafeBench GitHub](https://github.com/TRI-ML/safebench)。

---

## 目录结构说明

## 🚀 安装方法

确保你已经安装了以下依赖：

```bash
pip install gym pygame
 ```
---

## 核心文件功能说明

- **`carla_runner.py`**  
  直接连接 Carla 仿真器，快速加载静态地图、布置车辆，并执行预定义动作。  
  主要用于 **直接测试简单场景**，无需 Scenic 脚本。

- **`scenic_runner.py`**  
  读取 `.scenic` 脚本文件（描述静态场景），通过 Scenic 编译后，自动在 Carla 中搭建对应场景并执行仿真。  
  适合用于**自然语言转静态场景**的生成实验。

- **`scenic_runner_dynamic.py`**  
  支持动态场景（随时间变化的元素，如动态行人、车辆转向等），可以加载更复杂的 `.scenic` 动态脚本，在 Carla 中实时生成并控制场景。  
  主要用于**自然语言生成动态场景**，并仿真运行。

---

## 使用方法

1. **Carla环境初始化**
   - 需要提前启动 Carla Server，确保版本为 0.9.15。
   - 建议使用命令行启动：  
     ```bash
     ./CarlaUE4.sh -quality-level=Low
     ```

2. **运行 carla_runner.py**
   - 示例命令：
     ```bash
     python safebench/carla_runner.py
     ```
   - 默认加载固定场景，可修改内部配置来指定地图、交通参与者等。

3. **运行 scenic_runner.py**
   - 先准备一个静态 `.scenic` 场景文件。
   - 运行示例：
     ```bash
     python safebench/scenic_runner.py --scenario_file ./your_scenario.scenic
     ```
   - 程序会自动编译 Scenic 脚本并在 Carla 中布置场景。

4. **运行 scenic_runner_dynamic.py**
   - 适合用于需要生成动态行为的场景。
   - 运行示例：
     ```bash
     python safebench/scenic_runner_dynamic.py --scenario_file ./your_dynamic_scenario.scenic
     ```
   - 可以控制仿真时间、动作脚本等。

---
scenario

## 注意事项

- 本项目只保留了必要模块，因此**不支持** SafeBench 原版完整功能（如 Benchmark评估、对抗攻击等）。
- 仅适配 **Carla 0.9.15 + Python 3.7/3.8** 环境。
- Scenic 脚本需符合正确的语法规范，否则可能编译失败。

---

## 参考资料

- [SafeBench 官方仓库](https://github.com/TRI-ML/safebench)
- [CARLA Simulator](https://carla.org/)
- [Scenic Language 官方文档](https://scenic-lang.readthedocs.io/)

---

> 本目录仅作为 ChatScene 项目的辅助模块，建议如需进一步扩展测试功能，可参考原版 SafeBench 框架进行完善。
