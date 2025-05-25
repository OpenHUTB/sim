# 基于预训练大模型的高保真驾驶场景生成系统

## 项目简介

本项目旨在利用预训练大模型生成高保真驾驶场景，以支持自动驾驶车辆的安全性测试和验证。通过结合预训练大模型的强大生成能力和专业的仿真工具，我们能够生成多样化的驾驶场景，从而提高自动驾驶系统的安全性和可靠性。

## 项目地址

[GitHub - zrx0829222/scene: 郑睿翔](https://github.com/zrx0829222/scene.git)

## 环境配置

支持和测试的平台包括：Windows 11 

### 安装步骤

1. **安装依赖软件**
   - 下载并安装 Python 3.8、Carla 0.9.13、latex 2023、Texstudio 4.6.4、Git 2.42.0（Windows可使用 `TortoiseGit 2.15.0.0` 作为图形界面进行代码提交）。
   - 安装 Carla 的 Python API：
     ```bash
     export CARLA_ROOT={path/to/your/carla}
     export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
     export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
     export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
     export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
     ```

2. **克隆项目**
   ```bash
   git clone https://github.com/zrx0829222/scene.git
   cd scene

3. **安装依赖库**
   ```bash
   pip install -r requirements.txt

4. **运行场景生成脚本**
   ```bash
   python retrieve.py，run_train_dynamic.py和run_eval_dynamic.py
   ```

5. **查看生成的场景**
   - 生成的场景将保存在 `output` 目录下，场景信息在 `log` 目录下，每个场景包含一个 `.json` 文件和一个 `.txt` 文件，分别描述了场景的配置和生成的日志。
