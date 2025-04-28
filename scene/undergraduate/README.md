# Scene 项目

## 项目介绍

Scene 项目是一个基于预训练大模型的高保真三维智能驾驶场景生成系统。该系统利用自然语言描述自动生成驾驶场景，重点在于自动化场景生成、车载环境交互以及驾驶性能评估。项目基于 **CARLA** 仿真平台，结合 **SafeBench** 和 **Scenic**，实现了高效的交通场景生成与智能驾驶评估。

## 主要功能

- 使用自然语言描述生成高保真智能驾驶场景。
- 集成 GPT-4o 与检索数据库，动态生成多样化的场景。
- 评估自动驾驶系统在不同场景下的表现，包括安全性、碰撞风险、驾驶决策等。
- 基于 **SafeBench** 提供的环境进行场景的训练与评估。

## 安装指南

### 1. 创建虚拟环境

本项目推荐使用 Python 3.7 或 3.8 版本，首先创建并激活虚拟环境。

- **carla** 虚拟环境（Python 3.7）：

    ```bash
    conda create -n carla python=3.7
    conda activate carla
    ```

- **chatscene** 虚拟环境（Python 3.8）：

    ```bash
    conda create -n chatscene python=3.8
    conda activate chatscene
    ```

### 2. 安装依赖

本项目使用了 **CARLA 0.9.15**，需要先安装 CARLA，并确保 Python 环境正确配置。

#### 安装 CARLA

1. 下载并解压 **CARLA 0.9.15** 版本：[CARLA 0.9.15 下载链接](https://github.com/carla-simulator/carla/releases/tag/0.9.15)。
2. 设置环境变量，指向 CARLA 的 Python API 路径：

    ```bash
    export CARLA_ROOT={path/to/your/carla}
    export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
    export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
    export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
    export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
    ```

3. 安装必要的依赖：

    ```bash
    sudo apt install libomp5
    ```

#### 安装项目依赖

1. 克隆项目仓库：

    ```bash
    git clone git@github.com:zrx0829222/scene.git
    cd scene
    ```

2. 安装所需的 Python 包：

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

3. 安装 **Scenic** 包：

    ```bash
    cd Scenic
    python -m pip install -e .
    ```

## 运行场景

### 启动 CARLA

根据操作系统选择启动模式：

1. **桌面用户**：

    ```bash
    ./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000
    ```

2. **远程服务器用户**（无头模式）：

    ```bash
    ./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000
    ```

### 训练和评估场景

#### 模式 1: 训练场景

使用以下命令开始训练场景：

```bash
python scripts/run_train.py --agent_cfg=adv_scenic.yaml --scenario_cfg=train_scenario_scenic.yaml --mode train_scenario --scenario_id 1
```

### 动态场景生成
使用自然语言生成动态场景，可以通过修改 retrieve/scenario_descriptions.txt 文件来提供场景描述。然后使用以下命令生成场景：
```bash
python retrieve/retrieve.py
```
生成的场景数据将存放在 safebench/scenario/scenario_data/scenic_data/dynamic_scenario 目录下。
###量化评估
本项目提供了量化评估脚本，用于评估生成场景的质量，确保场景的多样性和驾驶性能。使用以下命令进行评估：
 ```bash
    python scripts/evaluate_scene_quality.py
    ```
#### 联系方式
如果你有任何问题或建议，可以通过以下方式联系我：

邮箱：2444819612@qq.com

