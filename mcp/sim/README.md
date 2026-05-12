# HUTB MCP - 基于 FastMCP 的人车仿真器交互控制系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastMCP](https://img.shields.io/badge/FastMCP-Latest-brightgreen)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

基于 FastMCP 框架的 HUTB 人车仿真器交互控制系统，支持通过 MCP 协议与 AI 助手进行自然语言交互。

## 功能特性

- 车辆控制: 生成、销毁、自动驾驶、手动控制、状态获取
- 无人机控制: 生成、飞行控制、悬停、降落
- 编辑器控制: Actor 管理、场景编辑、地图加载
- 天气控制: 天气参数设置、预设天气、极端天气
- 传感器管理: 相机、激光雷达、雷达的创建和管理

## 快速开始

### 1. 环境准备

```bash
conda create -n hutb-mcp python=3.10 --yes
conda activate hutb-mcp
pip install D:/hutb/PythonAPI/carla/dist/hutb-2.9.16-cp310-cp310-win_amd64.whl
cd mcp-main/sim
pip install -e .
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env 设置 HUTB_HOST 和 HUTB_PORT
```

### 3. 启动服务

```bash
# 先启动 HUTB/CARLA 仿真器
python -m hutb_mcp
```

## MCP 客户端配置

```json
{
  "mcpServers": {
    "hutb-mcp": {
      "command": "python",
      "args": ["-m", "hutb_mcp"],
      "env": {
        "HUTB_HOST": "localhost",
        "HUTB_PORT": "2000"
      }
    }
  }
}
```

## 项目结构

```
hutb_mcp/
├── __init__.py
├── __main__.py
├── server.py
├── connection.py
├── config.py
├── tools/
│   ├── vehicle_tools.py
│   ├── air_tools.py
│   ├── editor_tools.py
│   ├── weather_tools.py
│   └── sensor_tools.py
└── utils/
    └── logger.py
```

## 参考

- [FastMCP](https://github.com/jlowin/fastmcp)
- [CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/)
