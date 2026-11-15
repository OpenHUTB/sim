"""
基于水下工作环境的传感器与通信链路仿真系统 —— 模块入口程序

说明：
- 本文件是 comm 模块唯一的 Python 入口程序，其他功能模块在包内拆分，由本文件调度。
- 当前为开题阶段占位框架，具体仿真功能待实现；未在本地与同学电脑上测试通过的功能不要合并到主分支。
- 代码中统一使用相对路径，依赖统一写入本目录 requirements.txt 并指定版本。

运行方式（待依赖装好后）：
    python comm_sim.py --scene Town01
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Underwater sensor and communication link simulation system")
    parser.add_argument("--scene", default="Town01", help="仿真场景名称（如 Town01 / Town10）")
    parser.add_argument("--config", default="../config.ini", help="配置文件路径（相对路径）")
    return parser.parse_args()


def main():
    args = parse_args()
    print("[comm] 仿真系统入口（占位）")
    print(f"[comm] 场景: {args.scene}")
    print("[comm] 待实现：水下环境建模、传感器仿真、通信链路仿真、声呐成像与交互界面。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
