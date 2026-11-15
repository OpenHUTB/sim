"""
冒烟测试占位：验证入口程序可正常解析参数并返回。
后续新增功能时，在本目录按 test_*.py 命名补充功能测试与性能测试。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import underwater_sim  # noqa: E402


class TestSmoke(unittest.TestCase):
    def test_entry_callable(self):
        # 占位阶段仅验证模块可导入、入口函数存在
        self.assertTrue(callable(underwater_sim.main))


if __name__ == "__main__":
    unittest.main()
