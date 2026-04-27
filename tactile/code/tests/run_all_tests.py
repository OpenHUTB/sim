"""
测试运行器 - 运行所有测试并生成报告

作者: [你的名字]
日期: 2025
"""

import unittest
import sys
import time
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests():
    """运行所有测试"""
    # 发现所有测试
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed_time = time.time() - start_time
    
    # 打印摘要
    print("\n" + "="*70)
    print("测试摘要")
    print("="*70)
    print(f"运行时间: {elapsed_time:.2f} 秒")
    print(f"测试总数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print("\n✗ 测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
