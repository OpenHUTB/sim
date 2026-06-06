#!/usr/bin/env python3
"""测试smplx库支持的SMPL文件格式"""
import os
import sys

def test_smplx_formats():
    print("=" * 70)
    print("测试 smplx 库对SMPL文件格式的支持")
    print("=" * 70)
    print()

    try:
        from smplx import SMPL, SMPLH, SMPLX
        print("✓ smplx 库已安装")
        print()

        # 查看 SMPL 类的文档
        print("SMPL 类初始化参数:")
        print("  model_path: 模型文件路径（可以是文件或文件夹）")
        print("  model_type: 模型类型（'smpl', 'smplh', 'smplx'）")
        print("  gender: 性别（'male', 'female', 'neutral'）")
        print()

        print("支持的文件格式:")
        print("  1. PKL 格式: 标准的SMPL模型文件（.pkl）")
        print("  2. 文件夹: 包含模型文件的文件夹")
        print("  3. NPZ 格式: 如果包含SMPL参数的NPZ文件")
        print()

        print("=" * 70)
        print("结论:")
        print("=" * 70)
        print()
        print("smplx 库主要设计用于加载 .pkl 格式的SMPL模型。")
        print()
        print("如果您有 SMPL_NEUTRAL.npz 文件：")
        print("  选项1: 将 NPZ 转换为 PKL 格式")
        print("  选项2: 尝试直接使用（可能不成功）")
        print("  选项3: 下载标准的 .pkl 文件")
        print()

        return True

    except ImportError as e:
        print(f"✗ smplx 库未安装: {e}")
        return False

def convert_npz_to_pkl(npz_path, pkl_path):
    """尝试将NPZ文件转换为PKL格式"""
    import numpy as np
    import joblib

    print(f"尝试转换 {npz_path} -> {pkl_path}")
    print()

    try:
        # 加载NPZ
        data = np.load(npz_path, allow_pickle=True)
        print("✓ NPZ文件已加载")
        print(f"  包含的键: {data.files}")
        print()

        # 尝试提取SMPL参数
        smpl_data = {}
        for key in data.files:
            value = data[key]
            smpl_data[key] = value

        # 保存为PKL
        joblib.dump(smpl_data, pkl_path)
        print(f"✓ 已保存为 PKL 格式: {pkl_path}")
        print()

        return True

    except Exception as e:
        print(f"✗ 转换失败: {e}")
        print()
        return False

if __name__ == "__main__":
    # 测试smplx
    if not test_smplx_formats():
        sys.exit(1)

    # 询问用户
    print("=" * 70)
    print("您的 SMPL_NEUTRAL.npz 文件在哪里？")
    print("=" * 70)
    print()
    print("请告诉我文件的完整路径，例如：")
    print("  C:\\Users\\用户名\\Downloads\\SMPL_NEUTRAL.npz")
    print("  D:\\Downloads\\SMPL_NEUTRAL.npz")
    print()
    print("然后我会帮您：")
    print("  1. 检查文件内容")
    print("  2. 尝试转换为 PKL 格式")
    print("  3. 复制到正确位置")
    print()
