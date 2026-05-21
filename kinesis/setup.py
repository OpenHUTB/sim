#!/usr/bin/env python3
"""Kinesis 自动设置脚本 - 解压KIT数据并验证SMPL"""
import os
import tarfile
import shutil
import sys
import joblib

def extract_bz2(tar_path, extract_to):
    """解压 .tar.bz2 文件"""
    print(f"正在解压: {tar_path}")
    print(f"目标位置: {extract_to}")
    print()

    try:
        # 打开 bz2 压缩的 tar 文件
        tar = tarfile.open(tar_path, 'r:bz2')

        # 获取所有成员
        members = tar.getmembers()
        total_files = len(members)
        print(f"压缩包包含 {total_files} 个文件")

        # 解压
        for i, member in enumerate(members, 1):
            tar.extract(member, extract_to)
            if i % 100 == 0 or i == total_files:
                print(f"  进度: [{i}/{total_files}] {i/total_files*100:.1f}%")

        tar.close()
        print()
        print(f"✓ 解压完成！")
        print()

        return True

    except Exception as e:
        print(f"✗ 解压失败: {e}")
        return False

def check_smpl_file(smpl_path):
    """检查 SMPL 文件"""
    print("=" * 70)
    print("检查 SMPL 模型文件")
    print("=" * 70)
    print()

    if not os.path.exists(smpl_path):
        print(f"✗ 文件不存在: {smpl_path}")
        return False

    print(f"✓ 文件存在: {smpl_path}")
    print(f"  大小: {os.path.getsize(smpl_path)/1024/1024:.2f} MB")

    try:
        data = joblib.load(smpl_path)
        print(f"✓ 文件可读取")
        print(f"  类型: {type(data)}")

        if isinstance(data, dict):
            print(f"  包含字段: {len(data)} 个")
            print("  主要字段:")
            for key in list(data.keys())[:5]:
                print(f"    - {key}")
            if len(data) > 5:
                print(f"    ... (还有 {len(data)-5} 个)")
        print()
        return True

    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        return False

def check_kit_directory(kit_dir):
    """检查 KIT 目录"""
    print("=" * 70)
    print("检查 KIT 数据集")
    print("=" * 70)
    print()

    if not os.path.exists(kit_dir):
        print(f"✗ 目录不存在: {kit_dir}")
        return False

    print(f"✓ 目录存在: {kit_dir}")

    # 统计 NPZ 文件数量
    npz_files = []
    for root, dirs, files in os.walk(kit_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))

    print(f"✓ 找到 {len(npz_files)} 个 NPZ 文件")

    if len(npz_files) == 0:
        print("✗ 没有找到 NPZ 文件！")
        print()
        print("可能的原因:")
        print("  1. 还未解压")
        print("  2. 解压到了其他位置")
        print("  3. 下载的文件格式不对")
        return False

    print()

    # 显示前几个 NPZ 文件
    print("前 5 个 NPZ 文件:")
    for i, npz_file in enumerate(npz_files[:5], 1):
        rel_path = os.path.relpath(npz_file, kit_dir)
        print(f"  {i}. {rel_path}")

    if len(npz_files) > 5:
        print(f"  ... (还有 {len(npz_files)-5} 个)")

    print()
    return len(npz_files) > 0

def main():
    base_dir = r"D:\兼职\18\Kinesis"

    # 文件路径
    smpl_path = os.path.join(base_dir, "data", "smpl", "SMPL_NEUTRAL.pkl")
    kit_tar_path = os.path.join(base_dir, "data", "KIT_Data", "KIT.tar.bz2")
    kit_extract_dir = os.path.join(base_dir, "data", "KIT_Data", "KIT")

    print("=" * 70)
    print("Kinesis 自动设置")
    print("=" * 70)
    print()

    # 步骤1：检查 SMPL 文件
    smpl_ok = check_smpl_file(smpl_path)

    print()

    # 步骤2：检查并解压 KIT 数据集
    print("=" * 70)
    print("KIT 数据集处理")
    print("=" * 70)
    print()

    if os.path.exists(kit_extract_dir):
        print(f"✓ KIT 目录已存在: {kit_extract_dir}")
        print(f"  跳过解压，直接检查...")

        kit_ok = check_kit_directory(kit_extract_dir)
    elif os.path.exists(kit_tar_path):
        print(f"✓ 找到 KIT 压缩包: {kit_tar_path}")
        print()

        # 解压
        extract_ok = extract_bz2(kit_tar_path, kit_extract_dir)

        if extract_ok:
            # 检查解压结果
            kit_ok = check_kit_directory(kit_extract_dir)
        else:
            kit_ok = False
    else:
        print(f"✗ 没有找到 KIT 压缩包: {kit_tar_path}")
        print(f"  也没找到解压目录: {kit_extract_dir}")
        kit_ok = False

    print()

    # 总结
    print("=" * 70)
    print("设置完成总结")
    print("=" * 70)
    print()

    if smpl_ok:
        print("✅ SMPL 模型: 就绪")
    else:
        print("❌ SMPL 模型: 缺失或损坏")

    if kit_ok:
        print("✅ KIT 数据集: 就绪")
    else:
        print("❌ KIT 数据集: 缺失或未解压")

    print()

    # 下一步
    if smpl_ok and kit_ok:
        print("=" * 70)
        print("✅ 所有文件已就绪！")
        print("=" * 70)
        print()
        print("下一步：运行转换脚本")
        print("  命令:")
        print(f"    cd {base_dir}")
        print(f"    python src/utils/convert_kit.py --path {kit_extract_dir}")
        print()
    else:
        print("=" * 70)
        print("⚠️  还有文件需要处理")
        print("=" * 70)
        print()
        print("请检查上面显示的问题。")

    print("=" * 70)

if __name__ == "__main__":
    main()
