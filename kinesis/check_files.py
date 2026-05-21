#!/usr/bin/env python3
"""检查Kinesis必需文件"""
import os
import sys
import joblib

def check_smpl():
    """检查SMPL模型"""
    print("=" * 70)
    print("检查 SMPL 模型")
    print("=" * 70)

    base_dir = r"D:\兼职\18\Kinesis"
    smpl_path = os.path.join(base_dir, "data", "smpl", "SMPL_NEUTRAL.pkl")

    if not os.path.exists(smpl_path):
        print(f"✗ SMPL_NEUTRAL.pkl 不存在")
        print(f"  期望位置: {smpl_path}")
        return False

    print(f"✓ 文件存在: {smpl_path}")
    print(f"  大小: {os.path.getsize(smpl_path)/1024/1024:.2f} MB")

    try:
        data = joblib.load(smpl_path)
        print(f"✓ 文件可读取")
        print(f"  类型: {type(data)}")

        if isinstance(data, dict):
            print(f"  包含 {len(data)} 个字段")
            print(f"  主要字段:")
            for i, key in enumerate(list(data.keys())[:10], 1):
                print(f"    {i}. {key}")
            if len(data) > 10:
                print(f"    ... (还有 {len(data)-10} 个)")
        return True
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        return False

def check_kit_tar():
    """检查KIT压缩包"""
    print()
    print("=" * 70)
    print("检查 KIT 数据集")
    print("=" * 70)

    base_dir = r"D:\兼职\18\Kinesis"
    kit_tar_path = os.path.join(base_dir, "data", "KIT_Data", "KIT.tar.bz2")
    kit_extract_dir = os.path.join(base_dir, "data", "KIT_Data", "KIT")

    print(f"压缩包: {kit_tar_path}")
    print(f"解压目标: {kit_extract_dir}")

    tar_exists = os.path.exists(kit_tar_path)
    dir_exists = os.path.exists(kit_extract_dir)

    if not tar_exists:
        print(f"✗ KIT.tar.bz2 不存在")
        return False

    print(f"✓ 压缩包存在: {os.path.getsize(kit_tar_path)/1024/1024:.2f} MB")

    if dir_exists:
        print(f"✓ KIT 目录已存在")
        npz_count = 0
        for root, dirs, files in os.walk(kit_extract_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_count += 1
        print(f"  包含 {npz_count} 个 NPZ 文件")
        return npz_count > 0
    else:
        print(f"✗ KIT 目录不存在（需要解压）")
        return False

def extract_kit():
    """解压KIT数据集"""
    print()
    print("=" * 70)
    print("解压 KIT 数据集")
    print("=" * 70)

    import tarfile

    base_dir = r"D:\兼职\18\Kinesis"
    kit_tar_path = os.path.join(base_dir, "data", "KIT_Data", "KIT.tar.bz2")
    kit_extract_dir = os.path.join(base_dir, "data", "KIT_Data", "KIT")

    try:
        tar = tarfile.open(kit_tar_path, 'r:bz2')
        members = tar.getmembers()
        total = len(members)

        print(f"压缩包包含 {total} 个文件")
        print("开始解压...")

        for i, member in enumerate(members, 1):
            tar.extract(member, kit_extract_dir)
            if i % 100 == 0 or i == total:
                print(f"  进度: [{i}/{total}] {i/total*100:.1f}%")

        tar.close()

        print(f"✓ 解压完成")
        print()

        npz_count = 0
        for root, dirs, files in os.walk(kit_extract_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_count += 1

        print(f"✓ 共 {npz_count} 个 NPZ 文件")
        return True

    except Exception as e:
        print(f"✗ 解压失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_kit_converted():
    """检查转换后的KIT数据"""
    print()
    print("=" * 70)
    print("检查转换后的 KIT 数据")
    print("=" * 70)

    base_dir = r"D:\兼职\18\Kinesis"
    train_pkl = os.path.join(base_dir, "data", "kit_train_motion_dict.pkl")
    test_pkl = os.path.join(base_dir, "data", "kit_test_motion_dict.pkl")

    train_exists = os.path.exists(train_pkl)
    test_exists = os.path.exists(test_pkl)

    if train_exists:
        print(f"✓ kit_train_motion_dict.pkl 存在")
        try:
            data = joblib.load(train_pkl)
            print(f"  包含 {len(data)} 个训练动作")
        except:
            pass

    else:
        print(f"✗ kit_train_motion_dict.pkl 不存在")

    if test_exists:
        print(f"✓ kit_test_motion_dict.pkl 存在")
        try:
            data = joblib.load(test_pkl)
            print(f"  包含 {len(data)} 个测试动作")
        except:
            pass

    else:
        print(f"✗ kit_test_motion_dict.pkl 不存在")

    return train_exists and test_exists

def main():
    print()
    print("=" * 70)
    print("Kinesis 项目配置检查")
    print("=" * 70)
    print()

    # 检查SMPL
    smpl_ok = check_smpl()

    # 检查KIT
    kit_tar_ok = check_kit_tar()

    print()
    print("=" * 70)
    print("下一步建议")
    print("=" * 70)
    print()

    if smpl_ok and kit_tar_ok:
        print("✅ SMPL 模型就绪")
        print("✅ KIT 压缩包就绪")
        print()
        print("建议操作:")
        print("  1. 如果KIT已解压，直接运行转换脚本")
        print("  2. 如果KIT未解压，运行: python check_files.py --extract")
        print()

        # 检查是否已转换
        converted = check_kit_converted()

        if converted:
            print("=" * 70)
            print("✅✅✅ 所有文件已就绪！")
            print("=" * 70)
            print()
            print("可以运行Kinesis测试了！")
            print()
            print("测试命令:")
            print("  目标到达:")
            print("    python src/run.py exp_name=kinesis-target-goal-reach run=eval_run learning=pointgoal epoch=-1 run.headless=True")
            print()
            print("  运动模仿:")
            print("    python src/run.py exp_name=kinesis-moe-imitation epoch=-1 run=eval_run run.headless=True run.motion_file=data/kit_test_motion_dict.pkl")
        else:
            print("=" * 70)
            print("⚠️  需要转换 KIT 数据")
            print("=" * 70)
            print()
            print("运行转换:")
            print("  python src/utils/convert_kit.py --path data/KIT_Data/KIT")

    elif not smpl_ok:
        print("⚠️  请先下载 SMPL_NEUTRAL.pkl")
    elif not kit_tar_ok:
        print("⚠️  请先下载 KIT.tar.bz2")

    print()
    print("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="解压KIT数据集")
    args = parser.parse_args()

    if args.extract:
        extract_kit()

    main()
