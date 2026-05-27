#!/usr/bin/env python3
import os
import sys
import tarfile

def main():
    base_dir = r"D:\兼职\18\Kinesis"

    print("=" * 70)
    print("Kinesis 自动配置")
    print("=" * 70)
    print()

    # 1. 检查 SMPL
    print("[1/3] 检查 SMPL 模型...")
    smpl_path = os.path.join(base_dir, "data", "smpl", "SMPL_NEUTRAL.pkl")

    if os.path.exists(smpl_path):
        print("✓ SMPL_NEUTRAL.pkl 存在")
        print(f"  大小: {os.path.getsize(smpl_path)/1024/1024:.2f} MB")
        try:
            import joblib
            data = joblib.load(smpl_path)
            print(f"  ✓ 可读取，包含 {len(data)} 个字段")
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
    else:
        print("✗ SMPL_NEUTRAL.pkl 不存在")

    print()

    # 2. 检查 KIT
    print("[2/3] 检查 KIT 数据集...")
    kit_tar = os.path.join(base_dir, "data", "KIT_Data", "KIT.tar.bz2")
    kit_dir = os.path.join(base_dir, "data", "KIT_Data", "KIT")

    if os.path.exists(kit_dir):
        npz_count = 0
        for root, dirs, files in os.walk(kit_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_count += 1
        print(f"✓ KIT 目录存在，包含 {npz_count} 个 NPZ 文件")
    elif os.path.exists(kit_tar):
        print("✓ KIT.tar.bz2 存在")
        print("  需要解压...")
        try:
            tar = tarfile.open(kit_tar, 'r:bz2')
            members = tar.getmembers()
            print(f"  压缩包包含 {len(members)} 个文件")
            print("  开始解压...")
            tar.extractall(os.path.dirname(kit_dir))
            tar.close()

            npz_count = 0
            for root, dirs, files in os.walk(kit_dir):
                for f in files:
                    if f.endswith('.npz'):
                        npz_count += 1
            print(f"✓ 解压完成，包含 {npz_count} 个 NPZ 文件")
        except Exception as e:
            print(f"✗ 解压失败: {e}")
    else:
        print("✗ KIT 数据集不存在")

    print()

    # 3. 检查转换结果
    print("[3/3] 检查转换后的数据...")
    train_pkl = os.path.join(base_dir, "data", "kit_train_motion_dict.pkl")
    test_pkl = os.path.join(base_dir, "data", "kit_test_motion_dict.pkl")

    train_ok = os.path.exists(train_pkl)
    test_ok = os.path.exists(test_pkl)

    if train_ok and test_ok:
        try:
            import joblib
            train_data = joblib.load(train_pkl)
            test_data = joblib.load(test_pkl)
            print(f"✓ kit_train_motion_dict.pkl 存在（{len(train_data)} 个动作）")
            print(f"✓ kit_test_motion_dict.pkl 存在（{len(test_data)} 个动作）")
        except:
            print(f"✓ kit_train_motion_dict.pkl 存在")
            print(f"✓ kit_test_motion_dict.pkl 存在")
    else:
        if train_ok:
            print("✓ kit_train_motion_dict.pkl 存在")
        else:
            print("✗ kit_train_motion_dict.pkl 不存在")

        if test_ok:
            print("✓ kit_test_motion_dict.pkl 存在")
        else:
            print("✗ kit_test_motion_dict.pkl 不存在")

    print()
    print("=" * 70)
    print("配置总结")
    print("=" * 70)

    smpl_ready = os.path.exists(smpl_path)
    kit_ready = os.path.exists(kit_dir) and (os.listdir(kit_dir) if os.path.exists(kit_dir) else 0)
    converted = train_ok and test_ok

    if smpl_ready and converted:
        print("✅ 所有文件已就绪！")
        print()
        print("可以运行 Kinesis 测试了！")
        print()
        print("测试命令:")
        print("  1. 目标到达:")
        print("     python src/run.py exp_name=kinesis-target-goal-reach run=eval_run learning=pointgoal epoch=-1 run.headless=True")
        print()
        print("  2. 运动模仿:")
        print("     python src/run.py exp_name=kinesis-moe-imitation epoch=-1 run=eval_run run.headless=True")
        print()
    elif smpl_ready and kit_ready and not converted:
        print("✅ SMPL 和 KIT 就绪")
        print("⚠️  需要转换 KIT 数据")
        print()
        print("运行转换命令:")
        print("  python src/utils/convert_kit.py --path data/KIT_Data/KIT")
        print()
    else:
        print("⚠️  还有文件缺失")
        if not smpl_ready:
            print("  ✗ SMPL 模型")
        if not kit_ready:
            print("  ✗ KIT 数据集")

    print("=" * 70)

if __name__ == "__main__":
    main()
