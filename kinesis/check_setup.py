#!/usr/bin/env python3
import os
import tarfile
import sys

def check_smpl():
    print("=" * 70)
    print("[1/3] 检查 SMPL 模型")
    print("=" * 70)

    base_dir = r"D:\兼职\18\Kinesis"
    smpl_path = os.path.join(base_dir, "data", "smpl", "SMPL_NEUTRAL.pkl")

    if not os.path.exists(smpl_path):
        print(f"  状态: ✗ 不存在")
        print(f"  期望路径: {smpl_path}")
        return False

    print(f"  状态: ✓ 存在")
    print(f"  大小: {os.path.getsize(smpl_path)/1024/1024:.2f} MB")

    try:
        import joblib
        data = joblib.load(smpl_path)
        print(f"  可读性: ✓ 可读取")
        print(f"  类型: {type(data)}")
        print(f"  字段数: {len(data)}")
        return True
    except Exception as e:
        print(f"  读取: ✗ 失败 ({e})")
        return False

def check_kit():
    print()
    print("=" * 70)
    print("[2/3] 检查 KIT 数据集")
    print("=" * 70)

    base_dir = r"D:\兼职\18\Kinesis"
    kit_tar = os.path.join(base_dir, "data", "KIT_Data", "KIT.tar.bz2")
    kit_dir = os.path.join(base_dir, "data", "KIT_Data", "KIT")

    tar_exists = os.path.exists(kit_tar)
    dir_exists = os.path.exists(kit_dir)

    if not tar_exists:
        print(f"  压缩包: ✗ 不存在 ({kit_tar})")
        return False

    print(f"  压缩包: ✓ 存在 ({os.path.getsize(kit_tar)/1024/1024:.2f} MB)")

    if dir_exists:
        npz_count = 0
        for root, dirs, files in os.walk(kit_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_count += 1
        print(f"  解压目录: ✓ 存在")
        print(f"  NPZ文件数: {npz_count}")
        return npz_count > 0
    else:
        print(f"  解压目录: ✗ 不存在 ({kit_dir})")
        print(f"  需要解压: 是")
        return False

def extract_kit():
    print()
    print("=" * 70)
    print("解压 KIT 数据集")
    print("=" * 70)

    base_dir = r"D:\兼职\18\Kinesis"
    kit_tar = os.path.join(base_dir, "data", "KIT_Data", "KIT.tar.bz2")
    kit_dir = os.path.join(base_dir, "data", "KIT_Data", "KIT")

    try:
        tar = tarfile.open(kit_tar, 'r:bz2')
        members = tar.getmembers()
        total = len(members)

        print(f"  压缩包: {kit_tar}")
        print(f"  文件数: {total}")
        print(f"  目标目录: {kit_dir}")
        print(f"  开始解压...")

        for i, member in enumerate(members, 1):
            tar.extract(member, kit_dir)
            if i % 100 == 0 or i == total:
                print(f"  进度: [{i}/{total}] ({i/total*100:.1f}%)")

        tar.close()
        print(f"  状态: ✓ 解压完成")

        npz_count = 0
        for root, dirs, files in os.walk(kit_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_count += 1

        print(f"  NPZ文件: {npz_count}")
        return True

    except Exception as e:
        print(f"  状态: ✗ 失败 ({e})")
        import traceback
        traceback.print_exc()
        return False

def check_converted():
    print()
    print("=" * 70)
    print("[3/3] 检查转换后的数据")
    print("=" * 70)

    base_dir = r"D:\兼职\18\Kinesis"
    train_pkl = os.path.join(base_dir, "data", "kit_train_motion_dict.pkl")
    test_pkl = os.path.join(base_dir, "data", "kit_test_motion_dict.pkl")

    train_exists = os.path.exists(train_pkl)
    test_exists = os.path.exists(test_pkl)

    if train_exists:
        try:
            import joblib
            data = joblib.load(train_pkl)
            print(f"  kit_train_motion_dict.pkl: ✓ 存在 ({len(data)} 个动作)")
        except:
            print(f"  kit_train_motion_dict.pkl: ✓ 存在")
    else:
        print(f"  kit_train_motion_dict.pkl: ✗ 不存在")

    if test_exists:
        try:
            import joblib
            data = joblib.load(test_pkl)
            print(f"  kit_test_motion_dict.pkl: ✓ 存在 ({len(data)} 个动作)")
        except:
            print(f"  kit_test_motion_dict.pkl: ✓ 存在")
    else:
        print(f"  kit_test_motion_dict.pkl: ✗ 不存在")

    return train_exists and test_exists

def main():
    print()
    print("=" * 70)
    print("Kinesis 项目配置检查")
    print("=" * 70)
    print()

    smpl_ok = check_smpl()
    kit_tar_ok = check_kit()

    print()
    print("=" * 70)
    print("配置状态")
    print("=" * 70)

    if smpl_ok and kit_tar_ok:
        kit_dir_exists = os.path.exists(r"D:\兼职\18\Kinesis\data\KIT_Data\KIT")
        if not kit_dir_exists:
            print()
            print("建议: 需要解压 KIT 数据集")
            print("运行: python extract_kit.py")
            converted = False
        else:
            converted = check_converted()
    elif smpl_ok and kit_dir_exists:
        converted = check_converted()
    else:
        converted = False

    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)

    if smpl_ok and converted:
        print("  SMPL 模型: ✓ 就绪")
        print("  KIT 数据: ✓ 就绪")
        print()
        print("=" * 70)
        print("✅ 所有文件已就绪！")
        print("=" * 70)
        print()
        print("可以运行 Kinesis 测试了！")
        print()
        print("测试命令:")
        print("  1. 目标到达:")
        print("     python src/run.py exp_name=kinesis-target-goal-reach")
        print("     run=eval_run learning=pointgoal epoch=-1 run.headless=True")
        print()
        print("  2. 运动模仿:")
        print("     python src/run.py exp_name=kinesis-moe-imitation")
        print("     epoch=-1 run=eval_run run.headless=True")
        print("     run.motion_file=data/kit_test_motion_dict.pkl")
        print("     run.initial_pose_file=data/initial_pose/initial_pose_test.pkl")
        print("     env.termination_distance=0.5")
    else:
        if not smpl_ok:
            print("  SMPL 模型: ✗ 缺失")
        if not (kit_tar_ok or os.path.exists(r"D:\兼职\18\Kinesis\data\KIT_Data\KIT")):
            print("  KIT 数据: ✗ 缺失")
        if smpl_ok and (kit_tar_ok or os.path.exists(r"D:\兼职\18\Kinesis\data\KIT_Data\KIT")) and not converted:
            print("  转换数据: ⚠️  需要转换")
            print()
            print("运行转换:")
            print("  python src/utils/convert_kit.py --path data/KIT_Data/KIT")

    print("=" * 70)

if __name__ == "__main__":
    main()
