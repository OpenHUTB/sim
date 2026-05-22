#!/usr/bin/env python3
import os
import sys
import tarfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_file(path, name):
    """检查文件是否存在"""
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"[OK] {name}: {size:.2f} MB")
        return True
    else:
        print(f"[MISSING] {name}: Not found at {path}")
        return False

def extract_tar(tar_path, extract_to):
    """解压tar.bz2文件"""
    print(f"Extracting {tar_path} to {extract_to}...")
    try:
        tar = tarfile.open(tar_path, 'r:bz2')
        members = tar.getmembers()
        total = len(members)

        for i, member in enumerate(members, 1):
            tar.extract(member, extract_to)
            if i % 100 == 0 or i == total:
                print(f"  Progress: [{i}/{total}] ({i/total*100:.1f}%)")

        tar.close()

        # 统计NPZ文件
        npz_count = 0
        for root, dirs, files in os.walk(extract_to):
            for f in files:
                if f.endswith('.npz'):
                    npz_count += 1

        print(f"[OK] Extraction complete! ({npz_count} NPZ files)")
        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False

def main():
    print("=" * 70)
    print("KINESIS CONFIGURATION CHECK")
    print("=" * 70)
    print()

    # 检查文件
    smpl_path = os.path.join(BASE_DIR, "data", "smpl", "SMPL_NEUTRAL.pkl")
    kit_tar_path = os.path.join(BASE_DIR, "data", "KIT_Data", "KIT.tar.bz2")
    kit_dir = os.path.join(BASE_DIR, "data", "KIT_Data", "KIT")

    train_pkl = os.path.join(BASE_DIR, "data", "kit_train_motion_dict.pkl")
    test_pkl = os.path.join(BASE_DIR, "data", "kit_test_motion_dict.pkl")

    # 1. 检查SMPL
    print("[1/5] Checking SMPL model...")
    smpl_ok = check_file(smpl_path, "SMPL_NEUTRAL.pkl")
    print()

    # 2. 检查KIT压缩包
    print("[2/5] Checking KIT archive...")
    kit_tar_ok = check_file(kit_tar_path, "KIT.tar.bz2")
    print()

    # 3. 检查KIT解压目录
    print("[3/5] Checking KIT directory...")
    kit_dir_ok = os.path.exists(kit_dir)
    if kit_dir_ok:
        npz_count = 0
        for root, dirs, files in os.walk(kit_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_count += 1
        print(f"[OK] KIT directory exists ({npz_count} NPZ files)")
    else:
        print(f"[NOT FOUND] KIT directory: {kit_dir}")
    print()

    # 4. 检查转换后的数据
    print("[4/5] Checking converted KIT data...")
    train_ok = check_file(train_pkl, "kit_train_motion_dict.pkl")
    test_ok = check_file(test_pkl, "kit_test_motion_dict.pkl")
    print()

    # 总结
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if smpl_ok and (kit_dir_ok or kit_tar_ok):
        if train_ok and test_ok:
            print("[SUCCESS] All files are ready!")
            print()
            print("You can now run Kinesis tests:")
            print()
            print("Test 1 - Target Reaching:")
            print("  cd " + os.path.dirname(os.path.abspath(__file__)))
            print("  python src\\run.py exp_name=kinesis-target-goal-reach run=eval_run learning=pointgoal epoch=-1 run.headless=True")
            print()
            print("Test 2 - Motion Imitation:")
            print("  cd " + os.path.dirname(os.path.abspath(__file__)))
            print("  python src\\run.py exp_name=kinesis-moe-imitation epoch=-1 run=eval_run run.headless=True")
            print("  run.motion_file=data\\kit_test_motion_dict.pkl")
            print("  run.initial_pose_file=data\\initial_pose\\initial_pose_test.pkl")
            print("  env.termination_distance=0.5")
        else:
            print("[INFO] Need to convert KIT data")
            print()
            print("Run conversion:")
            print("  cd " + os.path.dirname(os.path.abspath(__file__)))
            print("  python src\\utils\\convert_kit.py --path data\\KIT_Data\\KIT")
    else:
        print("[ERROR] Missing required files:")
        if not smpl_ok:
            print("  - SMPL_NEUTRAL.pkl")
        if not (kit_tar_ok or kit_dir_ok):
            print("  - KIT data (KIT.tar.bz2 or KIT directory)")

    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="Extract KIT data")
    args = parser.parse_args()

    if args.extract:
        kit_tar_path = os.path.join(BASE_DIR, "data", "KIT_Data", "KIT.tar.bz2")
        kit_dir = os.path.join(BASE_DIR, "data", "KIT_Data", "KIT")
        if os.path.exists(kit_tar_path):
            extract_tar(kit_tar_path, kit_dir)

    main()
