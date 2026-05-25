#!/usr/bin/env python3
import os
import tarfile
import subprocess

# 硬编码路径
BASE_DIR = r"D:\兼职\18\Kinesis"
KIT_TAR = os.path.join(BASE_DIR, "data", "KIT_Data", "KIT.tar.bz2")
KIT_DIR = os.path.join(BASE_DIR, "data", "KIT_Data", "KIT")

print("=" * 70)
print("KINESIS - 完整自动配置")
print("=" * 70)
print()

# 步骤1：检查文件
print("[步骤1/3] 检查必需文件")
print("-" * 70)

smpl_ok = os.path.exists(os.path.join(BASE_DIR, "data", "smpl", "SMPL_NEUTRAL.pkl"))
kit_tar_ok = os.path.exists(KIT_TAR)

if smpl_ok:
    print("[OK] SMPL模型存在")
else:
    print("[MISSING] SMPL模型不存在")

if kit_tar_ok:
    print(f"[OK] KIT压缩包存在 ({os.path.getsize(KIT_TAR)/(1024*1024):.2f} MB)")
else:
    print("[MISSING] KIT压缩包不存在")

print()

# 步骤2：解压KIT
print("[步骤2/3] 解压KIT数据集")
print("-" * 70)

kit_dir_exists = os.path.exists(KIT_DIR)
npz_exists = False

if kit_dir_exists:
    npz_count = 0
    for root, dirs, files in os.walk(KIT_DIR):
        for f in files:
            if f.endswith('.npz'):
                npz_count += 1

    if npz_count > 0:
        print(f"[INFO] KIT已解压 ({npz_count} 个NPZ文件)")
        npz_exists = True
    else:
        print("[INFO] KIT目录为空")
else:
    print("[INFO] KIT目录不存在")

print()

if not kit_dir_exists:
    if kit_tar_ok:
        print("[ACTION] 开始解压KIT压缩包...")
        print()

        try:
            tar = tarfile.open(KIT_TAR, 'r:bz2')
            members = tar.getmembers()
            total = len(members)

            print(f"  文件数: {total}")

            for i, member in enumerate(members, 1):
                tar.extract(member, KIT_DIR)
                if i % 100 == 0 or i == total:
                    print(f"  进度: [{i}/{total}] {i/total*100:.1f}%")

            tar.close()

            print("[SUCCESS] 解压完成！")
            print()

            npz_count = 0
            for root, dirs, files in os.walk(KIT_DIR):
                for f in files:
                    if f.endswith('.npz'):
                        npz_count += 1

            print(f"  NPZ文件: {npz_count}")
            print()
        except Exception as e:
            print(f"[ERROR] 解压失败: {e}")
    else:
        print("[INFO] 跳过解压（压缩包不存在）")

print()

# 步骤3：转换KIT
print("[步骤3/3] 转换KIT数据集")
print("-" * 70)

train_pkl = os.path.join(BASE_DIR, "data", "kit_train_motion_dict.pkl")
test_pkl = os.path.join(BASE_DIR, "data", "kit_test_motion_dict.pkl")

train_exists = os.path.exists(train_pkl)
test_exists = os.path.exists(test_pkl)

if train_exists and test_exists:
    print("[INFO] KIT数据已转换，跳过...")
    converted = True
elif npz_exists:
    print("[ACTION] 开始转换KIT数据集（这可能需要10-15分钟）...")
    print()

    try:
        result = subprocess.run(
            [os.path.join(BASE_DIR, "Kinesis", "src", "utils", "convert_kit.py"),
            "--path", KIT_DIR],
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
            timeout=1200  # 20分钟超时
        )

        if result.returncode == 0:
            print("[SUCCESS] 转换完成！")
            if result.stdout:
                print(f"  输出: {result.stdout[-500:]}")
            train_exists = os.path.exists(train_pkl)
            test_exists = os.path.exists(test_pkl)
            converted = train_exists and test_exists
        else:
            print(f"[ERROR] 转换失败")
            if result.stderr:
                print(f"  错误: {result.stderr[-500:]}")
            converted = False
    except subprocess.TimeoutExpired:
        print("[ERROR] 转换超时（20分钟）")
        converted = False
    except Exception as e:
        print(f"[ERROR] 转换异常: {e}")
        converted = False
else:
    print("[INFO] 需要解压KIT才能转换")
    converted = False

print()

# 总结
print("=" * 70)
print("配置总结")
print("=" * 70)
print()

print("文件状态:")
print(f"  SMPL模型: {'[OK]' if smpl_ok else '[MISSING]'}")
print(f"  KIT压缩包: {'[OK]' if kit_tar_ok else '[MISSING]'}")
print(f"  KIT解压: {'[OK]' if npz_exists else '[PENDING]'}")
print(f"  KIT转换: {'[OK]' if converted else '[PENDING]'}")
print()

all_ready = smpl_ok and converted

if all_ready:
    print()
    print("=" * 70)
    print("[SUCCESS] 所有文件已就绪！")
    print("=" * 70)
    print()
    print("可以运行Kinesis测试了！")
    print()
    print("测试命令（复制到命令行）：")
    print()
    print("[测试1] 目标到达（最简单）:")
    print(f"  cd {BASE_DIR}\\Kinesis")
    print("  python src\\run.py exp_name=kinesis-target-goal-reach")
    print("  run=eval_run learning=pointgoal")
    print("  epoch=-1 run.headless=True")
    print()
    print("[测试2] 运动模仿:")
    print(f"  cd {BASE_DIR}\\Kinesis")
    print("  python src\\run.py exp_name=kinesis-moe-imitation")
    print("  epoch=-1 run=eval_run run.headless=True")
    print("  run.motion_file=data\\kit_test_motion_dict.pkl")
    print("  run.initial_pose_file=data\\initial_pose\\initial_pose_test.pkl")
    print("  env.termination_distance=0.5")
    print()
else:
    print()
    print("=" * 70)
    print("[INCOMPLETE] 还有文件需要处理")
    print("=" * 70)
    print()
    print("缺失的步骤:")
    if not kit_tar_ok:
        print("  - KIT压缩包未找到")
    if not npz_exists:
        print("  - KIT数据未解压")
    if not converted:
        print("  - KIT数据未转换")

    print()
    print("要继续配置吗？ (y/n): ", end="")

choice = input().strip().lower()

if choice == 'y' or choice == 'yes':
    print()
    print("重新运行此脚本以继续配置...")
    print("按Enter键退出...")
    input()
else:
    print()
    print("配置完成。")
    print("手动运行缺失的步骤后，再次运行此脚本。")
    print("按Enter键退出...")
    input()
