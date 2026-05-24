#!/usr/bin/env python3
import os
import sys
import tarfile
import subprocess

def main():
    print("=" * 70)
    print("KINESIS 一键自动配置")
    print("=" * 70)
    print()

    # 设置基础目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    print(f"工作目录: {base_dir}")
    print()

    # 步骤1：检查SMPL
    print("[步骤1/3] 检查SMPL模型")
    print("-" * 70)

    smpl_path = os.path.join("data", "smpl", "SMPL_NEUTRAL.pkl")
    if os.path.exists(smpl_path):
        size_mb = os.path.getsize(smpl_path) / (1024*1024)
        print(f"✓ SMPL_NEUTRAL.pkl 存在")
        print(f"  大小: {size_mb:.2f} MB")
        smpl_ok = True
    else:
        print(f"✗ SMPL_NEUTRAL.pkl 不存在")
        print(f"  期望位置: {smpl_path}")
        smpl_ok = False

    print()

    # 步骤2：解压KIT
    print("[步骤2/3] 解压KIT数据集")
    print("-" * 70)

    kit_tar = os.path.join("data", "KIT_Data", "KIT.tar.bz2")
    kit_dir = os.path.join("data", "KIT_Data", "KIT")

    if os.path.exists(kit_tar):
        size_mb = os.path.getsize(kit_tar) / (1024*1024)
        print(f"✓ KIT.tar.bz2 存在")
        print(f"  大小: {size_mb:.2f} MB")

        if os.path.exists(kit_dir):
            print(f"  KIT目录已存在，检查NPZ文件...")
            npz_count = 0
            for root, dirs, files in os.walk(kit_dir):
                for f in files:
                    if f.endswith('.npz'):
                        npz_count += 1
            print(f"  NPZ文件数: {npz_count}")
            kit_ready = npz_count > 0
        else:
            print(f"  开始解压...")
            try:
                tar = tarfile.open(kit_tar, 'r:bz2')
                members = tar.getmembers()
                total = len(members)

                print(f"  压缩包包含 {total} 个文件")

                for i, member in enumerate(members, 1):
                    tar.extract(member, kit_dir)
                    if i % 100 == 0 or i == total:
                        percent = (i / total) * 100
                        print(f"  进度: [{i}/{total}] {percent:.1f}%")

                tar.close()
                print(f"  ✓ 解压完成！")

                npz_count = 0
                for root, dirs, files in os.walk(kit_dir):
                    for f in files:
                        if f.endswith('.npz'):
                            npz_count += 1

                print(f"  NPZ文件数: {npz_count}")
                kit_ready = npz_count > 0

            except Exception as e:
                print(f"  ✗ 解压失败: {e}")
                kit_ready = False

        print()
    else:
        print(f"✗ KIT.tar.bz2 不存在")
        print(f"  期望位置: {kit_tar}")
        kit_ready = False

    # 步骤3：转换KIT
    print("[步骤3/3] 转换KIT数据集")
    print("-" * 70)

    train_pkl = os.path.join("data", "kit_train_motion_dict.pkl")
    test_pkl = os.path.join("data", "kit_test_motion_dict.pkl")

    train_exists = os.path.exists(train_pkl)
    test_exists = os.path.exists(test_pkl)

    if train_exists and test_exists:
        print(f"✓ kit_train_motion_dict.pkl 存在")
        print(f"✓ kit_test_motion_dict.pkl 存在")
        print(f"  KIT数据已转换，跳过...")
        print()
        converted = True
    elif kit_ready and kit_dir:
        print(f"  开始转换KIT数据...")
        print(f"  这可能需要10-15分钟...")
        print()

        try:
            result = subprocess.run(
                [sys.executable, "src/utils/convert_kit.py", "--path", kit_dir],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"✓ 转换完成！")
                if result.stdout:
                    print(f"  输出: {result.stdout}")
                converted = True
            else:
                print(f"✗ 转换失败")
                if result.stderr:
                    print(f"  错误: {result.stderr}")
                converted = False

        except Exception as e:
            print(f"✗ 转换异常: {e}")
            converted = False
    else:
        print(f"  KIT数据未就绪，跳过转换...")
        converted = False

    print()

    # 总结
    print("=" * 70)
    print("配置总结")
    print("=" * 70)
    print()

    all_ready = smpl_ok and converted

    if all_ready:
        print("✅✅✅ 所有文件已就绪！")
        print()
        print("=" * 70)
        print("可以运行Kinesis测试了！")
        print("=" * 70)
        print()
        print("测试命令（复制以下命令到命令行）：")
        print()
        print("[测试1] 目标到达（推荐，最简单）：")
        print("  python src/run.py exp_name=kinesis-target-goal-reach")
        print("    run=eval_run learning=pointgoal")
        print("    epoch=-1 run.headless=True")
        print()
        print("[测试2] 运动模仿（需要转换后的数据）：")
        print("  python src/run.py exp_name=kinesis-moe-imitation")
        print("    epoch=-1 run=eval_run run.headless=True")
        print("    run.motion_file=data/kit_test_motion_dict.pkl")
        print("    run.initial_pose_file=data/initial_pose/initial_pose_test.pkl")
        print("    env.termination_distance=0.5")
        print()
        print("=" * 70)
    else:
        print("⚠️ 还有文件需要处理")
        print()
        print(f"  SMPL模型: {'✓ 就绪' if smpl_ok else '✗ 缺失'}")
        print(f"  KIT解压目录: {'✓ 就绪' if kit_ready else '✗ 缺失'}")
        print(f"  KIT转换数据: {'✓ 就绪' if converted else '✗ 需要转换'}")
        print()
        print("=" * 70)

    print()
    input("按Enter键退出...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        input("按Enter键退出...")
