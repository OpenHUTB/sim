#!/usr/bin/env python3
import os
import sys
import tarfile
import subprocess

def main():
    print("=" * 70)
    print("KINESIS Auto Configuration")
    print("=" * 70)
    print()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    print(f"Working dir: {base_dir}")
    print()

    # Step 1: Check SMPL
    print("[1/3] Checking SMPL model")
    print("-" * 70)

    smpl_path = os.path.join("data", "smpl", "SMPL_NEUTRAL.pkl")
    if os.path.exists(smpl_path):
        size_mb = os.path.getsize(smpl_path) / (1024*1024)
        print(f"[OK] SMPL_NEUTRAL.pkl exists")
        print(f"     Size: {size_mb:.2f} MB")
        smpl_ok = True
    else:
        print(f"[MISSING] SMPL_NEUTRAL.pkl not found")
        print(f"     Expected: {smpl_path}")
        smpl_ok = False

    print()

    # Step 2: Extract KIT
    print("[2/3] Extracting KIT dataset")
    print("-" * 70)

    kit_tar = os.path.join("data", "KIT_Data", "KIT.tar.bz2")
    kit_dir = os.path.join("data", "KIT_Data", "KIT")

    if not os.path.exists(kit_tar):
        print(f"[MISSING] KIT.tar.bz2 not found")
        print(f"     Expected: {kit_tar}")
        kit_ready = False
    else:
        print(f"[OK] KIT.tar.bz2 found")
        size_mb = os.path.getsize(kit_tar) / (1024*1024)
        print(f"     Size: {size_mb:.2f} MB")

        if os.path.exists(kit_dir):
            print(f"[INFO] KIT directory already exists")
            npz_count = 0
            for root, dirs, files in os.walk(kit_dir):
                for f in files:
                    if f.endswith('.npz'):
                        npz_count += 1
            print(f"     NPZ files: {npz_count}")
            kit_ready = npz_count > 0
        else:
            print(f"[INFO] Extracting KIT archive...")
            try:
                tar = tarfile.open(kit_tar, 'r:bz2')
                members = tar.getmembers()
                total = len(members)

                print(f"     Total files: {total}")
                print(f"     Extracting to: {kit_dir}")

                for i, member in enumerate(members, 1):
                    tar.extract(member, kit_dir)
                    if i % 100 == 0 or i == total:
                        percent = (i / total) * 100
                        print(f"     Progress: [{i}/{total}] {percent:.1f}%")

                tar.close()
                print(f"[OK] Extraction complete!")

                npz_count = 0
                for root, dirs, files in os.walk(kit_dir):
                    for f in files:
                        if f.endswith('.npz'):
                            npz_count += 1

                print(f"     NPZ files extracted: {npz_count}")
                kit_ready = npz_count > 0

            except Exception as e:
                print(f"[ERROR] Extraction failed: {e}")
                kit_ready = False

    print()

    # Step 3: Convert KIT
    print("[3/3] Converting KIT dataset")
    print("-" * 70)

    train_pkl = os.path.join("data", "kit_train_motion_dict.pkl")
    test_pkl = os.path.join("data", "kit_test_motion_dict.pkl")

    train_exists = os.path.exists(train_pkl)
    test_exists = os.path.exists(test_pkl)

    if train_exists and test_exists:
        print(f"[OK] kit_train_motion_dict.pkl exists")
        print(f"[OK] kit_test_motion_dict.pkl exists")
        print(f"[INFO] KIT data already converted, skipping...")
        converted = True
    elif kit_ready:
        print(f"[INFO] KIT data ready for conversion")
        print(f"[INFO] Starting conversion (this may take 10-15 minutes)...")

        try:
            result = subprocess.run(
                [sys.executable, "src/utils/convert_kit.py", "--path", kit_dir],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"[OK] Conversion complete!")
                if result.stdout:
                    print(f"     Output: {result.stdout[-500:]}")
                converted = True
            else:
                print(f"[ERROR] Conversion failed")
                if result.stderr:
                    print(f"     Error: {result.stderr[-500:]}")
                converted = False

        except Exception as e:
            print(f"[ERROR] Conversion exception: {e}")
            converted = False
    else:
        print(f"[WAITING] KIT data not ready")
        converted = False

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if smpl_ok and converted:
        print("[SUCCESS] All files ready!")
        print()
        print("You can now run Kinesis tests:")
        print()
        print("Test 1 - Target Reaching (simplest):")
        print("  python src/run.py exp_name=kinesis-target-goal-reach \\")
        print("    run=eval_run learning=pointgoal epoch=-1 run.headless=True")
        print()
        print("Test 2 - Motion Imitation:")
        print("  python src/run.py exp_name=kinesis-moe-imitation \\")
        print("    epoch=-1 run=eval_run run.headless=True \\")
        print("    run.motion_file=data/kit_test_motion_dict.pkl \\")
        print("    run.initial_pose_file=data/initial_pose/initial_pose_test.pkl \\")
        print("    env.termination_distance=0.5")
        print()
    else:
        print("[INCOMPLETE] Some files are missing:")
        print()
        if not smpl_ok:
            print("  - SMPL_NEUTRAL.pkl")
        if not kit_ready and not converted:
            print("  - KIT data (needs extraction and conversion)")
        print()
        print("Please complete missing steps.")

    print("=" * 70)
    print()

    input("Press Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
