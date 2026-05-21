#!/usr/bin/env python3
import os
import tarfile

# 使用绝对路径
base_dir = r"D:\兼职\18\Kinesis"
kit_tar = os.path.join(base_dir, "data", "KIT_Data", "KIT.tar.bz2")
kit_dir = os.path.join(base_dir, "data", "KIT_Data", "KIT")

print("=" * 70)
print("Extracting KIT dataset")
print("=" * 70)
print()

if not os.path.exists(kit_tar):
    print(f"ERROR: {kit_tar} does not exist")
    input("Press Enter to exit...")
    exit(1)

print(f"Tar file: {kit_tar}")
print(f"Extract to: {kit_dir}")
print(f"Size: {os.path.getsize(kit_tar)/(1024*1024):.2f} MB")
print()

try:
    tar = tarfile.open(kit_tar, 'r:bz2')
    members = tar.getmembers()
    total = len(members)

    print(f"Total files: {total}")
    print("Extracting...")
    print()

    for i, member in enumerate(members, 1):
        tar.extract(member, kit_dir)
        if i % 100 == 0 or i == total:
            percent = (i / total) * 100
            print(f"Progress: [{i}/{total}] {percent:.1f}%")

    tar.close()
    print()
    print("[OK] Extraction complete!")
    print()

    npz_count = 0
    for root, dirs, files in os.walk(kit_dir):
        for f in files:
            if f.endswith('.npz'):
                npz_count += 1

    print(f"NPZ files extracted: {npz_count}")
    print()
    print("=" * 70)
    print("Next step:")
    print("Run conversion script:")
    print(f"  python {os.path.join(base_dir, 'Kinesis', 'src', 'utils', 'convert_kit.py')} --path {kit_dir}")
    print("=" * 70)
    input("Press Enter to exit...")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
