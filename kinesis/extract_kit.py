#!/usr/bin/env python3
import os
import tarfile

def main():
    # Base directory - use raw string for Windows path
    base_dir = r"D:\兼职\18\Kinesis"

    kit_tar_path = os.path.join(base_dir, "data", "KIT_Data", "KIT.tar.bz2")
    kit_dir = os.path.join(base_dir, "data", "KIT_Data", "KIT")

    print("=" * 70)
    print("Extracting KIT dataset")
    print("=" * 70)
    print()

    if not os.path.exists(kit_tar_path):
        print(f"ERROR: {kit_tar_path} does not exist")
        return False

    print(f"Tar file: {kit_tar_path}")
    print(f"Extract to: {kit_dir}")
    print(f"Size: {os.path.getsize(kit_tar_path)/(1024*1024):.2f} MB")
    print()

    # Extract
    try:
        tar = tarfile.open(kit_tar_path, 'r:bz2')
        members = tar.getmembers()
        total = len(members)

        print(f"Total files in archive: {total}")
        print("Extracting...")
        print()

        for i, member in enumerate(members, 1):
            tar.extract(member, kit_dir)
            if i % 100 == 0 or i == total:
                percent = (i / total) * 100
                print(f"  Progress: [{i}/{total}] {percent:.1f}%")

        tar.close()
        print()
        print("Extraction complete!")
        print()

        # Count NPZ files
        npz_count = 0
        for root, dirs, files in os.walk(kit_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_count += 1

        print(f"NPZ files extracted: {npz_count}")
        print()

        return True

    except Exception as e:
        print(f"ERROR: Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
