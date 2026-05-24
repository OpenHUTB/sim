#!/usr/bin/env python3
"""检查SMPL文件格式和内容"""
import os
import sys
import numpy as np

def check_file(filepath):
    """检查文件是否存在并显示内容"""
    if not os.path.exists(filepath):
        print(f"✗ 文件不存在: {filepath}")
        return False

    print(f"✓ 文件存在: {filepath}")
    print(f"  大小: {os.path.getsize(filepath)/1024/1024:.2f} MB")

    # 检查扩展名
    ext = os.path.splitext(filepath)[1].lower()
    print(f"  格式: {ext}")

    # 尝试加载
    try:
        if ext == '.npz':
            data = np.load(filepath, allow_pickle=True)
            print(f"✓ NPZ文件包含的键:")
            for key in data.files:
                print(f"    - {key}")
            print(f"  总共 {len(data.files)} 个键")
        elif ext == '.pkl':
            import joblib
            data = joblib.load(filepath)
            print(f"✓ PKL文件类型: {type(data)}")
            if isinstance(data, dict):
                print(f"  包含 {len(data)} 个条目")
                print(f"  键:")
                for key in list(data.keys())[:10]:
                    print(f"    - {key}")
                if len(data) > 10:
                    print(f"    ... (还有 {len(data)-10} 个)")
        else:
            print(f"✗ 不支持的格式: {ext}")
            return False

        return True

    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        return False

def main():
    base_dir = r"D:\兼职\18\Kinesis"
    smpl_dir = os.path.join(base_dir, "data", "smpl")

    print("=" * 70)
    print("SMPL 文件检查")
    print("=" * 70)
    print()

    # 检查 NPZ 文件
    npz_file = os.path.join(smpl_dir, "SMPL_NEUTRAL.npz")
    print("[1] 检查 SMPL_NEUTRAL.npz")
    npz_ok = check_file(npz_file)
    print()

    # 检查 PKL 文件
    pkl_file = os.path.join(smpl_dir, "SMPL_NEUTRAL.pkl")
    print("[2] 检查 SMPL_NEUTRAL.pkl")
    pkl_ok = check_file(pkl_file)
    print()

    # 结论
    print("=" * 70)
    if npz_ok:
        print("✓ 找到了 SMPL_NEUTRAL.npz 文件")
        print("  这可能是SMPL模型的数据文件。")
        print()
        print("  建议：")
        print("  1. 如果 smplx 库支持 NPZ 格式，可以尝试直接使用")
        print("  2. 或者将 NPZ 转换为 PKL 格式")
        print()
        if pkl_ok:
            print("✓ 同时也有 SMPL_NEUTRAL.pkl 文件，可以使用 PKL 版本")
        else:
            print("  没有找到 PKL 版本，需要转换")
    elif pkl_ok:
        print("✓ 找到了 SMPL_NEUTRAL.pkl 文件")
        print("  这是标准的SMPL模型文件，可以直接使用。")
    else:
        print("✗ 没有找到任何 SMPL 模型文件")
        print("  请从 https://smpl.is.tue.mpg.de/ 下载")

    print("=" * 70)

if __name__ == "__main__":
    main()
