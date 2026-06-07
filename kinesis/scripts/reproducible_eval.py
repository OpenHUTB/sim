"""Run the thesis-aligned reproducible Kinesis evaluation pipeline.

The pipeline uses real data only:
1. Validate converted KIT motions, initial poses, SMPL, XML, and checkpoint.
2. Run Kinesis imitation evaluation with fixed deterministic settings.
3. Generate precision plots from evaluation_metrics.json.
4. Generate lower-limb coordination plots from the converted motion file.
5. Generate baseline-vs-optimized reward comparison plots from real T2M motions.

Usage:
    python scripts/reproducible_eval.py
    python scripts/reproducible_eval.py --skip-eval
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{purpose} 缺失: {path}")


def missing_reason(path: Path, purpose: str) -> str | None:
    if path.exists():
        return None
    return f"{purpose} 缺失: {path}"


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="kinesis-moe-imitation")
    parser.add_argument("--epoch", default="-1")
    parser.add_argument("--motion-file", default="data/kit_test_motion_dict.pkl")
    parser.add_argument("--initial-pose-file", default="data/initial_pose/initial_pose_test.pkl")
    parser.add_argument("--num-motions", default="500")
    parser.add_argument("--skip-eval", action="store_true", help="Only regenerate plots from existing real metrics.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing inputs instead of skipping unavailable steps.",
    )
    args = parser.parse_args()

    motion_file = PROJECT_DIR / args.motion_file
    fallback_motion_files = sorted((PROJECT_DIR / "data" / "t2m").glob("*.pkl"))
    coordination_motion_file = motion_file if motion_file.exists() else (fallback_motion_files[0] if fallback_motion_files else motion_file)
    initial_pose_file = PROJECT_DIR / args.initial_pose_file
    smpl_file = PROJECT_DIR / "data" / "smpl" / "SMPL_NEUTRAL.pkl"
    xml_file = PROJECT_DIR / "data" / "xml" / "myolegs.xml"
    checkpoint = PROJECT_DIR / "data" / "trained_models" / args.exp_name / "model.pth"
    metrics_file = PROJECT_DIR / "output" / "precision" / "evaluation_metrics.json"

    missing_eval_inputs = [
        reason
        for reason in [
            missing_reason(motion_file, "真实 KIT 测试动作"),
            missing_reason(initial_pose_file, "真实测试初始姿态"),
            missing_reason(smpl_file, "SMPL 中性模型"),
            missing_reason(xml_file, "MyoLegs MuJoCo XML"),
            None if args.skip_eval else missing_reason(checkpoint, "预训练/训练后的策略 checkpoint"),
        ]
        if reason is not None
    ]

    missing_coord_inputs = [
        reason
        for reason in [
            missing_reason(coordination_motion_file, "可用下肢动作数据"),
        ]
        if reason is not None
    ]

    if args.strict and missing_eval_inputs:
        raise FileNotFoundError("\n".join(missing_eval_inputs))

    if not args.skip_eval and not missing_eval_inputs:
        run_command(
            [
                sys.executable,
                "src/run.py",
                f"exp_name={args.exp_name}",
                f"epoch={args.epoch}",
                "run=repro_eval",
                "env=env_im_eval",
                f"run.motion_file={args.motion_file}",
                f"run.initial_pose_file={args.initial_pose_file}",
                f"run.num_motions={args.num_motions}",
                "run.headless=True",
                "no_log=True",
            ]
        )
    elif args.skip_eval:
        print("\n[SKIP] 已指定 --skip-eval，跳过 Kinesis imitation eval。")
    else:
        print("\n[SKIP] 跳过 Kinesis imitation eval，原因：")
        for reason in missing_eval_inputs:
            print(f"  - {reason}")

    if metrics_file.exists():
        run_command([sys.executable, "scripts/plot_precision.py"])
    elif args.strict:
        require_file(metrics_file, "真实评估指标 JSON")
    else:
        print("\n[SKIP] 跳过精度图表，原因：")
        print(f"  - 真实评估指标 JSON 缺失: {metrics_file}")

    if not missing_coord_inputs:
        if coordination_motion_file != motion_file:
            print(f"\n[INFO] 未找到 KIT 转换数据，使用已有 T2M 动作做下肢协同分析: {coordination_motion_file}")
        run_command(
            [
                sys.executable,
                "scripts/analyze_coordination.py",
                "--motion-file",
                str(coordination_motion_file.relative_to(PROJECT_DIR)),
            ]
        )
    elif args.strict:
        raise FileNotFoundError("\n".join(missing_coord_inputs))
    else:
        print("\n[SKIP] 跳过真实下肢协同图表，原因：")
        for reason in missing_coord_inputs:
            print(f"  - {reason}")

    t2m_dir = PROJECT_DIR / "data" / "t2m"
    if any(t2m_dir.glob("*.pkl")):
        run_command([sys.executable, "scripts/compare_reward_optimization.py"])
    elif args.strict:
        raise FileNotFoundError(f"真实 T2M 动作目录为空: {t2m_dir}")
    else:
        print("\n[SKIP] 跳过奖励优化对比实验，原因：")
        print(f"  - 真实 T2M 动作目录为空: {t2m_dir}")

    print("\n可复现实验流水线完成。可用步骤已执行，缺失输入对应步骤已跳过。")


if __name__ == "__main__":
    main()
