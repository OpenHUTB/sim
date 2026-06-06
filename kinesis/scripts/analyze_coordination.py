"""Analyze lower-limb coordination from real converted Kinesis/KIT motion data.

This script does not generate synthetic gait, contact, muscle, or reward data.
It reads real converted motion data, normally data/kit_test_motion_dict.pkl, and
uses lower-body joints only because the reference MyoLegs controller has no
active upper-body control.

Usage:
    python scripts/analyze_coordination.py
    python scripts/analyze_coordination.py --motion-file data/kit_test_motion_dict.pkl
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from src.utils.smpl_skeleton.smpl_joint_names import SMPL_MUJOCO_NAMES


DEFAULT_MOTION_FILE = PROJECT_DIR / "data" / "kit_test_motion_dict.pkl"
DEFAULT_SMPL_DIR = PROJECT_DIR / "data" / "smpl"
DEFAULT_XML = PROJECT_DIR / "data" / "xml" / "smpl_humanoid.xml"
OUTPUT_DIR = PROJECT_DIR / "output" / "coordination"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


JOINT_INDEX = {name: i for i, name in enumerate(SMPL_MUJOCO_NAMES)}
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
REQUIRED_JOINTS = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
    "L_Toe",
    "R_Toe",
]
UPPER_BODY_JOINTS = [
    "Pelvis",
    "Chest",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]


def require_real_motion_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"真实动作数据不存在: {path}\n"
            "请先下载 KIT/AMASS 数据并运行转换，例如:\n"
            "python src/utils/convert_kit.py --path data/KIT_Data/KIT\n"
            "生成 data/kit_test_motion_dict.pkl 后再运行本脚本。"
        )


def normalize_pose_aa(pose_aa: np.ndarray) -> np.ndarray:
    pose_aa = np.asarray(pose_aa, dtype=np.float32)
    if pose_aa.ndim == 2 and pose_aa.shape[1] == 156:
        pose_aa = np.concatenate(
            [pose_aa[:, :66], np.zeros((pose_aa.shape[0], 6), dtype=np.float32)],
            axis=1,
        )
    if pose_aa.ndim == 2 and pose_aa.shape[1] == 72:
        pose_aa = pose_aa.reshape(-1, 24, 3)
    if pose_aa.ndim != 3 or pose_aa.shape[1:] != (24, 3):
        raise ValueError(f"不支持的 pose_aa 形状: {pose_aa.shape}")
    return pose_aa


def positions_from_fk(motion: dict[str, Any], smpl_dir: Path) -> np.ndarray:
    neutral_model = smpl_dir / "SMPL_NEUTRAL.pkl"
    if not neutral_model.exists():
        raise FileNotFoundError(
            f"需要 SMPL 模型做真实前向运动学，但文件不存在: {neutral_model}\n"
            "如果 motion pkl 内没有 global_translation，就必须放置 SMPL_NEUTRAL.pkl。"
        )

    import torch
    from src.KinesisCore.forward_kinematics import ForwardKinematics

    pose_aa = normalize_pose_aa(motion["pose_aa"])
    trans = np.asarray(motion.get("trans", motion.get("trans_orig")), dtype=np.float32)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError("motion 中缺少 trans/trans_orig，无法计算真实关节位置。")

    fk = ForwardKinematics(str(smpl_dir))
    result = fk.fk_batch(
        torch.from_numpy(pose_aa[None]).float(),
        torch.from_numpy(trans[None]).float(),
    )
    return result.global_translation[0].detach().cpu().numpy()


def parse_xml_offsets(xml_path: Path) -> np.ndarray:
    if not xml_path.exists():
        raise FileNotFoundError(f"缺少骨架 XML，无法在没有 SMPL 的情况下计算关节位置: {xml_path}")

    root = ET.parse(xml_path).getroot()
    offsets: dict[str, np.ndarray] = {}

    def visit(element: ET.Element) -> None:
        if element.tag == "body" and "name" in element.attrib:
            name = element.attrib["name"]
            pos = element.attrib.get("pos", "0 0 0")
            offsets[name] = np.array([float(v) for v in pos.split()], dtype=np.float32)
        for child in element:
            visit(child)

    visit(root)
    missing = [name for name in SMPL_MUJOCO_NAMES if name not in offsets]
    if missing:
        raise ValueError(f"骨架 XML 缺少关节: {missing}")
    return np.stack([offsets[name] for name in SMPL_MUJOCO_NAMES], axis=0)


def quat_rotate_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def positions_from_global_quat(motion: dict[str, Any], xml_path: Path) -> np.ndarray:
    if "pose_quat_global" not in motion:
        raise ValueError("motion 中缺少 pose_quat_global，无法用 XML 骨架计算关节位置。")

    rotations = np.asarray(motion["pose_quat_global"], dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (24, 4):
        raise ValueError(f"不支持的 pose_quat_global 形状: {rotations.shape}")

    root = motion.get("root_trans_offset", motion.get("trans", motion.get("trans_orig")))
    root = np.asarray(root, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError("motion 中缺少 root_trans_offset/trans/trans_orig。")

    offsets = parse_xml_offsets(xml_path).astype(np.float64)
    positions = np.zeros((rotations.shape[0], 24, 3), dtype=np.float64)
    positions[:, 0, :] = root
    for joint_id in range(1, 24):
        parent_id = PARENTS[joint_id]
        parent_rot = rotations[:, parent_id, :]
        local_offset = np.broadcast_to(offsets[joint_id], (rotations.shape[0], 3))
        positions[:, joint_id, :] = positions[:, parent_id, :] + quat_rotate_xyzw(parent_rot, local_offset)
    return positions.astype(np.float32)


def get_joint_positions(motion: dict[str, Any], smpl_dir: Path) -> np.ndarray:
    for key in ("global_translation", "global_body_pos", "body_pos"):
        if key in motion:
            values = np.asarray(motion[key], dtype=np.float32)
            if values.ndim == 3 and values.shape[1] >= 24 and values.shape[2] == 3:
                return values[:, :24, :]
    if "pose_quat_global" in motion:
        return positions_from_global_quat(motion, DEFAULT_XML)
    if "pose_aa" not in motion:
        raise ValueError("motion 中既没有 global_translation，也没有 pose_aa，无法做真实下肢协同分析。")
    return positions_from_fk(motion, smpl_dir)


def centered_axis_signal(positions: np.ndarray, joint_name: str, axis: int) -> np.ndarray:
    pelvis = positions[:, JOINT_INDEX["Pelvis"], :]
    joint = positions[:, JOINT_INDEX[joint_name], :]
    return joint[:, axis] - pelvis[:, axis]


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    denom = np.where(denom < 1e-8, np.nan, denom)
    cosine = np.sum(ba * bc, axis=-1) / denom
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def phase_lag_frames(left: np.ndarray, right: np.ndarray) -> int:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if len(left) < 3:
        return 0
    left = left - np.mean(left)
    right = right - np.mean(right)
    corr = np.correlate(left, right, mode="full")
    return int(np.argmax(corr) - (len(right) - 1))


def analyze_motion(name: str, positions: np.ndarray) -> dict[str, Any]:
    for joint in REQUIRED_JOINTS:
        if joint not in JOINT_INDEX or JOINT_INDEX[joint] >= positions.shape[1]:
            raise ValueError(f"动作 {name} 缺少下肢关节 {joint}。")

    left_toe_x = centered_axis_signal(positions, "L_Toe", axis=0)
    right_toe_x = centered_axis_signal(positions, "R_Toe", axis=0)
    left_ankle_z = centered_axis_signal(positions, "L_Ankle", axis=2)
    right_ankle_z = centered_axis_signal(positions, "R_Ankle", axis=2)

    left_knee_angle = joint_angle_deg(
        positions[:, JOINT_INDEX["L_Hip"], :],
        positions[:, JOINT_INDEX["L_Knee"], :],
        positions[:, JOINT_INDEX["L_Ankle"], :],
    )
    right_knee_angle = joint_angle_deg(
        positions[:, JOINT_INDEX["R_Hip"], :],
        positions[:, JOINT_INDEX["R_Knee"], :],
        positions[:, JOINT_INDEX["R_Ankle"], :],
    )

    left_stride_amp = float(np.nanpercentile(left_toe_x, 95) - np.nanpercentile(left_toe_x, 5))
    right_stride_amp = float(np.nanpercentile(right_toe_x, 95) - np.nanpercentile(right_toe_x, 5))
    stride_symmetry = 1.0 - abs(left_stride_amp - right_stride_amp) / max(left_stride_amp, right_stride_amp, 1e-8)

    return {
        "motion_id": name,
        "frames": int(positions.shape[0]),
        "foot_alternation_corr": safe_corr(left_toe_x, right_toe_x),
        "knee_angle_corr": safe_corr(left_knee_angle, right_knee_angle),
        "ankle_height_corr": safe_corr(left_ankle_z, right_ankle_z),
        "ankle_height_symmetry_error_m": float(np.nanmean(np.abs(left_ankle_z - right_ankle_z))),
        "stride_amplitude_symmetry": float(stride_symmetry),
        "toe_phase_lag_frames": phase_lag_frames(left_toe_x, right_toe_x),
        "signals": {
            "left_toe_x": left_toe_x,
            "right_toe_x": right_toe_x,
            "left_ankle_z": left_ankle_z,
            "right_ankle_z": right_ankle_z,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
        },
    }


def analyze_upper_body(name: str, positions: np.ndarray) -> dict[str, Any]:
    for joint in UPPER_BODY_JOINTS:
        if joint not in JOINT_INDEX or JOINT_INDEX[joint] >= positions.shape[1]:
            raise ValueError(f"动作 {name} 缺少上肢关节 {joint}。")

    left_wrist_x = centered_axis_signal(positions, "L_Wrist", axis=0)
    right_wrist_x = centered_axis_signal(positions, "R_Wrist", axis=0)
    left_hand_y = centered_axis_signal(positions, "L_Hand", axis=1)
    right_hand_y = centered_axis_signal(positions, "R_Hand", axis=1)
    left_shoulder_z = centered_axis_signal(positions, "L_Shoulder", axis=2)
    right_shoulder_z = centered_axis_signal(positions, "R_Shoulder", axis=2)

    left_elbow_angle = joint_angle_deg(
        positions[:, JOINT_INDEX["L_Shoulder"], :],
        positions[:, JOINT_INDEX["L_Elbow"], :],
        positions[:, JOINT_INDEX["L_Wrist"], :],
    )
    right_elbow_angle = joint_angle_deg(
        positions[:, JOINT_INDEX["R_Shoulder"], :],
        positions[:, JOINT_INDEX["R_Elbow"], :],
        positions[:, JOINT_INDEX["R_Wrist"], :],
    )

    left_arm_amp = float(np.nanpercentile(left_wrist_x, 95) - np.nanpercentile(left_wrist_x, 5))
    right_arm_amp = float(np.nanpercentile(right_wrist_x, 95) - np.nanpercentile(right_wrist_x, 5))
    arm_swing_symmetry = 1.0 - abs(left_arm_amp - right_arm_amp) / max(left_arm_amp, right_arm_amp, 1e-8)

    return {
        "motion_id": name,
        "frames": int(positions.shape[0]),
        "arm_swing_corr": safe_corr(left_wrist_x, right_wrist_x),
        "elbow_angle_corr": safe_corr(left_elbow_angle, right_elbow_angle),
        "hand_lateral_corr": safe_corr(left_hand_y, -right_hand_y),
        "shoulder_height_symmetry_error_m": float(np.nanmean(np.abs(left_shoulder_z - right_shoulder_z))),
        "arm_swing_symmetry": float(arm_swing_symmetry),
        "signals": {
            "left_wrist_x": left_wrist_x,
            "right_wrist_x": right_wrist_x,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_shoulder_z": left_shoulder_z,
            "right_shoulder_z": right_shoulder_z,
        },
    }


def plot_overview(metric: dict[str, Any]) -> None:
    signals = metric["signals"]
    frames = np.arange(len(signals["left_toe_x"]))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"真实下肢协同轨迹: {metric['motion_id']}", fontsize=15, fontweight="bold")

    axes[0].plot(frames, signals["left_toe_x"], label="左脚趾 X(相对骨盆)", color="#D03050")
    axes[0].plot(frames, signals["right_toe_x"], label="右脚趾 X(相对骨盆)", color="#2080F0")
    axes[0].set_title(f"左右脚交替相关: {metric['foot_alternation_corr']:.3f}")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(frames, signals["left_knee_angle"], label="左膝角度", color="#D03050")
    axes[1].plot(frames, signals["right_knee_angle"], label="右膝角度", color="#2080F0")
    axes[1].set_title(f"左右膝角度相关: {metric['knee_angle_corr']:.3f}")
    axes[1].set_ylabel("deg")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(frames, signals["left_ankle_z"], label="左踝高度(相对骨盆)", color="#D03050")
    axes[2].plot(frames, signals["right_ankle_z"], label="右踝高度(相对骨盆)", color="#2080F0")
    axes[2].set_title(f"踝高度对称误差: {metric['ankle_height_symmetry_error_m']:.4f} m")
    axes[2].set_xlabel("Frame")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lower_limb_coordination_overview.png", dpi=160)
    plt.close(fig)


def plot_upper_overview(metric: dict[str, Any]) -> None:
    signals = metric["signals"]
    frames = np.arange(len(signals["left_wrist_x"]))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"上半身运动学协同轨迹: {metric['motion_id']}", fontsize=15, fontweight="bold")

    axes[0].plot(frames, signals["left_wrist_x"], label="左腕 X(相对骨盆)", color="#D03050")
    axes[0].plot(frames, signals["right_wrist_x"], label="右腕 X(相对骨盆)", color="#2080F0")
    axes[0].set_title(f"左右摆臂相关: {metric['arm_swing_corr']:.3f}")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(frames, signals["left_elbow_angle"], label="左肘角度", color="#D03050")
    axes[1].plot(frames, signals["right_elbow_angle"], label="右肘角度", color="#2080F0")
    axes[1].set_title(f"左右肘角度相关: {metric['elbow_angle_corr']:.3f}")
    axes[1].set_ylabel("deg")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(frames, signals["left_shoulder_z"], label="左肩高度(相对骨盆)", color="#D03050")
    axes[2].plot(frames, signals["right_shoulder_z"], label="右肩高度(相对骨盆)", color="#2080F0")
    axes[2].set_title(f"肩高度对称误差: {metric['shoulder_height_symmetry_error_m']:.4f} m")
    axes[2].set_xlabel("Frame")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "upper_body_coordination_overview.png", dpi=160)
    plt.close(fig)


def plot_upper_summary(metrics: list[dict[str, Any]], data_source_label: str) -> None:
    arm_corr = np.array([m["arm_swing_corr"] for m in metrics], dtype=float)
    elbow_corr = np.array([m["elbow_angle_corr"] for m in metrics], dtype=float)
    hand_corr = np.array([m["hand_lateral_corr"] for m in metrics], dtype=float)
    shoulder_err = np.array([m["shoulder_height_symmetry_error_m"] for m in metrics], dtype=float)
    arm_sym = np.array([m["arm_swing_symmetry"] for m in metrics], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_title("KINESIS 上半身运动学分析汇总", fontsize=18, fontweight="bold", pad=20)
    rows = [
        ("动作数量", f"{len(metrics)}"),
        ("左右摆臂相关", f"{np.nanmean(arm_corr):.3f}"),
        ("左右肘角度相关", f"{np.nanmean(elbow_corr):.3f}"),
        ("左右手侧向对称相关", f"{np.nanmean(hand_corr):.3f}"),
        ("摆臂幅度对称性", f"{np.nanmean(arm_sym) * 100:.1f}%"),
        ("肩高度对称误差", f"{np.nanmean(shoulder_err):.4f} m"),
    ]
    y = 0.76
    for label, value in rows:
        ax.text(0.32, y, label, fontsize=14, ha="right")
        ax.text(0.40, y, value, fontsize=14, ha="left", color="#18A058")
        y -= 0.105
    ax.text(
        0.5,
        0.08,
        f"数据来源：{data_source_label}；这是上半身运动学分析，不代表已实现上肢肌肉主动控制。",
        fontsize=11,
        ha="center",
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "upper_body_metrics_summary.png", dpi=160)
    plt.close(fig)


def plot_summary(metrics: list[dict[str, Any]], data_source_label: str) -> None:
    foot_alt = np.array([m["foot_alternation_corr"] for m in metrics], dtype=float)
    knee_corr = np.array([m["knee_angle_corr"] for m in metrics], dtype=float)
    ankle_corr = np.array([m["ankle_height_corr"] for m in metrics], dtype=float)
    ankle_err = np.array([m["ankle_height_symmetry_error_m"] for m in metrics], dtype=float)
    stride_sym = np.array([m["stride_amplitude_symmetry"] for m in metrics], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_title("KINESIS 下肢协同分析汇总", fontsize=18, fontweight="bold", pad=20)
    rows = [
        ("动作数量", f"{len(metrics)}"),
        ("左右脚交替相关", f"{np.nanmean(foot_alt):.3f}"),
        ("左右膝角度相关", f"{np.nanmean(knee_corr):.3f}"),
        ("左右踝高度相关", f"{np.nanmean(ankle_corr):.3f}"),
        ("步幅对称性", f"{np.nanmean(stride_sym) * 100:.1f}%"),
        ("踝高度对称误差", f"{np.nanmean(ankle_err):.4f} m"),
    ]
    y = 0.76
    for label, value in rows:
        ax.text(0.30, y, label, fontsize=14, ha="right")
        ax.text(0.38, y, value, fontsize=14, ha="left", color="#18A058")
        y -= 0.105
    ax.text(
        0.5,
        0.08,
        f"数据来源：{data_source_label}；仅使用下肢关节；本脚本不生成随机模拟步态或模拟肌肉数据。",
        fontsize=11,
        ha="center",
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coordination_metrics_summary.png", dpi=160)
    plt.close(fig)

    labels = [m["motion_id"] for m in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.45), 5))
    ax.bar(x - 0.25, foot_alt, 0.25, label="脚趾前后相关")
    ax.bar(x, knee_corr, 0.25, label="膝角相关")
    ax.bar(x + 0.25, stride_sym, 0.25, label="步幅对称性")
    ax.set_title("真实下肢协同指标（按动作）", fontsize=14, fontweight="bold")
    ax.set_ylabel("metric")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lower_limb_metrics_by_motion.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument("--smpl-dir", type=Path, default=DEFAULT_SMPL_DIR)
    parser.add_argument("--max-motions", type=int, default=20)
    args = parser.parse_args()

    require_real_motion_file(args.motion_file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    motion_dict = joblib.load(args.motion_file)
    if not isinstance(motion_dict, dict) or not motion_dict:
        raise ValueError(f"动作文件不是非空 dict: {args.motion_file}")
    source_name = args.motion_file.name.lower()
    if "kit_" in source_name:
        data_source_label = "KIT/SMPL 转换动作"
    elif "mdm_" in source_name or "t2m" in str(args.motion_file).lower():
        data_source_label = "项目 assets 中的 T2M 预生成动作"
    else:
        data_source_label = str(args.motion_file)

    metrics = []
    upper_metrics = []
    for idx, (name, motion) in enumerate(motion_dict.items()):
        if idx >= args.max_motions:
            break
        positions = get_joint_positions(motion, args.smpl_dir)
        metrics.append(analyze_motion(str(name), positions))
        upper_metrics.append(analyze_upper_body(str(name), positions))

    plot_overview(metrics[0])
    plot_summary(metrics, data_source_label)
    plot_upper_overview(upper_metrics[0])
    plot_upper_summary(upper_metrics, data_source_label)

    export_rows = [{k: v for k, v in metric.items() if k != "signals"} for metric in metrics]
    joblib.dump(export_rows, OUTPUT_DIR / "real_lower_limb_coordination_metrics.pkl")
    export_upper_rows = [{k: v for k, v in metric.items() if k != "signals"} for metric in upper_metrics]
    joblib.dump(export_upper_rows, OUTPUT_DIR / "upper_body_kinematic_metrics.pkl")

    print("真实下肢协同分析完成")
    print(f"  动作数量: {len(metrics)}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("  - lower_limb_coordination_overview.png")
    print("  - lower_limb_metrics_by_motion.png")
    print("  - coordination_metrics_summary.png")
    print("  - upper_body_coordination_overview.png")
    print("  - upper_body_metrics_summary.png")


if __name__ == "__main__":
    main()
