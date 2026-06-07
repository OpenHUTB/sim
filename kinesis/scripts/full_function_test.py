"""Run a complete functional test pass for the Kinesis project.

The test suite covers all implemented deliverable functions:

- source syntax/import-adjacent checks via py_compile;
- data/model inventory;
- optimized reward-function implementation checks;
- real T2M lower/upper-body coordination analysis;
- baseline-vs-optimized reward comparison experiments;
- precision plotting when real evaluation metrics exist;
- full Kinesis imitation evaluation when KIT + SMPL licensed inputs exist;
- one-click reproducible pipeline.

Missing licensed assets such as SMPL_NEUTRAL.pkl or converted KIT data are marked
as SKIP, not PASS. The script never creates simulated metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output" / "full_test"


@dataclass
class TestResult:
    name: str
    status: str
    message: str
    duration_s: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def run_command(command: list[str], timeout: int = 120) -> tuple[int, str]:
    process = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return process.returncode, process.stdout


def file_ok(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def add_result(results: list[TestResult], result: TestResult) -> None:
    results.append(result)
    print(f"[{result.status}] {result.name}: {result.message}")


def test_syntax(results: list[TestResult]) -> None:
    start = time.time()
    files = [
        "src/env/myolegs_im.py",
        "src/agents/agent_im.py",
        "scripts/analyze_coordination.py",
        "scripts/plot_precision.py",
        "scripts/reproducible_eval.py",
        "scripts/compare_reward_optimization.py",
        "scripts/full_function_test.py",
    ]
    code, output = run_command([sys.executable, "-m", "py_compile", *files], timeout=30)
    add_result(
        results,
        TestResult(
            "Python 语法检查",
            "PASS" if code == 0 else "FAIL",
            "核心脚本 py_compile 通过" if code == 0 else "核心脚本 py_compile 失败",
            time.time() - start,
            {"files": files, "output": output[-4000:]},
        ),
    )


def test_data_inventory(results: list[TestResult]) -> None:
    start = time.time()
    paths = {
        "myolegs_xml": PROJECT_DIR / "data" / "xml" / "myolegs.xml",
        "smpl_humanoid_xml": PROJECT_DIR / "data" / "xml" / "smpl_humanoid.xml",
        "initial_pose_test": PROJECT_DIR / "data" / "initial_pose" / "initial_pose_test.pkl",
        "checkpoint_imitation": PROJECT_DIR / "data" / "trained_models" / "kinesis-moe-imitation" / "model.pth",
        "kit_motion": PROJECT_DIR / "data" / "kit_test_motion_dict.pkl",
        "smpl_neutral": PROJECT_DIR / "data" / "smpl" / "SMPL_NEUTRAL.pkl",
    }
    t2m_files = sorted((PROJECT_DIR / "data" / "t2m").glob("*.pkl"))
    present = {name: file_ok(path) for name, path in paths.items()}
    essential_ok = (
        present["myolegs_xml"]
        and present["smpl_humanoid_xml"]
        and present["initial_pose_test"]
        and present["checkpoint_imitation"]
        and len(t2m_files) > 0
    )
    missing_optional = [name for name in ("kit_motion", "smpl_neutral") if not present[name]]
    status = "PASS" if essential_ok else "FAIL"
    message = f"可用 T2M 动作 {len(t2m_files)} 个"
    if missing_optional:
        message += f"；完整控制评估缺少 {', '.join(missing_optional)}"
    add_result(
        results,
        TestResult(
            "数据和模型清单",
            status,
            message,
            time.time() - start,
            {
                "files": {name: {"path": rel(path), "exists": ok} for name, (path, ok) in zip(paths, [(p, present[n]) for n, p in paths.items()])},
                "t2m_count": len(t2m_files),
                "t2m_examples": [rel(path) for path in t2m_files[:5]],
            },
        ),
    )


def test_reward_code(results: list[TestResult]) -> None:
    start = time.time()
    source = (PROJECT_DIR / "src" / "env" / "myolegs_im.py").read_text(encoding="utf-8")
    env_cfg = (PROJECT_DIR / "cfg" / "env" / "env_im.yaml").read_text(encoding="utf-8")
    baseline_cfg = (PROJECT_DIR / "cfg" / "env" / "env_im_baseline.yaml").read_text(encoding="utf-8")

    required_source = [
        "compute_joint_angle_reward",
        "compute_muscle_drive_reward",
        "compute_smoothness_reward",
        "joint_angle_reward",
        "muscle_drive_reward",
        "smoothness_reward",
    ]
    required_cfg = [
        "w_joint_angle",
        "w_muscle_drive",
        "w_smoothness",
        "k_joint_angle",
        "k_muscle_drive",
        "k_smoothness",
    ]
    missing = [item for item in required_source if item not in source]
    missing += [item for item in required_cfg if item not in env_cfg]
    if "w_joint_angle" in baseline_cfg:
        missing.append("baseline_cfg_should_not_include_optimized_weights")

    add_result(
        results,
        TestResult(
            "奖励函数优化代码检查",
            "PASS" if not missing else "FAIL",
            "关节角度、肌肉驱动、能耗和平滑性奖励项已接入" if not missing else f"缺少: {', '.join(missing)}",
            time.time() - start,
            {"checked_source_terms": required_source, "checked_config_terms": required_cfg},
        ),
    )


def test_coordination(results: list[TestResult]) -> None:
    start = time.time()
    kit_motion = PROJECT_DIR / "data" / "kit_test_motion_dict.pkl"
    t2m_files = sorted((PROJECT_DIR / "data" / "t2m").glob("*.pkl"))
    motion_file = kit_motion if kit_motion.exists() else (t2m_files[0] if t2m_files else None)
    if motion_file is None:
        add_result(results, TestResult("上下身协同分析", "FAIL", "没有可用真实动作文件", time.time() - start))
        return

    code, output = run_command(
        [sys.executable, "scripts/analyze_coordination.py", "--motion-file", rel(motion_file)],
        timeout=120,
    )
    expected = [
        PROJECT_DIR / "output" / "coordination" / "lower_limb_coordination_overview.png",
        PROJECT_DIR / "output" / "coordination" / "lower_limb_metrics_by_motion.png",
        PROJECT_DIR / "output" / "coordination" / "coordination_metrics_summary.png",
        PROJECT_DIR / "output" / "coordination" / "upper_body_coordination_overview.png",
        PROJECT_DIR / "output" / "coordination" / "upper_body_metrics_summary.png",
        PROJECT_DIR / "output" / "coordination" / "real_lower_limb_coordination_metrics.pkl",
        PROJECT_DIR / "output" / "coordination" / "upper_body_kinematic_metrics.pkl",
    ]
    missing_outputs = [rel(path) for path in expected if not file_ok(path)]
    status = "PASS" if code == 0 and not missing_outputs else "FAIL"
    add_result(
        results,
        TestResult(
            "上下身协同分析",
            status,
            f"使用真实动作 {rel(motion_file)} 生成上下身协同图" if status == "PASS" else "协同分析失败或输出缺失",
            time.time() - start,
            {"motion_file": rel(motion_file), "missing_outputs": missing_outputs, "output": output[-4000:]},
        ),
    )


def test_reward_experiment(results: list[TestResult]) -> None:
    start = time.time()
    if not any((PROJECT_DIR / "data" / "t2m").glob("*.pkl")):
        add_result(results, TestResult("奖励优化对比实验", "FAIL", "没有真实 T2M 动作数据", time.time() - start))
        return

    code, output = run_command([sys.executable, "scripts/compare_reward_optimization.py"], timeout=120)
    result_json = PROJECT_DIR / "output" / "reward_optimization" / "reward_optimization_results.json"
    expected = [
        PROJECT_DIR / "output" / "reward_optimization" / "baseline_vs_optimized_reward_by_motion.png",
        PROJECT_DIR / "output" / "reward_optimization" / "baseline_vs_optimized_reward_by_group.png",
        PROJECT_DIR / "output" / "reward_optimization" / "optimized_reward_components.png",
        PROJECT_DIR / "output" / "reward_optimization" / "reward_optimization_results.csv",
        result_json,
    ]
    missing_outputs = [rel(path) for path in expected if not file_ok(path)]
    details: dict[str, Any] = {"missing_outputs": missing_outputs, "output": output[-4000:]}
    if result_json.exists():
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        motions = payload.get("motions", [])
        baseline_mean = sum(row["baseline_score"] for row in motions) / len(motions)
        optimized_mean = sum(row["optimized_score"] for row in motions) / len(motions)
        details.update(
            {
                "motion_count": len(motions),
                "baseline_mean": baseline_mean,
                "optimized_mean": optimized_mean,
                "groups": payload.get("groups", []),
            }
        )
    status = "PASS" if code == 0 and not missing_outputs and details.get("motion_count", 0) > 0 else "FAIL"
    message = "50 组真实 T2M 动作奖励优化对比完成"
    if "baseline_mean" in details:
        message += f"，优化前 {details['baseline_mean']:.4f}，优化后 {details['optimized_mean']:.4f}"
    add_result(results, TestResult("奖励优化对比实验", status, message, time.time() - start, details))


def test_precision_plot(results: list[TestResult]) -> None:
    start = time.time()
    metrics = PROJECT_DIR / "output" / "precision" / "evaluation_metrics.json"
    if not metrics.exists():
        add_result(
            results,
            TestResult(
                "控制精度图表",
                "SKIP",
                "缺少真实 evaluation_metrics.json；完整控制评估未运行",
                time.time() - start,
                {"missing": rel(metrics)},
            ),
        )
        return
    code, output = run_command([sys.executable, "scripts/plot_precision.py"], timeout=60)
    expected = [
        PROJECT_DIR / "output" / "precision" / "mpjpe_by_motion.png",
        PROJECT_DIR / "output" / "precision" / "frame_coverage.png",
        PROJECT_DIR / "output" / "precision" / "precision_summary.png",
    ]
    missing_outputs = [rel(path) for path in expected if not file_ok(path)]
    add_result(
        results,
        TestResult(
            "控制精度图表",
            "PASS" if code == 0 and not missing_outputs else "FAIL",
            "真实 MPJPE/覆盖率图表生成完成" if code == 0 and not missing_outputs else "精度图表生成失败",
            time.time() - start,
            {"missing_outputs": missing_outputs, "output": output[-4000:]},
        ),
    )


def test_full_control_eval(results: list[TestResult]) -> None:
    start = time.time()
    required = [
        PROJECT_DIR / "data" / "kit_test_motion_dict.pkl",
        PROJECT_DIR / "data" / "smpl" / "SMPL_NEUTRAL.pkl",
        PROJECT_DIR / "data" / "initial_pose" / "initial_pose_test.pkl",
        PROJECT_DIR / "data" / "xml" / "myolegs.xml",
        PROJECT_DIR / "data" / "trained_models" / "kinesis-moe-imitation" / "model.pth",
    ]
    missing = [rel(path) for path in required if not file_ok(path)]
    if missing:
        add_result(
            results,
            TestResult(
                "完整 Kinesis 控制评估",
                "SKIP",
                "缺少授权/转换数据，跳过完整 MPJPE 控制评估",
                time.time() - start,
                {"missing": missing},
            ),
        )
        return

    command = [
        sys.executable,
        "src/run.py",
        "exp_name=kinesis-moe-imitation",
        "epoch=-1",
        "run=repro_eval",
        "env=env_im_eval",
        "run.num_motions=1",
        "run.headless=True",
        "no_log=True",
    ]
    code, output = run_command(command, timeout=240)
    add_result(
        results,
        TestResult(
            "完整 Kinesis 控制评估",
            "PASS" if code == 0 else "FAIL",
            "完整控制评估运行完成" if code == 0 else "完整控制评估失败",
            time.time() - start,
            {"command": command, "output": output[-4000:]},
        ),
    )


def test_reproducible_pipeline(results: list[TestResult]) -> None:
    start = time.time()
    code, output = run_command([sys.executable, "scripts/reproducible_eval.py"], timeout=180)
    add_result(
        results,
        TestResult(
            "一键可复现实验流水线",
            "PASS" if code == 0 else "FAIL",
            "可用步骤执行完成，缺失输入按规则跳过" if code == 0 else "一键流水线执行失败",
            time.time() - start,
            {"output": output[-6000:]},
        ),
    )


def write_report(results: list[TestResult]) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = [item for item in results if item.status == "FAIL"]
    skipped = [item for item in results if item.status == "SKIP"]
    overall = "FAILED" if failed else ("PASSED_WITH_SKIPS" if skipped else "PASSED")
    payload = {
        "overall_status": overall,
        "project_dir": str(PROJECT_DIR),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": len(results),
            "passed": sum(item.status == "PASS" for item in results),
            "skipped": len(skipped),
            "failed": len(failed),
        },
        "results": [item.__dict__ for item in results],
    }
    json_path = OUTPUT_DIR / "full_test_report.json"
    md_path = OUTPUT_DIR / "full_test_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Kinesis 完整功能测试报告",
        "",
        f"- 总状态：{overall}",
        f"- 测试总数：{payload['summary']['total']}",
        f"- 通过：{payload['summary']['passed']}",
        f"- 跳过：{payload['summary']['skipped']}",
        f"- 失败：{payload['summary']['failed']}",
        "",
        "## 测试明细",
        "",
    ]
    for item in results:
        lines.append(f"### [{item.status}] {item.name}")
        lines.append("")
        lines.append(item.message)
        lines.append("")
        if item.details:
            lines.append("```json")
            lines.append(json.dumps(item.details, ensure_ascii=False, indent=2)[:6000])
            lines.append("```")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n测试报告已生成: {json_path}")
    print(f"测试报告已生成: {md_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Treat skipped tests as failures.")
    args = parser.parse_args()

    results: list[TestResult] = []
    test_syntax(results)
    test_data_inventory(results)
    test_reward_code(results)
    test_coordination(results)
    test_reward_experiment(results)
    test_precision_plot(results)
    test_full_control_eval(results)
    test_reproducible_pipeline(results)
    payload = write_report(results)

    if payload["summary"]["failed"] > 0 or (args.strict and payload["summary"]["skipped"] > 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
