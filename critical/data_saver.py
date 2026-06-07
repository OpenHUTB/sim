# utils/data_saver.py
# 数据持久化工具：CSV 日志、JSON 指标、模型检查点、OpenSCENARIO 导出

import os
import csv
import json
import pickle
from datetime import datetime


# ================================================================
# 目录管理
# ================================================================

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def init_result_dir(scenario_name, base_dir="results"):
    """为场景/实验创建带时间戳的结果目录，返回目录结构字典"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(base_dir, f"{scenario_name}_{timestamp}")

    dirs = {
        "root": root,
        "data": ensure_dir(os.path.join(root, "data")),
        "logs": ensure_dir(os.path.join(root, "logs")),
        "xosc": ensure_dir(os.path.join(root, "xosc")),
        "models": ensure_dir(os.path.join(root, "models")),
        "plots": ensure_dir(os.path.join(root, "plots")),
    }
    return dirs


# ================================================================
# CSV 日志
# ================================================================

def save_vehicle_log(filepath, logs, headers=None):
    """
    保存车辆轨迹日志到 CSV。

    logs: list of list，每行一条记录
    headers: list of str，列名
    """
    if headers is None:
        headers = ["timestamp", "x", "y", "z", "speed_kmh", "steer", "brake"]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(logs)
    return filepath


def save_training_log(filepath, episode_data):
    """
    逐行追加训练日志（每 episode 一行）。

    episode_data: dict，包含 episode, reward, loss, epsilon 等字段
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(episode_data)


def load_training_log(filepath):
    """读取训练日志返回 list[dict]"""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ================================================================
# JSON 指标
# ================================================================

def save_metrics(filepath, metrics_dict):
    """保存评估指标为 JSON"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    return filepath


def load_metrics(filepath):
    """从 JSON 加载评估指标"""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================================================
# 模型检查点
# ================================================================

def save_model(filepath, model_state_dict):
    """使用 pickle 保存模型参数字典"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model_state_dict, f)
    return filepath


def load_model(filepath):
    """从 pickle 加载模型参数"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ================================================================
# OpenSCENARIO
# ================================================================

def save_xosc(filepath, xosc_content):
    """保存 .xosc 场景文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(xosc_content)
    return filepath


# ================================================================
# 场景配置存档
# ================================================================

def save_scenario_config(filepath, config_dict):
    """保存场景配置副本（便于实验结果复现）"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)


# ================================================================
# 综合保存
# ================================================================

def save_experiment_results(output_dir, scenario_name, results):
    """
    保存完整实验结果。

    results: dict，可包含:
        - metrics: dict, 评估指标
        - xosc: str, OpenSCENARIO 内容
        - vehicle_logs: list, 车辆轨迹
        - episode_rewards: list, 每集奖励
    """
    ensure_dir(output_dir)

    if "metrics" in results:
        save_metrics(
            os.path.join(output_dir, f"{scenario_name}_metrics.json"),
            results["metrics"],
        )
    if "xosc" in results:
        save_xosc(
            os.path.join(output_dir, f"{scenario_name}.xosc"),
            results["xosc"],
        )
    if "vehicle_logs" in results:
        save_vehicle_log(
            os.path.join(output_dir, f"{scenario_name}_trajectory.csv"),
            results["vehicle_logs"],
        )
    if "episode_rewards" in results:
        save_metrics(
            os.path.join(output_dir, f"{scenario_name}_rewards.json"),
            {"episode_rewards": results["episode_rewards"]},
        )
