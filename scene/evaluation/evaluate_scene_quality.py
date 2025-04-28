import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
import cv2

# 文件路径
DESCRIPTION_FILE = 'D:/sceneMain/chatScene/retrieve/scenario_descriptions.txt'
HISTORY_FILE = 'D:/sceneMain/chatScene/retrieve/scenario_history.txt'
SCENE_IMAGE_DIR = 'D:/sceneMain/chatScene/outputs/'

# 加载最新的场景描述（只读取文件的第一行）
def load_latest_description(path):
    """只读取描述文件中的第一行"""
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()  # 读取第一行
    return first_line

# 将新的场景描述追加到历史记录文件
def append_to_history(new_description, history_path):
    """将新的场景描述追加到历史文件"""
    with open(history_path, 'a', encoding='utf-8') as f:
        f.write(new_description + '\n')

# 计算图像相似度（使用结构相似度）
def calculate_image_similarity(image1, image2):
    """计算两张图像之间的相似度"""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = cv2.quality.QualitySSIM_compute(gray1, gray2)
    return score

# 计算场景的多样性（使用生成图像之间的距离）
def calculate_scene_diversity(image_dir):
    """计算所有图像之间的多样性"""
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(image_dir, filename))
            images.append(img)

    # 转换为数组（每个图像的特征）
    image_features = [np.reshape(img, (-1, 3)) for img in images]
    image_features = np.concatenate(image_features, axis=0)

    # 计算每对图像的最小距离
    distances = pairwise_distances_argmin_min(image_features, image_features)
    avg_distance = np.mean(distances[1])  # 平均最小距离
    return avg_distance

# 评估场景质量：语义一致性，图像质量，多样性
def evaluate_scene_quality(image_dir):
    """评估场景的质量"""

    # 语义一致性（假设为手动指定或从其他方法中获得）
    semantic_consistency = 0.9  # 假设的值，通常需要根据具体情况进行计算

    # 图像质量：假设使用已有的参考图像进行评估（此处为一个示例）
    reference_image = cv2.imread('D:/sceneMain/chatScene/reference_image.png')  # 参考图像
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    avg_image_quality = 0
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_dir, image_file))
        similarity = calculate_image_similarity(reference_image, img)
        avg_image_quality += similarity
    avg_image_quality /= len(image_files)

    # 多样性
    diversity = calculate_scene_diversity(image_dir)

    # 打印评估结果
    print(f"语义一致性: {semantic_consistency}")
    print(f"平均图像质量: {avg_image_quality}")
    print(f"场景多样性: {diversity}")

    return semantic_consistency, avg_image_quality, diversity

# 主函数
def main():
    # 读取最新的场景描述
    latest_description = load_latest_description(DESCRIPTION_FILE)

    # 将描述追加到历史记录
    append_to_history(latest_description, HISTORY_FILE)

    # 打印最新描述
    print(f"最新的场景描述: {latest_description}")

    # 评估生成的场景质量
    semantic_consistency, avg_image_quality, diversity = evaluate_scene_quality(SCENE_IMAGE_DIR)

    # 可以根据需要将评估结果保存为JSON或其他格式
    evaluation_results = {
        "semantic_consistency": semantic_consistency,
        "avg_image_quality": avg_image_quality,
        "diversity": diversity
    }

    # 保存评估结果到文件
    with open('D:/sceneMain/chatScene/outputs/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

if __name__ == "__main__":
    main()
