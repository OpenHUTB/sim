import os
import json
import random

# 设置 CUDA_VISIBLE_DEVICES 环境变量，选择 GPU 0 和 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 设置场景描述文件路径
scene_desc_file = "D:/sceneMain/chatScene/retrieve/scenario_descriptions.txt"
output_dir = "D:/sceneMain/chatScene/demo/generated_scenes"
output_file = os.path.join(output_dir, "generated_dynamic_scene.json")

# 确保输出目录存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取场景描述文件
def read_scene_descriptions(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scene description file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()

# 使用第一行场景描述并生成相应的场景数据
def generate_scene_data(descriptions):
    if not descriptions:
        raise ValueError("Scene description file is empty.")

    selected_description = descriptions[0].strip()

    scene_data = {
        "description": selected_description,
        "vehicles": [
            {"type": "car", "x": random.uniform(0, 50), "y": random.uniform(0, 50), "yaw": random.uniform(0, 360)},
            {"type": "truck", "x": random.uniform(0, 50), "y": random.uniform(0, 50), "yaw": random.uniform(0, 360)}
        ],
        "pedestrians": [
            {"type": "person", "x": random.uniform(0, 50), "y": random.uniform(0, 50), "yaw": random.uniform(0, 360)}
        ]
    }
    return scene_data

# 将生成的场景数据保存到 JSON 文件
def save_scene_data(scene_data, output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(scene_data, output_file, indent=4, ensure_ascii=False)

def main():
    try:
        scene_descriptions = read_scene_descriptions(scene_desc_file)
        scene_data = generate_scene_data(scene_descriptions)
        save_scene_data(scene_data, output_file)

        print(f"Scene data generated and saved to {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
