import subprocess

# Step 1: 使用 scenic 环境生成场景 JSON
subprocess.run([
    r"D:\Users\24448\anaconda3\envs\scenic\python.exe",
    "D:/sceneMain/chatScene/demo/generate_dynamic_scenes.py"
])

# Step 2: 使用 carla 环境读取并加载 JSON 到 Carla 世界
subprocess.run([
    r"D:\Users\24448\anaconda3\envs\carla\python.exe",
    "D:/sceneMain/chatScene/demo/load_dynamic_scene.py"
])
