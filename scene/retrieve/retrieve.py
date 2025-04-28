import os
import setGPU
import csv
import pickle
import re
from sentence_transformers import SentenceTransformer, models
from os import path as osp
from tqdm import tqdm
import argparse
from architecture import LLMChat
from utils import load_file, retrieve_topk, generate_code_snippet, save_scenic_code


# no need for faiss currently
# import faiss

# Argument parsing
parser = argparse.ArgumentParser(description="Set up configurations for your script.")
parser.add_argument('--port_ip', type=int, default=2000, help='Port IP address (default: 2000)')
parser.add_argument('--topk', type=int, default=3, help='Top K value (default: 3) for retrieval')
parser.add_argument('--model', type=str, default='gpt-4o', help="Model name (default: 'gpt-4o'), also support transformers model")
parser.add_argument('--use_llm', action='store_true', help='if use llm for generating new snippets')
args = parser.parse_args()

# Configuration
port_ip = args.port_ip
topk = args.topk
use_llm = args.use_llm

# LLM model initialization
llm_model = LLMChat(args.model)
local_path = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))
extraction_prompt = load_file(osp.join(local_path, 'retrieve', 'prompts', 'extraction.txt'))
behavior_prompt = load_file(osp.join(local_path, 'retrieve', 'prompts', 'behavior.txt'))
geometry_prompt = load_file(osp.join(local_path, 'retrieve', 'prompts', 'geometry.txt'))
spawn_prompt = load_file(osp.join(local_path, 'retrieve', 'prompts', 'spawn.txt'))
scenario_descriptions = load_file(osp.join(local_path, 'retrieve', 'scenario_descriptions.txt')).split('\n')

# 🔥 修改开始：本地加载 sentence-t5-large 模型
model_dir = r"D:\sceneMain\chatScene\models\sentence-t5-large"
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"本地模型路径不存在：{model_dir}")

required_files = ["config.json", "pytorch_model.bin"]
for filename in required_files:
    if not os.path.exists(os.path.join(model_dir, filename)):
        raise FileNotFoundError(f"缺少必要的文件: {filename} 在 {model_dir} 中")

word_embedding_model = models.Transformer(model_dir, max_seq_length=512)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode='mean'
)
encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda')
print("✅ 成功加载本地 sentence-t5-large 模型！")
# 🔥 修改结束

# Load the database
with open(osp.join(local_path, 'retrieve/database_v1.pkl'), 'rb') as file:
    database = pickle.load(file)

behavior_descriptions = database['behavior']['description']
geometry_descriptions = database['geometry']['description']
spawn_descriptions = database['spawn']['description']
behavior_snippets = database['behavior']['snippet']
geometry_snippets = database['geometry']['snippet']
spawn_snippets = database['spawn']['snippet']

behavior_embeddings = encoder.encode(behavior_descriptions, device='cuda', convert_to_tensor=True)
geometry_embeddings = encoder.encode(geometry_descriptions, device='cuda', convert_to_tensor=True)
spawn_embeddings = encoder.encode(spawn_descriptions, device='cuda', convert_to_tensor=True)

# This is the head for scenic file, you can modify the carla map or ego model here
head = '''param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
'''

log_file_path = osp.join(local_path, 'safebench', 'scenario', 'scenario_data', 'scenic_data', 'dynamic_scenario', 'dynamic_log.csv')

# Write log results
with open(log_file_path, mode='w', newline='') as file:
    log_writer = csv.writer(file)
    log_writer.writerow(['Scenario', 'AdvObject', 'Behavior Description', 'Behavior Snippet', 'Geometry Description', 'Geometry Snippet', 'Spawn Description', 'Spawn Snippet', 'Success'])

    # Process each scenario description
    for q, current_scenario in tqdm(enumerate(scenario_descriptions)):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": extraction_prompt.format(scenario=current_scenario)},
        ]

        response = llm_model.generate(messages)

        try:
            match = re.search(r"Adversarial Object:(.*?)Behavior:(.*?)Geometry:(.*?)Spawn Position:(.*)", response, re.DOTALL)
            if not match:
                raise ValueError("Failed to extract components from the response")

            current_adv_object, current_behavior, current_geometry, current_spawn = [s.strip() for s in match.groups()]

            # Retrieve the top K relevant snippets
            top_behavior_descriptions, top_behavior_snippets = retrieve_topk(encoder, topk, behavior_descriptions, behavior_snippets, behavior_embeddings, current_behavior)
            top_geometry_descriptions, top_geometry_snippets = retrieve_topk(encoder, topk, geometry_descriptions, geometry_snippets, geometry_embeddings, current_geometry)
            top_spawn_descriptions, top_spawn_snippets = retrieve_topk(encoder, topk, spawn_descriptions, spawn_snippets, spawn_embeddings, current_spawn)

            # Generate code snippets using the LLM
            generated_behavior_code = generate_code_snippet(
                llm_model, behavior_prompt, top_behavior_descriptions, top_behavior_snippets, current_behavior, topk, use_llm
            )

            generated_geometry_code = generate_code_snippet(
                llm_model, geometry_prompt, top_geometry_descriptions, top_geometry_snippets, current_geometry, topk, use_llm
            )

            generated_spawn_code = generate_code_snippet(
                llm_model, spawn_prompt, top_spawn_descriptions, top_spawn_snippets, current_spawn, topk, use_llm
            )

            # Log the results
            log_writer.writerow([current_scenario, current_adv_object, current_behavior, generated_behavior_code, current_geometry, generated_geometry_code, current_spawn, generated_spawn_code, 1])

            Town, generated_geometry_code = generated_geometry_code.split('\n', 1)
            scenic_code = '\n'.join([f"'''{current_scenario}'''", Town, head, generated_behavior_code, generated_geometry_code, generated_spawn_code.format(AdvObject=current_adv_object)])
            save_scenic_code(local_path, port_ip, scenic_code, q)

        except Exception as e:
            log_writer.writerow([current_scenario, '', '', '', '', '', '', '', 0])
            print(f"Failure for scenario: {current_scenario} - Error: {e}")
