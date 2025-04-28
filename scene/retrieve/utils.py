import re
from os import path as osp
import shutil
from safebench.util.scenic_utils import ScenicSimulator

def load_file(file_path):
    try:
        # 指定编码为 utf-8 以避免编码错误
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError as e:
        print(f"Error reading file {file_path}: {e}")
        raise  # 重新抛出异常以便上层调用处理
    except Exception as e:
        print(f"An unexpected error occurred while reading file {file_path}: {e}")
        raise

def retrieve_topk(model, topk, descriptions, snippets, embeddings, current_description):
    try:
        current_embedding = model.encode([current_description], device='cuda', convert_to_tensor=True)
        scores = (current_embedding @ embeddings.T).squeeze(0)
        top_indices = scores.topk(k=topk).indices
        top_descriptions = [descriptions[i] for i in top_indices]
        top_snippets = [snippets[i] for i in top_indices]
        return top_descriptions, top_snippets
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return [], []

def extract_scenic_code(text):
    try:
        pattern = r"```scenic(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        else:
            print("No Scenic code found in the text.")
            return ""
    except Exception as e:
        print(f"Error extracting Scenic code: {e}")
        return ""

def generate_code_snippet(llm_model, category_prompt, descriptions, snippets, current_description, topk, use_llm):
    system_prompt ='''Your goal is to assist me in writing snippets using Scenic 2.1 for CARLA simulation. Scenic is a domain-specific probabilistic programming language designed for modeling environments in cyber-physical systems like robots and autonomous vehicles. Please adhere strictly to the Scenic 2.1 API, avoiding the use of any non-existent APIs or the Python random package.'''

    try:
        if not use_llm:
            return snippets[0]
        
        content = '\n'
        for j in range(topk):
            content += f'Description: {descriptions[j]}\nSnippet:\n```scenic\n{snippets[j]}```\n'

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": category_prompt.format(content=content, current_description=current_description)}
        ]
        generated_code = llm_model.generate(messages)
        return extract_scenic_code(generated_code)
    
    except Exception as e:
        print(f"An error occurred while generating code: {e}")
        return None

def save_scenic_code(local_path, port_ip, scenic_code, q):
    file_path = osp.join(local_path, f'safebench/scenario/scenario_data/scenic_data/dynamic_scenario/dynamic_{q}.scenic')
    backup_path = osp.join(local_path, f'safebench/scenario/scenario_data/scenic_data/dynamic_scenario/dynamic_{q}.txt')

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(scenic_code)

        extra_params = {'port': port_ip, 'traffic_manager_port': port_ip + 6000}

        print("Checking if the Scenic code is compilable...")
        ScenicSimulator(file_path, extra_params)
        print(f"Scenic code saved and verified as compilable at {file_path}")
        return True
    except Exception as e:
        print(f"Failure in compiling Scenic code with ID {q}: {str(e)}")
        shutil.move(file_path, backup_path)  # Move the problematic Scenic file to a .txt extension
        print(f"Moved problematic Scenic file to {backup_path}")
        return False
