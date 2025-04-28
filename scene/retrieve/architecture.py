import os
import openai
import torch
import transformers

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = 'sk-...s0kA'  # 请替换为你的实际 API 密钥


class LLMChat:
    def __init__(self, model_name='gpt-4', use_gpu=True):
        """
        初始化聊天模型。
        - model_name: 使用的模型名称（例如 'gpt-4' 或 HuggingFace 模型名称）。
        - use_gpu: 是否使用 GPU（默认是 True）。
        """
        super(LLMChat, self).__init__()
        self.model_name = model_name
        self.use_gpu = use_gpu

        # 如果是 GPT 模型，使用 OpenAI API
        if model_name.startswith('gpt'):
            openai.api_key = os.environ["OPENAI_API_KEY"]  # 设置 OpenAI API 密钥
        else:
            # 设置 HuggingFace 模型并选择 GPU 或 CPU
            self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.float32},  # 使用 float32 避免兼容性问题
                device=0 if self.device == "cuda" else -1,  # 如果使用 GPU，选择 GPU 设备
            )

    def generate(self, messages, max_new_tokens=500):
        """
        生成文本。
        - messages: 输入消息（适用于 GPT 模型和 HuggingFace 模型）。
        - max_new_tokens: 最大生成的 tokens 数量。
        """
        if self.model_name.startswith('gpt'):
            # 对于 OpenAI GPT 模型，使用新的聊天 API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,  # 将消息数组直接传递给 chat 模型
                temperature=0.7,  # 控制生成文本的多样性
                max_tokens=max_new_tokens,
            )
            return response['choices'][0]['message']['content'].strip()  # 返回生成的文本

        else:
            # 对于 HuggingFace 模型，使用文本生成 pipeline
            if isinstance(messages, str):
                # 如果输入是单个字符串，构建消息格式
                messages = [{"role": "user", "content": messages}]

            outputs = self.pipeline(
                messages[0]['content'],  # 传递用户消息的内容
                max_new_tokens=max_new_tokens,
                do_sample=True,  # 启用采样，生成更多样的文本
            )
            return outputs[0]["generated_text"].strip()  # 返回生成的文本
