import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 加入模型所在子目录
import sys
sys.path.append('./cornet')  # 👈 加入子文件夹路径
from cornet_z import CORnet_Z  # 👈 从子模块中导入模型

# ----------- 加载模型并注册中间层 hook ----------
model = CORnet_Z()
model.eval()

activations = {}

def get_hook(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

# 注册 hook 到每一层
model[0].output.register_forward_hook(get_hook('V1'))
model[1].output.register_forward_hook(get_hook('V2'))
model[2].output.register_forward_hook(get_hook('V4'))
model[3].output.register_forward_hook(get_hook('IT'))

# ---------- 图像预处理 ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 载入并预处理图像
image_path = 'your_image.jpg'  # 👈 替换为你的图像路径
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# ---------- 前向传播 ----------
with torch.no_grad():
    model(input_tensor)

# ---------- 可视化中间激活 ----------
def show_activation(name, act_tensor, num_channels=6):
    act = act_tensor.squeeze(0)[:num_channels]
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
    fig.suptitle(name)
    for i in range(num_channels):
        axes[i].imshow(act[i], cmap='viridis')
        axes[i].axis('off')
    plt.savefig(f'{name}_activations.png')
    plt.close()

for layer_name in ['V1', 'V2', 'V4', 'IT']:
    show_activation(layer_name, activations[layer_name])
