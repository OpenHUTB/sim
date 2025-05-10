# 类脑视觉识别模型：CORnet-S复现、CORnet-Z 系列优化与 Brain-Score 评估



## 📘 项目依赖

本项目主要参考[CORnet](https://github.com/dicarlolab/CORnet)项目以及[Brainscore](https://github.com/brain-score/vision)项目，需按照其官方要求安装好相关依赖。

本项目使用的数据集为缩小版的Imagenet数据集tiny-imagenet-200，下载地址为http://cs231n.stanford.edu/tiny-imagenet-200.zip。数据集需要经过处理，处理过后下载地址可见https://pan.baidu.com/s/1eTfqOfTwYIaxzmYNqMjv5g?pwd=k3ys ，提取码: k3ys。

---

## 🚀 环境配置

本项目按照参考项目推荐使用python版本为3.11，其他版本可能出现安装依赖报错问题。可通过以下命令创建虚拟环境。

```bash
# 创建虚拟环境
conda create -n brainvision python=3.11
conda activate brainvision
```

## 📁 项目结构说明

```bash
{YOUR_FILE}/
	CORnet-master
	brainscore
	tiny-imagenet-200
```

#### CORnet-S复现以及模型训练：

```bash
CORnet-master/
	cornet/
		cornet_s.py
		cornet_z.py
		cornet_z_se.py
	result                       # 保存结果的文件夹
	draw.py                      # 绘制损失曲线
	extract_activations.py       # 绘制cornet-z模型激活图
	extract_activations_se.py    # 绘制cornet-z-se模型激活图
	run.py
	run_se.py
```

#### 类脑相似性评估：

```bash
brainscore/
	alexnet_model.py          # AlexNet的brainscore封装
	brainscore_model.py       # CORnet模型的brainscore封装
	cbma.py
	cornet_z_cbma.py          # 加入CBAM模块的CORnet-Z
	cornet_z_cbma_model.py    # CORnet+CBMA模型的brainscore封装
	cornet_z_se.py            # 加入SE模块的CORnet-Z
	cornet_z_se_model.py      # CORnet+SE模型的brainscore封装
	cornet_z_vob.py           # 加入VOneBlock的CORnet-Z
	evaluate.py               # CORnet-Z的类脑评估
	evaluate_alexnet.py       # AlexNet模型评估
	evaluate_resnet.py        # ResNet-18模型评估
	evaluate_z_cbma.py        # CBAM模型评估
	evaluate_z_se.py          # CORnet-Z+SE的类脑评估
	resnet_model.py           # Resnet-18模型的brainscore封装
	utils.py
	voneblock.py
```

## 🏃‍♂️ 模型训练与测试

### 1. 使用 run.py / run_se.py 进行训练：

```bash
# 训练CORnet-Z模型
python run.py train --model Z --workers 20 --ngpus 1 --step_size 10 --epochs 40 --lr .01 --data_path <图像路径> --output_path <运行结果保存路径，推荐为run.py同路径下的result文件夹>

# 训练CORnet-Z+SE模型
python run_se.py train --model Z_SE --workers 20 --ngpus 1 --step_size 10 --epochs 40 --lr .01 --data_path <图像路径> --output_path <运行结果保存路径，推荐为run_se.py同路径下的result文件夹>
```

### 2. 绘制损失曲线以及激活图：

```bash
# 绘制损失曲线
python draw.py

# 绘制激活图
python extract_activations.py
```

------

## 📊 Brain-Score 类脑评估

使用 MajajHong2015 神经数据对齐评估：

```bash
# CORnet-Z 评估
python evaluate.py

# CORnet-Z+SE 评估
python evaluate_z_se.py

# CBAM/ResNet/AlexNet 模型评估
python evaluate_z_cbma.py
python evaluate_resnet.py
python evaluate_alexnet.py
```

## 🙏 引用与致谢

本项目参考以下开源工作并在其基础上进行改进与扩展：

- CORnet 模型：https://github.com/dicarlolab/CORnet
- Brain-Score 框架：https://github.com/brain-score/vision
