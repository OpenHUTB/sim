import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision

import cornet

# 导入PIL的Image模块并忽略所有警告
from PIL import Image
Image.warnings.simplefilter('ignore')

# Numpy和PyTorch随机种子设置为0
np.random.seed(0)
torch.manual_seed(0)

# 启用CuDNN的自动寻找最优卷积算法的功能
torch.backends.cudnn.benchmark = True
# 定义一个归一化变换，使用ImageNet数据集的均值和标准差
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

# 创建参数解析器对象，用于处理命令行参数
parser = argparse.ArgumentParser(description='ImageNet Training')

# 必需参数：指定ImageNet数据集路径
# parser.add_argument('--data_path', required=True,
#                     help='包含train和val文件夹的ImageNet根目录路径')
parser.add_argument('--data_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')

# 输出路径参数（可选）
# parser.add_argument('-o', '--output_path', default=None,
#                     help='模型输出和日志的保存路径')
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')

# 模型选择参数（默认Z模型）
# parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S'], default='Z',
#                     help='选择要训练的CORnet模型变体：'
#                          'Z-基础版, R-循环版, RT-时间循环版, S-深度版')
parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S'], default='Z',
                    help='which model to train')

# R模型专用参数（时间步数）
# parser.add_argument('--times', default=5, type=int,
#                     help='CORnet-R模型运行的时间步数（仅对R模型有效）')
parser.add_argument('--times', default=5, type=int,
                    help='number of time steps to run the model (only R model)')

# 硬件配置参数
# parser.add_argument('--ngpus', default=0, type=int,
#                     help='使用的GPU数量（0表示使用CPU）')
# parser.add_argument('-j', '--workers', default=4, type=int,
#                     help='数据加载的并行进程数（建议设为CPU核心数）')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')

# 训练超参数
# parser.add_argument('--epochs', default=20, type=int,
#                     help='总训练轮次')
# parser.add_argument('--batch_size', default=256, type=int,
#                     help='每个批次的图像数量（需根据GPU显存调整）')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')

# 优化器参数
# parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
#                     help='初始学习率（典型值范围：0.01-0.1）')
# parser.add_argument('--step_size', default=10, type=int,
#                     help='学习率衰减周期（每N个epoch学习率下降10倍）')
# parser.add_argument('--momentum', default=.9, type=float,
#                     help='SGD动量参数（帮助加速收敛）')
# parser.add_argument('--weight_decay', default=1e-4, type=float,
#                     help='L2正则化系数（防止过拟合）')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')

# 解析命令行参数
FLAGS, FIRE_FLAGS = parser.parse_known_args()

# 函数目标：自动选择系统中可用的GPU，并根据显存情况选取最合适的几个
# n=1，默认使用1个GPU
def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    # 执行nvidia-smi命令获取GPU信息
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
    # 将命令输出转换为DataFrame
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    # 筛选总显存大于10GB的GPU（排除小显存设备）
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    # 处理预定义的可见GPU列表
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]  # 只在可见GPU中筛选
    # 按剩余显存降序排序
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    # 设置GPU编号与nvidia-smi显示一致
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    # 选择前n个GPU的索引
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


# 当用户指定使用GPU时调用GPU选择函数
if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)


def get_model(pretrained=False):
    # 确定模型加载位置：GPU环境无需指定，CPU需要映射到'cpu'
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    # 动态获取模型类（根据 - -model参数选择Z / R / RT / S）
    model = getattr(cornet, f'cornet_{FLAGS.model.lower()}')
    # 特殊处理循环版本R模型
    if FLAGS.model.lower() == 'r':
        model = model(pretrained=pretrained, map_location=map_location, times=FLAGS.times)
    else:
        model = model(pretrained=pretrained, map_location=map_location)

    # CPU模式处理DataParallel包装
    if FLAGS.ngpus == 0:
        model = model.module  # remove DataParallel# 解除并行封装（预训练模型可能保存为DataParallel格式）
    if FLAGS.ngpus > 0:
        model = model.cuda()  # 将模型移至GPU（自动使用set_gpus设置的可见设备）
    return model


def train(restore_path=None,  # useful when you want to restart training # 恢复训练时的检查点路径
          save_train_epochs=.1,  # how often save output during training # 每0.1个epoch保存一次训练指标
          save_val_epochs=.5,  # how often save output during validation # 每0.5个epoch进行一次验证
          save_model_epochs=5,  # how often save model weigths # 每5个epoch保存一次模型权重
          save_model_secs=60 * 10  # how often save model (in sec) # 每10分钟保存一次临时检查点
          ):

    model = get_model()  # 获取模型（根据FLAGS参数）
    trainer = ImageNetTrain(model)  # 训练器（包含优化器/损失函数）
    validator = ImageNetVal(model)  # 验证器

    start_epoch = 0  # 初始epoch
    # 恢复训练逻辑
    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']  # 恢复起始epoch
        model.load_state_dict(ckpt_data['state_dict'])  # 加载模型权重
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])  # 恢复优化器状态

    records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)  # 每个epoch的步数（总批次数）
    # 将epoch频率转换为全局步数
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in tqdm.trange(0, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan
        for step, data in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            data_load_time = time.time() - data_load_start  # 数据加载耗时统计
            global_step = epoch * len(trainer.data_loader) + step  # 计算全局步数

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()  # 执行验证并记录结果
                    trainer.model.train()  # 恢复训练模式

            if FLAGS.output_path is not None:
                records.append(results)  # 记录当前结果
                # 序列化保存到磁盘
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results.pkl'), 'wb'))

                # 模型检查点保存
                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                # 按时间间隔保存
                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()
                # 按步数间隔保存
                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           f'epoch_{epoch:02d}.pth.tar'))

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)  # 计算当前epoch进度
                record = trainer(frac_epoch, *data)  # 执行训练步骤（前向+反向传播）
                record['data_load_dur'] = data_load_time  # 记录数据加载耗时
                # 更新结果记录
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record  # 保存训练指标

            data_load_start = time.time()


# 用于从预训练的CORnet模型中提取指定层的特征
def test(layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    Suitable for small image sets. If you have thousands of images or it is
    taking too long to extract features, consider using
    `torchvision.datasets.ImageFolder`, using `ImageNetVal` as an example.

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """
    # 加载预训练模型（自动根据FLAGS.model选择架构）
    model = get_model(pretrained=True)
    # 定义图像预处理流程
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize, imsize)),  # 调整图像尺寸
                    torchvision.transforms.ToTensor(),  # 转为Tensor
                    normalize,  # ImageNet标准化
                ])
    # 设置模型为评估模式（关闭Dropout/BatchNorm）
    model.eval()

    # Hook函数：捕获指定层的输出
    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().numpy()  # 将GPU张量转CPU numpy数组
        # 展平特征（保留batch维度）
        _model_feats.append(np.reshape(output, (len(output), -1)))

    # 处理可能的DataParallel封装（多GPU情况）
    try:
        m = model.module  # 获取实际模型（解除并行封装）
    except:
        m = model  # 单GPU/CPU情况直接使用
    # 获取目标层对象
    model_layer = getattr(getattr(m, layer), sublayer)
    # 注册前向传播hook
    model_layer.register_forward_hook(_store_feats)

    # 特征收集主循环
    model_feats = []
    with torch.no_grad():  # 禁用梯度计算
        model_feats = []
        # 获取所有图像文件路径
        fnames = sorted(glob.glob(os.path.join(FLAGS.data_path, '*.*')))
        if len(fnames) == 0:
            raise FileNotFoundError(f'No files found in {FLAGS.data_path}')
        for fname in tqdm.tqdm(fnames):
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise FileNotFoundError(f'Unable to load {fname}')
            # 预处理图像
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            # 前向传播触发hook
            _model_feats = []
            model(im)
            # 保存指定时间步的特征（对CORnet-R等时序模型有效）
            model_feats.append(_model_feats[time_step])
        # 合并所有特征（shape: [num_images, feature_dim]）
        model_feats = np.concatenate(model_feats)

    # 保存特征文件
    if FLAGS.output_path is not None:
        fname = f'CORnet-{FLAGS.model}_{layer}_{sublayer}_feats.npy'
        np.save(os.path.join(FLAGS.output_path, fname), model_feats)

# 训练
class ImageNetTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()  # 创建数据加载器
        # 优化器配置
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         FLAGS.lr,
                                         momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        # 学习率调度器
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.step_size)
        # 损失函数
        self.loss = nn.CrossEntropyLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        # 数据集配置
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪
                torchvision.transforms.RandomHorizontalFlip(),  # 水平翻转
                torchvision.transforms.ToTensor(),  # 转为张量
                normalize,  # 标准化
            ]))
        # 数据加载器配置
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,  # 从FLAGS获取批次大小
                                                  shuffle=True,  # 每个epoch打乱数据
                                                  num_workers=FLAGS.workers,  # 并行加载进程数
                                                  pin_memory=True)  # 锁页内存加速传输
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        # 学习率调整（按epoch进度）
        self.lr.step(epoch=frac_epoch)
        if FLAGS.ngpus > 0:
            target = target.cuda(non_blocking=True)
        # 前向传播
        output = self.model(inp)

        record = {}
        # 损失计算
        loss = self.loss(output, target)
        # 记录指标
        record['loss'] = loss.item()
        record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        record['top1'] /= len(output)
        record['top5'] /= len(output)
        record['learning_rate'] = self.lr.get_lr()[0]

        # 反向传播与参数更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record

# 验证
class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'  # 验证器标识
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'val_in_folders'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),  # 缩放至256x256
                torchvision.transforms.CenterCrop(224),  # 中心裁剪224x224
                torchvision.transforms.ToTensor(),  # 张量化
                normalize,  # 标准化
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,  # 保持顺序一致
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()  # 切换评估模式（关闭Dropout等）
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}  # 初始化指标
        with torch.no_grad():  # 禁用梯度计算
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                # GPU数据传输（异步非阻塞）
                if FLAGS.ngpus > 0:
                    target = target.cuda(non_blocking=True)
                output = self.model(inp)

                # 累计损失（未平均）
                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                # 正确样本数累加
                record['top1'] += p1
                record['top5'] += p5

        # 计算平均指标（除以总样本数）
        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        # 计算每批次平均耗时
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record

# 计算分类任务中 Top-K 准确率
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():  # 禁用梯度计算以节省资源
        # 获取预测结果中概率最高的前max(topk)个类别
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        # 转置预测结果矩阵 [batch_size × K] → [K × batch_size]
        pred = pred.t()
        # 生成正确标签的布尔矩阵
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # 计算每个K值的正确预测数
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)


# =========================
# mymodel.py
# =========================

import torch
import torchvision.transforms as transforms
import numpy as np
import xarray as xr
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.data import StimulusSet
from brainscore_vision.tools.stimuli import transform_stimulus_set
import cornet_zz  # 注意：你的 cornet-zz.py 文件名要对应

class MyModel(BrainModel):
    def __init__(self):
        self._model = cornet_zz.CORnet_Z()
        self._model.eval()

        self._preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def visual_degrees(self) -> float:
        return 8.0  # CORnet默认8度视角

    def look_at(self, stimuli: StimulusSet) -> xr.DataArray:
        images = transform_stimulus_set(stimuli, transform=self._preprocess)
        outputs = []
        for image in images:
            with torch.no_grad():
                image = image.unsqueeze(0)
                output = self._model(image)
                output = output.view(output.size(0), -1)  # flatten成(batch_size, features)
                outputs.append(output.cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)
        return xr.DataArray(outputs, dims=["presentation", "neuroid"])

# =========================
# myregistry.py
# =========================

from brainscore_vision import registry
from mymodel import MyModel

@registry.register_model(identifier='cornetzz_custom', version=1)
def get_model():
    return MyModel()

