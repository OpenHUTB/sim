from collections import OrderedDict
from torch import nn
from voneblock import VOneBlock
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CORblock_Z(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.output(x)
        return x

def get_gabor_params(n_channels=64, sf_range=(0.2, 0.4), theta_range=(0, np.pi),
                     aspect_ratio=0.7, phase_range=(0, np.pi), random_seed=0):
    rng = np.random.RandomState(random_seed)
    sf = rng.uniform(sf_range[0], sf_range[1], n_channels)
    theta = rng.uniform(theta_range[0], theta_range[1], n_channels)
    sigx = 1. / sf
    sigy = aspect_ratio * sigx
    phase = rng.uniform(phase_range[0], phase_range[1], n_channels)
    return sf, theta, sigx, sigy, phase

def CORnet_Z():
    # 生成 64 通道（32 simple + 32 complex）参数
    sf, theta, sigx, sigy, phase = get_gabor_params(n_channels=64)

    model = nn.Sequential(OrderedDict([
        ('V1', VOneBlock(sf, theta, sigx, sigy, phase, simple_channels=32, complex_channels=32)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', nn.Identity())
        ])))
    ]))

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model
