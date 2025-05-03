from collections import OrderedDict
from torch import nn

HASH = 'cornet_se'

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):  # 减小reduction比例
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        # 新增初始化方法
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_Z(nn.Module):
    # +++ 修改初始化函数 +++
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_se=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        
        # +++ 添加SE模块 +++
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)    # 先激活
        x = self.se(x)        # 后接SE
        x = self.pool(x)
        x = self.output(x)
        return x


def CORnet_Z():
    model = nn.Sequential(OrderedDict([
        # +++ 为每个模块启用SE +++
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2, use_se=True)),
        ('V2', CORblock_Z(64, 128, use_se=True)),
        ('V4', CORblock_Z(128, 256, use_se=True)),
        ('IT', CORblock_Z(256, 512, use_se=True)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # +++ 修改初始化部分 +++
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        # +++ 新增SE模块的初始化 +++
        elif isinstance(m, SEBlock):
            nn.init.kaiming_normal_(m.fc[0].weight, mode='fan_out')
            nn.init.constant_(m.fc[0].bias, 0)
            nn.init.normal_(m.fc[2].weight, mean=0, std=0.01)
            nn.init.constant_(m.fc[2].bias, 0)

    return model