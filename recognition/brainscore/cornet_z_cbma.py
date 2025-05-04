from collections import OrderedDict
from torch import nn
from cbam import CBAM

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CORblock_Z(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_cbam=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = self.output(x)
        return x

def CORnet_Z():
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(3, 64)),                      # 原始结构
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512, use_cbam=True)),    # ✅ CBAM 仅用于 IT 层
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
