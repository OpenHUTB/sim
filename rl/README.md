# 基于数字孪生的自动驾驶强化学习仿真系统

## 说明
我是在windows系统上通过pycharm使用sac、ppo、dqp这三种强化学习算法来实现自动驾驶。
## Carla环境配置

a）Windows的话，工程文件夹carla里面有python3.7的wins的egg版本，可以通过在import carla之前加以下代码：

```

import os
import sys
import glob
try:
    sys.path.append(glob.glob('自己的绝对路径/carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('Couldn\'t import Carla egg properly')
```

来成功import carla。


## PPO 方法

Architectural layout encapsulating the three most essential components: 

1. CARLA Simulation. 
2. VAE. 
3. PPO Agent.


## 测试

此模型的预训练序列化文件放置在 `preTrained_models/PPO/<town>` 文件夹中。

配置好参数，train设置为False为测试，设置好对应的Town，设置算法模型，运行

```
python continuous_driver_xxx.py
```

或

```
python discrete_driver_xxx.py
```

也就是连续动作的模型或者离散动作的模型, xxx表示脚本的算法名字。

## 训练

1.如果训练新的Town，需要在preTrained_models/和checkpoints/文件夹里面手动新建：算法名/Town名的文件夹路径，可以参考已经有的。

2.配置好参数，train设置为True为测试，设置好对应的Town，设置算法模型，运行

```
python continuous_driver_xxx.py
```

或

```
python discrete_driver_xxx.py
```

也就是连续动作的模型或者离散动作的模型, xxx表示脚本的算法名字。

这将开始用默认参数训练代理，参数检查点将被写入`checkpoints/xxx/<town>/`，模型训练权重保存在`preTrained_models/xxx/<town>/`。如上所述，默认情况下我们在Town01上进行训练。

## resume训练
parameters.py里面MODEL_LOAD设置为True，运行

```
python continuous_driver_xxx.py
```

或

```
python discrete_driver_xxx.py
```
xxx表示脚本的算法名字。resume就是恢复上次最新记录的训练过程，可以防止电脑忽然断了，从头训练就浪费一些时间。


## To view the training progress/plots in the Tensorboard:

```
tensorboard --logdir runs/xxx
```

