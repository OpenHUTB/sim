# 图片生成与来源说明（figure_generation.md）

##  文档说明

本文件用于说明论文《基于模拟器的极端驾驶仿真场景的生成算法研究》中各类图片的来源、生成方式以及复现方法。

论文中的图片按照来源分为以下四类：

1. 流程图与示意图（Draw.io / PPT 绘制）；
2. MATLAB 数据分析图；
3. Python 训练曲线与算法示意图；
4. CARLA 仿真运行截图。

其中，代码生成图提供生成脚本与生成方法；仿真截图提供软件来源与获取方式说明。

---

## 1. 流程图与示意图

### 论文结构框架图

对应图片：

```text
图1 论文结构框架图
```

源文件：

```text
images/figure_1.png
```

生成方式：

1. 使用 Draw.io / Microsoft PowerPoint 绘制原始流程图；
2. 导出为 PNG 图片；
3. 使用 potrace 将 PNG 转为 SVG，再通过 svglib 转为矢量 PDF，确保放大不失真。

说明：

该图展示论文整体研究框架与技术路线，属于结构示意图，不属于代码生成图。

---

### 算法流程图与网络结构图

对应图片：

```text
图3  PPO网络结构图       （figure_3.png）
图4  算法流程图           （figure_4.png）
图19  .xosc生成流程图      （figure_19.png）
```

源文件：

```text
images/figure_3.png
images/figure_4.png
images/figure_19.png
```

生成方式：

使用 Draw.io / Microsoft PowerPoint 手动绘制，导出为 PNG 图片。

说明：

该类图片属于算法结构与流程示意图，不属于代码生成图。推荐保留原始 `.drawio` 或 `.pptx` 源文件以备后续修改。

---

## 2. MATLAB 数据分析图

### 训练结果对比图（四种场景）

对应图片：

```text
图5  暴雨天跟车场景训练结果对比      （figure_5.png  → figure_5_rainy_following.pdf）
图6  前车紧急制动场景训练结果对比    （figure_6.png  → figure_6_emergency_braking.pdf）
图7  行人横穿马路场景训练结果对比    （figure_7.png  → figure_7_pedestrian_crossing.pdf）
图8  雾天鬼探头场景训练结果对比      （figure_8.png  → figure_8_foggy_overtake.pdf）
```

生成工具：

```text
MATLAB（R2020a 及以上版本）
```

生成脚本：

```text
src/plot_figure5_rainy_following.m
src/plot_figure6_emergency_braking.m
src/plot_figure7_pedestrian_crossing.m
src/plot_figure8_foggy_overtake.m
```

说明：

每张图包含左右两个子图：左图为训练奖励曲线（Attention-DQN 蓝色 vs Smooth-PPO 橙色），右图为评估指标柱状图（碰撞率、安全完成率等五项指标）。奖励曲线数据从原始 PNG 图像素级提取并映射到 0~100 数值范围。脚本使用 `exportgraphics` 输出矢量 PDF，确保放大不模糊。


## 3. CARLA 仿真运行截图

对应图片：

```text
图9 ～ 图18  十种极端驾驶场景仿真截图
（figure_9.png ～ figure_18.png）
```

来源软件：

```text
CARLA 0.9.16 (hutb 定制版)
Windows 
Python API
```

生成方式：

1. 启动 CARLA 仿真服务器；
2. 加载指定地图与天气设置；
3. 运行训练好的 DQN / PPO 模型进行场景测试；
4. 在仿真过程中通过 CARLA Python API 或手动截取运行画面；
5. 保存为 PNG 图片用于论文插图。

场景列表：

| 图片文件 | 场景名称 | 天气/地图设置 |
|----------|----------|---------------|
| figure_9.png | [暴雨跟车] | [暴雨/Town10hd] |
| figure_10.png | [大雾跟车] | [大雾/Town10hd] |
| figure_11.png | [夜间行车] | [夜晚/Town10hd] |
| figure_12.png | [前车急刹] | [晴天/Town10hd] |
| figure_13.png | [旁车加塞] | [晴天/Town10hd] |
| figure_14.png | [行人横穿马路] | [晴天/Town10hd] |
| figure_15.png | [行人闯红灯] | [晴天/Town10hd] |
| figure_16.png | [鬼探头] | [晴天/Town10hd] |
| figure_17.png | [夜间行人横穿] | [夜晚/Town10hd] |
| figure_18.png | [雾天鬼探头] | [大雾/Town10hd] |

说明：

该类图片属于仿真实验结果截图，不属于代码生成图。建议保留截图时的 CARLA 版本、地图名称、天气参数等配置信息，以便后续复现。

---

##  总结

论文中的图片来源如下：

| 图片类型 | 数量 | 来源 |
|----------|------|------|
| 流程图与示意图 | 4~5 张 | Draw.io / PPT 手绘 |
| MATLAB 数据分析图 | 9 张 | MATLAB 脚本生成 |
| CARLA 仿真截图 | 10 张 | CARLA 运行截图 |
| 其他 | 若干 | [需补充] |

其中所有 MATLAB 生成图均提供 `.m` 脚本，可复现；流程图与仿真截图均提供软件来源和获取方式说明，满足论文工程文件可追溯和可复现要求。