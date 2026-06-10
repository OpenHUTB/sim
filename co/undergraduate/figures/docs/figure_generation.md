\# 图片生成与来源说明（figure\_generation.md）



\## 1. 文档说明



本文件用于说明论文中各类图片的来源、生成方式以及复现方法。



论文中的图片按照来源分为以下四类：



1\. Python代码生成图；

2\. MATLAB数据分析图；

3\. SolidWorks建模与装配截图；

4\. Gazebo仿真运行截图。



其中，代码生成图提供生成脚本与生成方法；工程设计图和实验结果图提供软件来源与获取方式说明。



\---



\# 2. Python代码生成图



\## 2.1 系统分层架构图



图名：系统分层架构图



生成方式：



Python + Matplotlib



源文件：



```text

figures/src/system\_architecture.py

```



输出文件：



```text

figures/system\_architecture.png

figures/system\_architecture.pdf

figures/system\_architecture.svg

```



生成命令：



```bash

python figures/src/system\_architecture.py

```



说明：



该图根据论文中的自动对接控制系统结构设计绘制，包含飞机辅助感知层、任务与模式管理层、局部轨迹生成层、误差反馈控制层、4WS映射与约束处理层、Gazebo执行层以及状态反馈闭环。



\---



\## 2.2 自动对接控制流程图



图名：自动对接控制流程图



生成方式：



Python + Matplotlib



源文件：



```text

figures/src/control\_flowchart.py

```



输出文件：



```text

figures/control\_flowchart.png

figures/control\_flowchart.pdf

figures/control\_flowchart.svg

```



生成命令：



```bash

python figures/src/control\_flowchart.py

```



说明：



该图根据自动对接任务控制逻辑绘制，描述从目标识别、轨迹生成、误差计算、控制映射到停车判定的完整控制流程。



\---



\# 3. SolidWorks模型与装配图



对应图片：



```text

图3-1 ～ 图3-8

（牵引车模型、飞机模型、装配体模型、运动仿真模型等）

```



来源软件：



```text

SolidWorks 2025

```



生成方式：



1\. 使用 SolidWorks 建立零部件模型；

2\. 建立装配体约束关系；

3\. 完成运动仿真配置；

4\. 在建模环境或运动仿真环境中导出截图；

5\. 保存为 PNG 图片用于论文插图。



源文件类型：



```text

\*.SLDPRT

\*.SLDASM

```



说明：



该类图片属于CAD建模与结构设计结果展示，不属于代码生成图。



\---



\# 4. Gazebo仿真运行截图



对应图片：



```text

图4-1 ～ 图4-16

（机场场景、车辆运行、自动对接过程、控制结果等）

```



来源软件：



```text

Ubuntu 16.04

ROS Melodic

Gazebo 9.19.0

```



生成方式：



1\. 启动机场仿真环境；

2\. 加载飞机与牵引车模型；

3\. 运行自动对接控制程序；

4\. 在实验过程中截取Gazebo运行画面；

5\. 导出PNG图片作为实验结果展示。



说明：



该类图片属于实验结果截图，不属于代码生成图。



\---



\# 5. MATLAB数据分析图



对应图片：



```text

图4-17 ～ 图4-20

```



主要内容：



```text

性能对比柱状图

误差收敛曲线

控制效果曲线

综合性能雷达图

```



生成工具：



```text

MATLAB R2023a

```



生成脚本：



```text

figures/MATLAB性能对比.py

```



生成命令：



```bash

python MATLAB性能对比.py

```



输出文件：



```text

image17.png

image18.png

image19.png

image20.png

```



说明：



根据实验数据自动生成性能分析图，用于比较不同控制方法的效率、误差收敛特性和综合性能表现。



\---



\# 6. 图片编译说明



LaTeX论文中通过 `\\includegraphics{}` 调用图片文件。



推荐优先使用 PDF 格式图片：



```latex

\\begin{figure}\[htbp]

&#x20;   \\centering

&#x20;   \\includegraphics\[width=0.85\\textwidth]{figures/system\_architecture.pdf}

&#x20;   \\caption{系统分层架构图}

&#x20;   \\label{fig:system\_architecture}

\\end{figure}

```



其中：



\* `.tex` 文件参与论文编译；

\* `.pdf/.png/.svg` 为图片文件；

\* `.py` 为图片生成脚本；

\* `.md` 为图片来源说明文件，不参与LaTeX编译。



\---



\# 7. 总结



论文中的图片来源如下：



| 图片类型          | 数量 | 来源                  |

| ------------- | -- | ------------------- |

| Python生成流程图   | 2  | Python + Matplotlib |

| MATLAB数据分析图   | 4  | MATLAB脚本生成          |

| SolidWorks建模图 | 若干 | SolidWorks截图        |

| Gazebo实验结果图   | 若干 | Gazebo运行截图          |



其中所有代码生成图片均提供生成脚本；工程设计图与实验结果图均提供软件来源和获取方式说明，满足论文工程文件可追溯和可复现要求。



