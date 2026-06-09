# 基于高保真模拟器的车辆与飞行器协同控制

## 项目简介

本项目基于 ROS Melodic 与 Gazebo 9 构建机场地面车辆与飞行器协同控制仿真平台，实现牵引车辆对飞机目标的自动识别、自动接近、自动对接以及协同控制功能。

系统融合激光雷达、深度相机等多种感知信息，通过路径规划与运动控制算法实现车辆对目标飞机的自主接近与对接验证，并在 Gazebo 高保真仿真环境中完成算法测试与性能评估。

本项目为湖南工商大学机器人工程专业本科毕业设计：

> 《基于高保真模拟器的车辆与飞行器协同控制》

---

## 功能特点

* 飞机目标识别与身份确认
* 自动接近控制
* 自动对接控制
* 障碍物安全检测
* 四轮转向车辆控制
* 多传感器联合感知
* Gazebo高保真仿真验证
* ROS模块化系统架构

---

## 系统架构

```text
ROS
│
├── 感知层
│   ├── 激光雷达
│   ├── 深度相机
│   └── 飞机辅助识别标识
│
├── 决策层
│   ├── 身份确认
│   ├── 目标定位
│   ├── 轨迹规划
│   └── 对接决策
│
├── 控制层
│   ├── 速度控制
│   ├── 转向控制
│   └── 四轮转向控制
│
└── Gazebo仿真平台
```

---

## 环境要求

推荐使用 Ubuntu 16.04 LTS。

### 软件环境

| 软件           | 版本         |
| ------------ | ---------- |
| Ubuntu       | 16.04 LTS  |
| ROS          | Melodic    |
| Gazebo       | 9.19.0     |
| Python       | 2.7 / 3.8+ |
| RViz         | Melodic    |
| catkin_tools | latest     |

---

### ROS依赖安装

```bash
sudo apt update

sudo apt install \
ros-melodic-gazebo-ros \
ros-melodic-gazebo-plugins \
ros-melodic-gazebo-ros-control \
ros-melodic-controller-manager \
ros-melodic-joint-state-controller \
ros-melodic-effort-controllers \
ros-melodic-joint-state-publisher \
ros-melodic-robot-state-publisher \
ros-melodic-xacro
```

---

### Python依赖

```bash
pip install numpy matplotlib
```

---

### 推荐硬件配置

| 项目   | 配置            |
| ---- | ------------- |
| CPU  | Intel i5 8代以上 |
| 内存   | ≥ 8 GB        |
| 显卡   | 支持 OpenGL 4.0 |
| 磁盘空间 | ≥ 10 GB       |

---

## 初始化

### 1. 创建工作空间

```bash
mkdir -p ~/tug_ws/src

cd ~/tug_ws/src
```

---

### 2. 克隆项目

```bash
git clone https://github.com/OpenHUTB/sim.git
```

---

### 3. 编译工程

```bash
cd ~/tug_ws

catkin_make
```

---

### 4. 加载环境变量

```bash
source devel/setup.bash
```

建议加入系统启动项：

```bash
echo "source ~/tug_ws/devel/setup.bash" >> ~/.bashrc

source ~/.bashrc
```

---

### 5. 配置Gazebo模型路径

```bash
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/tug_ws/src/tug_gazebo/models' >> ~/.bashrc

source ~/.bashrc
```

---

### 6. 检查ROS环境

启动ROS Master：

```bash
roscore
```

新终端执行：

```bash
rostopic list
```

若系统正常运行则表示ROS环境配置成功。

---

## 项目结构

```text
tug_ws
├── src
│
├── zongzhuang
│   ├── urdf
│   ├── launch
│   ├── config
│   ├── meshes
│   └── scripts
│
├── tug_gazebo
│   ├── worlds
│   ├── models
│   └── launch
│
├── tug_description
│
├── build
├── devel
└── logs
```

---

## 启动仿真

### 启动Gazebo

```bash
roslaunch zongzhuang gazebo.launch
```

---

### 查看控制器状态

```bash
rosservice call /zongzhuang/controller_manager/list_controllers
```

正常情况下应显示：

```text
joint_state_controller

front_steer_position_controller

back_steer_position_controller

front_left_wheel_velocity_controller

front_right_wheel_velocity_controller

back_left_wheel_velocity_controller

back_right_wheel_velocity_controller
```

并全部处于：

```text
running
```

状态。

---

## 测试程序

### 自动牵引测试

```bash
rosrun zongzhuang test_1_1_0.py
```

测试内容包括：

1. 车辆回中校准
2. 前进测试
3. 后退测试
4. 前桥转向测试
5. 双桥小半径转弯测试
6. 蟹行运动测试
7. 自动回中

---

## 常用命令

### 查看ROS节点

```bash
rosnode list
```

### 查看ROS话题

```bash
rostopic list
```

### 查看控制器

```bash
rosservice call /zongzhuang/controller_manager/list_controllers
```

### 重置Gazebo世界

```bash
rosservice call /gazebo/reset_world
```

### 获取车辆状态

```bash
rostopic echo /gazebo/model_states
```

---

## 实验结果

项目完成了以下功能验证：

* 自动目标识别
* 自动接近控制
* 自动对接控制
* 障碍物检测
* 四轮转向控制
* 高保真联合仿真验证

实验结果表明系统能够稳定完成车辆与飞机的协同控制任务。

---

## 作者

雷宇杰

湖南工商大学

机器人2201班

---

## 致谢

感谢湖南工商大学相关指导教师在课题研究过程中给予的指导与帮助。

感谢ROS、Gazebo等开源社区提供的软件平台支持。
