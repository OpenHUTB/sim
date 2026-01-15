# 本科毕业设计

基于高保真模拟器的车辆和无人机的协同控制

## 功能特点

- 🎮 **实时键盘控制** - 支持W/A/S/D方向键 + 空格刹车
- 📊 **专业仪表盘** - 带转速表、档位指示器和速度显示  
- 🚗 **自动挡支持** - D/R/N档位自动切换
- 📈 **实时车辆状态监控** - 位置、速度、朝向实时更新
- 🎨 **可定制化控制参数** - 油门、转向灵敏度可调


## 使用

### 要求
- Python 3.8+
- AirSim 1.8.1+
- pygame 2.6.1+
- numpy 1.21.0+

### 初始化
```bash
# 克隆项目
git clone https://github.com/OpenHUTB/sim.git
cd sim/keyboard_control

# 安装依赖
pip install -r requirements.txt

# 确保AirSim正在运行
# 启动Blocks环境
cd ~/Blocks/LinuxNoEditor/Blocks/Binaries/Linux
./Blocks -opengl -nosound -windowed -ResX=800 -ResY=600