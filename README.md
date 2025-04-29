# 仿真平台
实现车辆和行人代理的仿真。

## 环境配置
支持和测试的平台包括：Windows 10 和 Ubuntu 20.04。
1. 下载并安装 Python 3.7、Carla 0.9.15、latex 2023、Texstudio 4.6.4、Git 2.42.0（Windows可使用`TortoiseGit 2.15.0.0`作为图形界面进行代码提交）。
2. 使用`git clone https://github.com/OpenHUTB/sim.git` 进行仓库的递归克隆。
3. 使用latex编译`{MODULE_NAME}/undergraduate/hutbthesis_main.tex`（而不是其他.tex文件）生成PDF文件。

## 注意事项
代码目录结构
```dtd
{MODULE_NAME}/
    undergraduate/
        fig/
            carla.tex
            carla.pptx
        hutbthesis_main.tex
    README.md
data/
utils/
    generate_img.py
    test/
requirements.txt
launch.py
config.ini
README.md
```
1. 每个模块的使用和说明文档以`*.md`格式更新到 [文档仓库](https://github.com/OpenHUTB/carla_doc) `docs/{MODULE_NAME}`目录下。
2. 仓库中只保留`*.md`、`*.py`、`*.cpp`、`*.conf`、`*.sh`、`*.bat`等文件，最多保留一个演示数据，其他数据都保留到百度网盘并提供链接和地址到 [`README.md`](https://github.com/OpenHUTB/sim/blob/master/README.md) 文件中。
3. 每次提交统一用 Pull Request 的方式进行，至少需要一个人的进行 Code Review 和测试才能合并到主分支。
4. 每个项目主要代码都放在一个目录下，保证每次更新时整个项目都能运行（写功能代码前先写测试代码和样例，包括功能测试、性能测试等），每个模块只有一个 Python 脚本作为入口程序。
5. 作图统一使用 TikZ 或 PPT 绘图，放置在目录`{MODULE_NAME}/undergraduate/fig/`目录下，使用`generate_img.py`脚本生成相应的`*.eps`格式作为统一的图片格式（不需要上传）。
6. 代码中不要使用绝对路径，统一使用相对路径。
7. 所有模块的配置统一放到`sim.config`文件中。
8. Python的依赖放在`requirements.txt`中，并指定版本，可以将项目的配置文件放置在文件 [`config.ini`](https://github.com/OpenHUTB/sim/blob/master/config.ini) 中。
9. 页面设计的风格尽量统一，依赖的包尽可能少。
10. 软件依赖、输入数据、中间文件、输出结果等都放在`data/`目录下。
11. 界面统一用pygame进行显示和交互，起始程序参考[手动控制示例](https://github.com/OpenHUTB/carla_doc/blob/master/src/examples/manual_control.py) 。


## 参考链接

- [中文文档](https://openhutb.github.io/carla_doc/)
- [毕业论文模板](https://github.com/OpenHUTB/undergraduate)
- [pygame示例](https://github.com/guliang21/pygame) 、[超级玛丽](https://github.com/mx0c/super-mario-python) 等
