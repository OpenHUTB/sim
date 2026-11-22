# aerial
基于 OpenHUTB 无人机载体的航拍图像目标检测与定位系统相关论文模块

## 目录
- undergraduate：本科毕业论文LaTeX源码、图片、模板文件
  - `content/`：论文正文分章节tex文件
  - `images/`：论文用到的图片
  - `.gitignore`：git忽略配置
  - `cmdel.bat`：辅助清理脚本
  - `hutbthesis.cls`：学校论文模板类文件
  - `hutbthesis_main.tex`：论文主入口tex文件
  - `init_proj.m`：Matlab初始化脚本
  - `latexmkrcc`：latexmk配置文件
  - `README.md`：本工程内部说明文档
  - `reference.bib`：参考文献bib文件
  
- `clean.bat`：Windows一键清理编译临时文件

## 编译方式
1. 本地：使用LaTeXmk编译，推荐XeLaTeX
2. 远程：GitHub Actions自动编译，提交代码后自动生成PDF

## 注意
- 模板依赖 `hutbthesis.cls`，不要随意删除
- bat脚本仅Windows可用，用于清除aux、log等中间文件


## 编译
仓库配置 GitHub Actions，提交后自动编译LaTeX，输出PDF。
