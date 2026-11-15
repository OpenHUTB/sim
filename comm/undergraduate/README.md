# 毕业论文：基于水下工作环境的传感器与通信链路仿真系统

## 编译说明
1. 主文件：`hutbthesis_main.tex`（只编译这个文件，不要编译其他 .tex）。
2. 使用 TeXstudio 选择 **XeLaTeX** 编译，连续编译 2 次；若目录、图表编号或参考文献跳转未更新，删除 `aux/toc/out` 等中间文件后再编译 2 次。
3. 公式统一使用 `equation` 环境；图片与图注放入 `figure` 环境，并添加 `\label`/`\ref` 交叉引用。
4. 表格使用 LaTeX 三线表；代码段使用 `listings` 环境。
5. 参考文献使用 `thebibliography` + `\bibitem`（见 `content/references.tex`），正文引用使用 `\cite{}`，不需要运行 biber；参考文献总数不得低于 40 篇。
6. 图片统一放在 `images/` 目录，优先使用 PPT/TikZ 绘制并导出 eps/pdf 矢量图；不要使用绝对路径。

## 目录说明
- `hutbthesis_main.tex`：论文主文件（编译入口）。
- `content/cover.tex`：题目、姓名、学号、导师等基本信息（带【】的为待填写占位）。
- `content/chapter1.tex` … `chapter7.tex`：正文章节。
- `content/abstractzh.tex` / `abstracten.tex`：中文、英文摘要。
- `content/references.tex`：参考文献条目。
- `images/`：论文插图。
