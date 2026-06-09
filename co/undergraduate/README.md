编译说明：
1. 主文件：hutbthesis_main.tex。
2. 使用 TeXstudio 选择 XeLaTeX 编译，连续编译 2 次即可更新目录、交叉引用、图表编号和参考文献跳转。
3. 图片和表格已添加 \label / \ref 交叉引用；正文文献引用已改为 \cite{}；参考文献使用 thebibliography + \bibitem，不需要运行 biber。
4. 表格已改成 LaTeX 三线表；代码段使用 listings 环境。
5. 字体按新发 Word 原文恢复：中文主要为宋体/黑体，英文为 Times New Roman；如果在 Windows/TeXstudio 下编译，会优先调用 SimSun/SimHei/Times New Roman。
6. 若修改后引用显示为 ??，删除 aux/toc/out 后重新 XeLaTeX 编译两次。
