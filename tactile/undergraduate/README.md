编译说明：
1. 主文件：hutbthesis_main.tex。
2. 使用 TeXstudio 选择 XeLaTeX 编译，连续编译 2 次。
3. 公式已统一使用 equation 环境；图片与图注已合并到 figure 环境，并添加 \label/\ref 交叉引用。
4. 表格已改为 LaTeX 三线表；代码段使用 listings 环境。
5. 参考文献已改为 thebibliography + \bibitem，正文引用使用 \cite{}，不需要运行 biber。
6. 若目录、图表编号或参考文献跳转未更新，删除 aux/toc/out 后再用 XeLaTeX 编译 2 次。
