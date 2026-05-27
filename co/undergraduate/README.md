编译说明

1. 使用 TeXstudio 打开 hutbthesis_main.tex。
2. 编译器选择 XeLaTeX。
3. 正常情况下连续编译 2~3 次即可生成目录、图表交叉引用和参考文献引用。
4. 本工程参考文献采用 content/references.tex 中的 thebibliography 环境，正文使用 \cite{} 引用，不需要运行 BibTeX 或 biber。
5. 若提交源文件，请在最终编译成功并确认 PDF 无误后运行 cmdel.bat，删除 .aux、.log、.toc、.out 等临时文件，再进行打包。
