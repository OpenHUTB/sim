@echo off



rem 编译触觉本科毕业论文
pushd "%cd%\sim\tactile\undergraduate\"
xelatex -synctex=1 -interaction=nonstopmode hutbthesis_main.tex
popd