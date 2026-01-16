@echo off


rem 编译危险场景生成论文
pushd "%cd%\critical\undergraduate\"
"C:/software/texlive/texstudio\2023/bin/windows/xelatex.exe" -synctex=1 -interaction=nonstopmode "hutbthesis_main".tex
popd