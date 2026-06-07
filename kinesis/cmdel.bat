@echo off
REM 清理 LaTeX 编译产生的临时文件，不删除 tex、cls、bib、图片和最终 PDF。
del /a /f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.toc *.xdv *.synctex.gz 2>nul
del /a /f content\*.aux 2>nul
