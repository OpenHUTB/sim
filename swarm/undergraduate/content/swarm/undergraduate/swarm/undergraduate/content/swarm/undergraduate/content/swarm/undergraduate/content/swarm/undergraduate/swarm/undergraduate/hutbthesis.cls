\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hutbthesis}[2025/01/01 HUTB Thesis Template]

\LoadClass[12pt,a4paper]{article}

% 中文字体设置
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setCJKmainfont{SimSun}

% 页面设置
\usepackage[top=2.5cm,bottom=2.5cm,left=3cm,right=2.5cm]{geometry}

% 定义命令
\newcommand{\titlecn}[1]{\def\@titlecn{#1}}
\newcommand{\titleen}[1]{\def\@titleen{#1}}
\newcommand{\author}[1]{\def\@author{#1}}
\newcommand{\studentid}[1]{\def\@studentid{#1}}
\newcommand{\college}[1]{\def\@college{#1}}
\newcommand{\major}[1]{\def\@major{#1}}
\newcommand{\class}[1]{\def\@class{#1}}
\newcommand{\supervisor}[1]{\def\@supervisor{#1}}

% 封面
\newcommand{\maketitle}{
    \begin{titlepage}
        \centering
        \vspace*{2cm}
        {\zihao{1} \heiti 湖南工商大学\\}
        \vspace{1cm}
        {\zihao{2} \heiti 本科毕业设计（论文）\\}
        \vspace{3cm}
        {\zihao{2} \heiti \@titlecn\\}
        \vspace{1cm}
        {\zihao{3} \heiti \@titleen\\}
        \vspace{3cm}
        {\zihao{4} \songti
        学\hspace{2em}院：\@college\\
        专\hspace{2em}业：\@major\\
        班\hspace{2em}级：\@class\\
        学\hspace{2em}号：\@studentid\\
        姓\hspace{2em}名：\@author\\
        指导教师：\@supervisor\\}
        \vspace{2cm}
        {\zihao{4} \@date}
    \end{titlepage}
}

% 章节设置
\usepackage{titlesec}
\titleformat{\chapter}{\centering\zihao{3}\heiti}{第\chinese{chapter}章}{1em}{}

% 摘要环境
\newenvironment{abstractcn}{
    \chapter*{摘要}
    \addcontentsline{toc}{chapter}{摘要}
}{}
\newenvironment{abstracten}{
    \chapter*{ABSTRACT}
    \addcontentsline{toc}{chapter}{ABSTRACT}
}{}
\newcommand{\keywords}[1]{\par\vspace{1em}\noindent\textbf{关键词：}#1}

% 致谢环境
\newenvironment{acknowledgements}{
    \chapter*{致谢}
    \addcontentsline{toc}{chapter}{致谢}
}{}

% 参考文献
\usepackage[backend=biber,style=gb7714-2015]{biblatex}

\endinput
