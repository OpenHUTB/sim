# 设置后面指令的基础镜像
FROM alpine:3.21

# 设置位于容器内部的工作目录
WORKDIR /usr/src

# 拷贝action所需要的源文件
COPY entrypoint.sh .

# 当docker容器启动时所要执行的代码文件（entrypoint.sh）
ENTRYPOINT ["/usr/src/entrypoint.sh"]
