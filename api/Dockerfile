# 1.确认基础镜像源
FROM python:3.12-slim

# 2.配置容器工作空间
WORKDIR /app

# 3.安装curl + docker-cli
# docker-cli用于运行stdio类MCP服务（例如command=docker的notion服务）
RUN apt-get update && apt-get install -y curl docker.io

# 4.安装v22版本的node.js
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# 5.安装uv, 使用uv来同步依赖环境
RUN pip install uv

# 6.拷贝requirements.txt并安装依赖
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# 7.拷贝项目文件(忽略开发环境的库)
COPY . .

# 8.为运行脚本配置权限
RUN chmod +x run.sh

# 9.设置Python路径变量
ENV PYTHONPATH=/app

# 10.暴露端口号
EXPOSE 8000

# 11.启动命令脚本
CMD ["./run.sh"]
