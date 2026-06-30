# 1.使用ubuntu基础镜像来源
FROM ubuntu:22.04

# 2.安装过程避免交互式提示并设置主机名
ENV DEBIAN_FRONTEND=noninteractive
ENV HOSTNAME=sandbox

# 3.更新和安装基础软件后并移除依赖
RUN apt-get update && apt-get install -y \
    sudo bc curl wget gnupg software-properties-common supervisor \
    xterm socat xvfb x11vnc websockify\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5.创建用户ubuntu并赋予sudo权限（C5d-3: pin uid:gid 1000:1000 so run_as_user is repo-owned）
RUN groupadd -g 1000 ubuntu && \
    useradd -u 1000 -g 1000 -m -d /home/ubuntu -s /bin/bash ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu

# 5a.C5d-4: pre-create the memory :ro bind mountpoint so it composes under a read-only rootfs
# (runc cannot create a missing bind-mount target dir on a read-only rootfs). Flag-independent +
# inert when read_only is OFF (the dir simply pre-exists; the writable-rootfs path is unaffected).
# No chown — a :ro bind needs no ownership change; the dir is a mountpoint, not written.
RUN mkdir -p /workspace/.memory

# 6.安装Python 3.10版本(非uv)
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 7.安装node.js 24.12.0版本(LTS)
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 8.安装Google Chrome浏览器
RUN add-apt-repository ppa:xtradeb/apps -y && \
    apt-get update && \
    apt-get install -y chromium --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 9.安装中文字体和语言支持
RUN apt-get update && apt-get install -y \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    language-pack-zh-hans \
    locales \
    && locale-gen zh_CN.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 10.设置默认编码
ENV LANG=zh_CN.UTF-8 \
    LANGUAGE=zh_CN:zh \
    LC_ALL=zh_CN.UTF-8

# 11.设置工作空间
WORKDIR /sandbox

# 额外的 Python 包索引，默认使用国内镜像以降低构建阶段超时概率
ARG PYTHON_PACKAGE_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PYTHON_PACKAGE_INDEX_URL}

# 12.拷贝项目依赖文件到工作空间并安装依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 12a.安装 PDF 处理依赖
RUN pip install --no-cache-dir pymupdf4llm

# 12b.安装音频/视频处理依赖
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# 12c.可选：sandbox 本地 Whisper 转录（仅当 audio.provider=sandbox_whisper 时需要）
# 通过 build arg 控制，避免默认镜像膨胀 ~500MB
# 安装 faster-whisper 并预下载 base 模型，避免运行时首次下载导致超时
ARG INSTALL_WHISPER=false
RUN if [ "$INSTALL_WHISPER" = "true" ]; then \
      pip install --no-cache-dir faster-whisper && \
      python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', compute_type='int8')"; \
    fi

# 13.拷贝当前整个项目文件
COPY . .

# 14.将supervisord.conf拷贝至supervisor服务配置目录中
COPY supervisord.conf /etc/supervisor/conf.d/sandbox.conf

# 15.暴露端口: 8080(FastAPI) 9222(CDP) 5900(vnc)
EXPOSE 8080 9222 5900 5901

# 16.额外的环境变量配置
ENV CHROME_ARGS=""
ENV UVI_ARGS=""

# 17.使用supervisor管理所有进程
CMD ["supervisord", "-n", "-c", "/sandbox/supervisord.conf"]
