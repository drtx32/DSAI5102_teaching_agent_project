FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 使用国内PyPI镜像（可选）
ENV PIP_DEFAULT_TIMEOUT=120
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 切换 apt 源为清华镜像以加速 apt 下载
RUN rm -f /etc/apt/sources.list.d/* || true && \
    cat > /etc/apt/sources.list <<'EOF'
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ trixie main contrib non-free
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ trixie main contrib non-free

deb https://mirrors.tuna.tsinghua.edu.cn/debian-security trixie-security main contrib non-free

deb https://mirrors.tuna.tsinghua.edu.cn/debian/ trixie-updates main contrib non-free
EOF

# 安装系统依赖（编译 wheel 所需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 更新 pip、setuptools、wheel
RUN pip install --upgrade pip setuptools wheel

# 先拷贝元信息以利用缓存（仅当 pyproject.toml 未变时可复用缓存）
COPY pyproject.toml .
# 若有 requirements.txt 或 setup.cfg，可一并拷贝以进一步缓存
# COPY requirements.txt .

# 复制项目其它文件
COPY . .

# 安装依赖（可改为 pip install -r requirements.txt）
RUN pip install --no-cache-dir -e .

# 暴露端口（Streamlit 默认 8501）
EXPOSE 8501

# 启动命令
CMD ["python", "main.py"]
