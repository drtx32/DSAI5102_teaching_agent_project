FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    poppler-utils \  #PDF处理
    tesseract-ocr \  #OCR支持
    && rm -rf /var/lib/apt/lists/*

# 安装uv
RUN pip install --upgrade pip && pip install uv

# 复制项目文件
COPY pyproject.toml .
COPY .env.example .env
COPY . .

# 用uv安装依赖
RUN uv pip install .

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
