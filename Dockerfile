FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装uv
RUN pip install --upgrade pip && pip install uv

# 复制项目文件
COPY pyproject.toml .
COPY .env.example .env
COPY . .

# 用uv安装依赖
RUN uv pip install . --system --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["python", "main.py"]
