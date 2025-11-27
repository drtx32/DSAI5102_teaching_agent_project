#!/usr/bin/env bash
echo "开始构建并运行教学代理容器（Ollama容器需另外启动）..."
# 构建镜像
docker build -t teaching-agent:latest .

# 停掉并删除已存在的同名容器（如果有），以避免端口冲突
docker rm -f teaching-agent 2>/dev/null || true

# 后台运行容器，命名为 teaching-agent，并映射主机 8501 到容器 8501
docker run -d --name teaching-agent -p 8501:8501 teaching-agent:latest

echo "容器已在后台运行：名称 'teaching-agent'。"
echo "查看日志：  docker logs -f teaching-agent"
echo "停止并删除容器： docker rm -f teaching-agent"