docker build -t teaching-agent:latest .
docker run --rm -it -p 8501:8501 teaching-agent:latest