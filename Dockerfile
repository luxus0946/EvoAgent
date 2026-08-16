"""EvoAgent Docker 镜像：FastAPI + Gradio 双服务。

构建: docker build -t evoagent .
运行 API:  docker run -p 8000:8000 evoagent api
运行 UI:  docker run -p 7860:7860 evoagent ui
"""

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evoagent/ evoagent/
COPY app/ app/
COPY experiments/ experiments/
COPY config.py config.py 2>/dev/null || true

EXPOSE 8000 7860

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]