"""EvoAgent Docker 镜像：FastAPI + Gradio 双服务。

构建: docker build -t evoagent .
运行 API:  docker run -p 8000:8000 evoagent api
运行 UI:  docker run -p 7860:7860 evoagent ui
"""

FROM python:3.12-slim

WORKDIR /app

# 国内网络下用镜像源加速依赖安装（构建时可用 --build-arg 覆盖）
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirements.txt .
RUN pip install --no-cache-dir -i $PIP_INDEX_URL -r requirements.txt

COPY evoagent/ evoagent/
COPY app/ app/
COPY experiments/ experiments/
COPY config.py ./

EXPOSE 8000 7860

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]