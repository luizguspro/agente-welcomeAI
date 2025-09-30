FROM python:3.10-slim

WORKDIR /app

ENV MODEL_BASE_PATH=/models
ENV DATA_PATH=/data
ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY configs/ configs/

RUN mkdir -p /models /data /artifacts

ENTRYPOINT ["python", "-m", "ai_toolkit.preditor.pipeline.run"]