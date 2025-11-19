FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    wget tar zstd \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# 앱 코드 복사
COPY app/ ./app/

# 모델 다운로드 (빌드 시)
RUN wget https://github.com/BOB-DSPM/SAGE_Identity-AI/releases/download/v0.1.0/xlmr-large-min.tar.zst && \
    tar --zstd -xf xlmr-large-min.tar.zst && \
    rm xlmr-large-min.tar.zst

ENV MODEL_DIR=/app/xlmr-large-min
ENV DEVICE=cpu

EXPOSE 8900

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8900"]