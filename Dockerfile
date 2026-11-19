
# Python 3.12 Linux 基礎映像
FROM python:3.12-slim

# Python 與 TensorFlow 執行環境
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# 容器內的專案目錄
WORKDIR /app

# 安裝 TensorFlow GPU 與 FastAPI 需要的套件
RUN python -m pip install --upgrade pip && \
    python -m pip install \
        "tensorflow[and-cuda]==2.21.0" \
        fastapi \
        "uvicorn[standard]" \
        python-multipart

# 只複製推論服務和模型
COPY app/ /app/app/
COPY models/ /app/models/

# API 服務連接埠
EXPOSE 8000

# 單一 Worker，避免重複載入模型佔用 GPU 記憶體

# 啟動 Python 前加入 NVIDIA CUDA 函式庫路徑
CMD ["sh", "-c", "export LD_LIBRARY_PATH=\"$(find /usr/local/lib/python3.12/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib | paste -sd: -)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"; exec python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1"]

