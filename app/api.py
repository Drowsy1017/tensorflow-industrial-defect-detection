
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import os
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .inference import DefectPredictor


MAX_FILE_SIZE = 10 * 1024 * 1024


# =====================================
# 1. 服務啟動時載入模型
# =====================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.predictor = DefectPredictor()

    print("Inference API ready.")

    yield

    print("Inference API shutting down.")


# =====================================
# 2. 建立 FastAPI
# =====================================

app = FastAPI(
    title="Industrial Defect Detection API",
    description="MobileNetV2 industrial defect classifier",
    version="1.0.0",
    lifespan=lifespan
)


# =====================================
# 3. Health Check
# =====================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": hasattr(
            app.state, "predictor"
        )
    }


# =====================================
# 4. 圖片推論 API
# =====================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    filename = Path(file.filename or "").name

    suffix = Path(filename).suffix.lower()

    if suffix not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG files are supported"
        )

    if file.content_type not in (
        "image/png",
        "image/jpeg"
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid image content type"
        )

    # 限制最大檔案大小
    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 10 MB limit"
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )

    temp_path = None

    try:

        # 建立暫存檔供既有 Predictor 使用
        with NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:

            temp_file.write(content)
            temp_path = temp_file.name

        predictor = app.state.predictor

        # 避免同步 TensorFlow 推論阻塞 Event Loop
        result = await run_in_threadpool(
            predictor.predict,
            temp_path
        )

        # 將暫存檔名稱換成原始檔名
        result["filename"] = filename

        return result

    except (
    ValueError,
    OSError,
    RuntimeError,
    tf.errors.InvalidArgumentError
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image"
        ) from exc

    finally:

        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

        await file.close()
