# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import time
import torch

from config.inference_config import InferenceConfig
from utils.utils import process_single_image, draw_detections_on_image

app = FastAPI(
    title="YOLO Classroom People Counter API - Ensemble",
    description=(
        "Upload ảnh lớp học → chạy ensemble inference "
        "(yolo26x gốc + flip + yolo12x CLAHE + WBF) "
        "→ trả về ảnh có bounding box, tâm và confidence score"
    ),
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ────────────────────────────────────────────────
# Xác định device một lần
# ────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else "cpu"
print(f"API khởi động - sử dụng device: {device}")

# ────────────────────────────────────────────────
# Load config và models (chỉ load 1 lần khi khởi động)
# ────────────────────────────────────────────────
print("Đang load inference config và models...")
cfg = InferenceConfig()

model1 = YOLO(cfg.model1_path)
model1.to(device)

model2 = YOLO(cfg.model2_path)
model2.to(device)

print("Load model hoàn tất!")
print(f"  • Model 1: {cfg.model1_path}")
print(f"  • Model 2: {cfg.model2_path}\n")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Upload ảnh → chạy ensemble inference (3 nguồn + WBF)
    → trả về ảnh đã annotate (bbox xanh, tâm đỏ, conf vàng)
    """
    # Kiểm tra file là ảnh
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh (jpg, jpeg, png, ...)")

    try:
        # Đọc ảnh từ upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if orig_img is None:
            raise ValueError("Không decode được ảnh")

        start_time = time.time()

        # Chạy inference ensemble
        head_count, final_boxes_pixel, final_scores = process_single_image(
            orig_img=orig_img,
            model1=model1,
            model2=model2,
            device=device,
            cfg=cfg
        )

        # Vẽ kết quả lên ảnh
        annotated_img = draw_detections_on_image(orig_img, final_boxes_pixel, final_scores)

        infer_time = time.time() - start_time
        fps = 1.0 / infer_time if infer_time > 0 else 0.0

        # Encode ảnh thành bytes (JPG)
        _, buffer = cv2.imencode(".jpg", annotated_img)
        img_bytes = BytesIO(buffer)
        img_bytes.seek(0)

        # Headers trả về thông tin bổ sung
        headers = {
            "X-Head-Count": str(head_count),
            "X-Inference-Time": f"{infer_time:.3f} giây",
            "X-FPS": f"{fps:.1f}",
            "X-Model": "ensemble (yolo26x + yolo12x CLAHE + WBF)",
            "Content-Disposition": f"inline; filename=result_{file.filename.rsplit('.', 1)[0]}_ensemble.jpg"
        }

        return StreamingResponse(
            img_bytes,
            media_type="image/jpeg",
            headers=headers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "API Ensemble sẵn sàng!",
        "docs": "/docs (Swagger UI để test upload ảnh)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)