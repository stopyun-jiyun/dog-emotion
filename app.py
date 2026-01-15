# app.py
import io
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ultralytics import YOLO



# ===============================
# App
# ===============================
app = FastAPI()

# CORS (프론트/웹캠 호출 안전)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시엔 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

# ===============================
# Paths
# ===============================
BASE_DIR = Path(__file__).resolve().parent

CKPT_PATH = BASE_DIR / "weights" / "best.pth"
YOLO_PATH = BASE_DIR / "models" / "dogface.pt"  # 강아지 얼굴 검출 모델
WEB_DIR = BASE_DIR / "web"

# 정적 파일 서빙: /web/style.css, /web/app.js
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

# ===============================
# Device
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Load EfficientNet checkpoint
# ===============================
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

CLASSES = ckpt["classes"]
MODEL_NAME = ckpt["model_name"]
IMG_SIZE = ckpt["img_size"]

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=len(CLASSES),
)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE)
model.eval()

# ✅ transform (tfm) 전역 정의 — NameError 방지
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===============================
# Load YOLO (once at startup)
# ===============================
yolo = YOLO(str(YOLO_PATH))

# ===============================
# Routes
# ===============================
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "img_size": IMG_SIZE,
        "classes": CLASSES,
        "yolo": str(YOLO_PATH.name),
        "device": DEVICE,
    }

@app.get("/cam")
def cam_page():
    # index.html 안에서 /web/style.css, /web/app.js로 불러오도록 해야 함
    return FileResponse(str(WEB_DIR / "index.html"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Flow:
    1) Image -> YOLO detect face -> crop (largest box)
    2) EfficientNet classify on crop (fallback: full image)
    """
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = img.size

    # -------- YOLO detect --------
    face_crop = None
    best_box = None

    # imgsz 낮추면 속도↑ (얼굴이 너무 작으면 실패할 수도)
    results = yolo.predict(img, conf=0.25, imgsz=320, verbose=False)
    r0 = results[0]

    if r0.boxes is not None and len(r0.boxes) > 0:
        boxes = r0.boxes.xyxy.cpu().numpy()  # (N,4)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[i]

        # padding
        pad = 0.12
        bw = (x2 - x1)
        bh = (y2 - y1)

        x1 = max(0, int(x1 - bw * pad))
        y1 = max(0, int(y1 - bh * pad))
        x2 = min(W, int(x2 + bw * pad))
        y2 = min(H, int(y2 + bh * pad))

        # sanity (너무 작은 박스는 무시)
        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            face_crop = img.crop((x1, y1, x2, y2))
            best_box = [x1, y1, x2, y2]

    # fallback: 얼굴 못 찾으면 전체 이미지
    src = face_crop if face_crop is not None else img

    # -------- EfficientNet classify --------
    x = tfm(src).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(prob).item())

    return {
        "class": CLASSES[idx],
        "confidence": float(prob[idx].item()),
        "probs": {CLASSES[i]: float(prob[i].item()) for i in range(len(CLASSES))},
        "face_detected": face_crop is not None,
        "box_xyxy": best_box,
    }
