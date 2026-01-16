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
# Lazy-loaded globals (IMPORTANT for Render)
# ===============================
model = None
yolo = None

ckpt = None
CLASSES = None
MODEL_NAME = None
IMG_SIZE = None
tfm = None


def load_models():
    """
    Render(Web Service)에서 가장 흔한 문제(포트 스캔 타임아웃)는
    앱 import 시점에 무거운 모델 로딩이 들어가서 uvicorn이 포트를 열기 전에 죽는 거야.
    그래서 필요한 순간(첫 요청)에만 딱 1번 로딩하도록 lazy-load로 바꿔둠.
    """
    global model, yolo, ckpt, CLASSES, MODEL_NAME, IMG_SIZE, tfm

    # ---- EfficientNet (timm) ----
    if model is None:
        # Render Free 안정성을 위해 CPU 로드 권장 (필요하면 아래 map_location=DEVICE로 바꿔도 됨)
        ckpt = torch.load(CKPT_PATH, map_location="cpu")

        CLASSES = ckpt["classes"]
        MODEL_NAME = ckpt["model_name"]
        IMG_SIZE = ckpt["img_size"]

        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=len(CLASSES),
        )
        model.load_state_dict(ckpt["state_dict"])

        # GPU 사용 가능하면 옮기기 (로컬은 cuda, Render는 보통 cpu)
        model.to(DEVICE)
        model.eval()

        tfm = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ---- YOLO ----
    if yolo is None:
        yolo = YOLO(str(YOLO_PATH))


# ===============================
# Routes
# ===============================
@app.get("/")
def root():
    # 가벼운 상태 확인용: 모델/YOLO를 여기서 로드하지 않음!
    return {
        "status": "ok",
        "device": DEVICE,
        "has_model_loaded": model is not None,
        "has_yolo_loaded": yolo is not None,
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
    # ✅ 첫 요청에서만 모델 로드 (Render 포트 문제 해결)
    load_models()

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
        "model": MODEL_NAME,
        "img_size": IMG_SIZE,
    }
