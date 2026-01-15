# src/model.py
import os
import torch
import timm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = os.path.join("weights", "best.pth")
DEFAULT_MODEL_NAME = "efficientnet_b1"

model = None
CLASSES = None
MODEL_NAME = None


def load_from_checkpoint(ckpt_path: str):
    global CLASSES, MODEL_NAME

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # 1) 모델 이름 안전 추출
    model_name = None
    if isinstance(ckpt, dict):
        model_name = ckpt.get("model_name")
        if model_name is None and isinstance(ckpt.get("args"), dict):
            model_name = ckpt["args"].get("model")
        if model_name is None:
            model_name = ckpt.get("model")

    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    # 2) 클래스 순서
    if isinstance(ckpt, dict) and "classes" in ckpt:
        CLASSES = ckpt["classes"]
    else:
        CLASSES = ['alert', 'angry', 'frown', 'happy', 'relax']

    num_classes = len(CLASSES)

    # 3) 모델 생성
    m = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # 4) 가중치 로드
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        m.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        m.load_state_dict(ckpt, strict=True)

    m.to(DEVICE)
    m.eval()

    MODEL_NAME = model_name
    return m


def load_model():
    global model
    if model is None:
        if not os.path.isfile(CKPT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
        model = load_from_checkpoint(CKPT_PATH)
    return model


def reload_model():
    global model
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    model = load_from_checkpoint(CKPT_PATH)
    return model
