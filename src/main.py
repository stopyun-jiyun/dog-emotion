from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

from model import predict  # model.py의 predict() 사용

app = FastAPI()

# /static 아래에 있는 파일들(HTML/JS/CSS) 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return {"message": "server ok"}

# 웹캠 페이지
@app.get("/webcam")
def webcam_page():
    return FileResponse("static/index.html")

# 웹캠 프레임(이미지)을 받아서 예측
@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return predict(image)

