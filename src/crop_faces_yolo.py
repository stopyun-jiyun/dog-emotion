import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

SRC_ROOT = "dataset/dataset"
DST_ROOT = "dataset_face"
WEIGHTS = "models/dogface.pt"
IMG_SIZE = 512
MARGIN = 0.2

def crop_with_margin(img, box, margin=0.2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return img[y1:y2, x1:x2]

def pick_largest_box(res):
    if res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    i = int(areas.argmax())
    return boxes[i].astype(int)

def main():
    # 경로 체크
    train_dir = os.path.join(SRC_ROOT, "train")
    val_dir = os.path.join(SRC_ROOT, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train 폴더 없음: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"val 폴더 없음: {val_dir}")

    model = YOLO(WEIGHTS)

    # 클래스 폴더 목록
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print("Classes:", classes)

    for split in ["train", "val"]:
        for cls in classes:
            src_dir = os.path.join(SRC_ROOT, split, cls)
            dst_dir = os.path.join(DST_ROOT, split, cls)
            os.makedirs(dst_dir, exist_ok=True)

            if not os.path.isdir(src_dir):
                print("Skip missing:", src_dir)
                continue

            files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
            for fn in tqdm(files, desc=f"{split}/{cls}", ncols=100):
                img_path = os.path.join(src_dir, fn)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                res = model.predict(img, imgsz=IMG_SIZE, conf=0.25, device=0, verbose=False)[0]
                box = pick_largest_box(res)

                if box is None:
                    # 얼굴 못 찾으면 원본 그대로 저장(데이터 손실 방지 + 폴더 생성 확실)
                    out = img
                else:
                    out = crop_with_margin(img, box, MARGIN)

                out_path = os.path.join(dst_dir, fn)
                cv2.imwrite(out_path, out)

    print(f"✅ Done. Saved to: {DST_ROOT}")

if __name__ == "__main__":
    main()
