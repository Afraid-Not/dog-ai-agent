"""
강아지 이미지를 분류하고 결과를 Supabase dog_classifications 테이블에 업로드하는 스크립트

사용법:
  단일 이미지: python upload_to_supabase.py dog.jpg
  폴더 전체:   python upload_to_supabase.py ./images/
  임계값 지정: python upload_to_supabase.py ./images/ --threshold 0.5

Supabase 테이블 DDL:
  create table dog_classifications (
    id          uuid primary key default gen_random_uuid(),
    filename    text,
    is_purebred boolean,
    breed_en    text,
    breed_ko    text,
    size        text,
    top3        jsonb,
    threshold   float4,
    created_at  timestamptz default now()
  );
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL_PATH      = "trained_models/model_1.h5"
BREED_DATA_FILE = "breed_data.json"
INPUT_SIZE      = (224, 224)
MIXED_THRESHOLD = 0.40
TOP_K           = 3
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── 리소스 로드 ────────────────────────────────────────────────────────────────
print("모델 로딩 중...")
model = tf.keras.models.load_model(MODEL_PATH)
print("모델 로딩 완료.")

with open(BREED_DATA_FILE, 'r', encoding='utf-8') as f:
    BREED_DATA = json.load(f)  # list[{synset, en, ko, size}], 인덱스 = 모델 클래스 순서

# ── 예측 함수 ──────────────────────────────────────────────────────────────────
def predict(pil_image: Image.Image, threshold: float = MIXED_THRESHOLD):
    img = pil_image.convert('RGB').resize(INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:TOP_K]
    top3 = [
        {
            "rank": rank + 1,
            "breed": BREED_DATA[idx]["en"],
            "probability": round(float(preds[idx]), 4),
            "probability_pct": f"{preds[idx] * 100:.2f}%"
        }
        for rank, idx in enumerate(top_indices)
    ]

    top1_idx    = top_indices[0]
    top1_breed  = BREED_DATA[top1_idx]
    is_purebred = top3[0]["probability"] >= threshold

    return {
        "is_purebred": is_purebred,
        "breed_en":    top1_breed["en"],
        "breed_ko":    top1_breed["ko"],
        "size":        top1_breed["size"],
        "top3":        top3,
        "threshold":   threshold,
    }

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError(".env 파일에 SUPABASE_URL 과 SUPABASE_KEY 를 설정하세요.")

    supabase = create_client(url, key)

    parser = argparse.ArgumentParser(description="강아지 종 분류 결과를 Supabase에 업로드")
    parser.add_argument("path", help="이미지 파일 또는 폴더 경로")
    parser.add_argument("--threshold", type=float, default=MIXED_THRESHOLD,
                        help=f"순종 판단 임계값 (기본값: {MIXED_THRESHOLD})")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_file():
        image_paths = [target]
    elif target.is_dir():
        image_paths = [p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    else:
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {target}")

    print(f"총 {len(image_paths)}개 이미지 처리 시작\n")

    success, fail = 0, 0
    for img_path in image_paths:
        try:
            result = predict(Image.open(img_path), threshold=args.threshold)
            supabase.table("dog_classifications").insert({"filename": img_path.name, **result}).execute()

            purity = "순종" if result["is_purebred"] else "잡종"
            print(f"[OK] {img_path.name}  →  {purity} | {result['breed_en']} | {result['breed_ko']} | {result['size']}")
            success += 1
        except Exception as e:
            print(f"[FAIL] {img_path.name}: {e}")
            fail += 1

    print(f"\n완료: 성공 {success}개 / 실패 {fail}개")


if __name__ == "__main__":
    main()
