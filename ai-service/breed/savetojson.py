"""
강아지 이미지를 분류하고 결과를 JSON 파일로 저장하는 스크립트

사용법:
  단일 이미지: python savetojson.py dog.jpg          → data/dog.json
  폴더 전체:   python savetojson.py ./images/         → data/{이미지명}.json (개별 저장)
  임계값 지정: python savetojson.py ./images/ --threshold 0.5
  출력 폴더:   python savetojson.py ./images/ --outdir results/
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tf_keras
from PIL import Image

# ── 설정 ──────────────────────────────────────────────────────────────────────
_DIR            = Path(__file__).parent
MODEL_PATH      = str(_DIR / "trained_models" / "model_1.h5")
BREED_DATA_FILE = str(_DIR / "breed_data.json")
INPUT_SIZE      = (224, 224)
MIXED_THRESHOLD = 0.50
TOP_K           = 3
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── 리소스 로드 ────────────────────────────────────────────────────────────────
class FixedDepthwiseConv2D(tf_keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

_model = None
_breed_data = None


def _load_resources():
    global _model, _breed_data
    if _model is None:
        print("모델 로딩 중...")
        _model = tf_keras.models.load_model(
            MODEL_PATH,
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
        )
        print("모델 로딩 완료.")
    if _breed_data is None:
        with open(BREED_DATA_FILE, 'r', encoding='utf-8') as f:
            _breed_data = json.load(f)
    return _model, _breed_data

# ── 예측 함수 ──────────────────────────────────────────────────────────────────
def predict(pil_image: Image.Image, threshold: float = MIXED_THRESHOLD):
    model, breed_data = _load_resources()
    img = pil_image.convert('RGB').resize(INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:TOP_K]
    top3 = [
        {
            "rank": rank + 1,
            "breed": breed_data[idx]["en"],
            "probability": round(float(preds[idx]), 4),
            "probability_pct": f"{preds[idx] * 100:.2f}%"
        }
        for rank, idx in enumerate(top_indices)
    ]

    top1_idx    = top_indices[0]
    top1_breed  = breed_data[top1_idx]
    is_purebred = top3[0]["probability"] >= threshold

    return {
        "is_purebred": is_purebred,
        "breed_en":    top1_breed["en"],
        "breed_ko":    top1_breed["ko"],
        "size":        top1_breed["size"],
        "top3":        top3,
        "threshold":   threshold,
    }


def save_to_json(image_path: str, outdir: str = "data", threshold: float = MIXED_THRESHOLD) -> str:
    """
    강아지 이미지를 분류하고 결과를 JSON 파일로 저장합니다.

    Args:
        image_path: 이미지 파일 경로
        outdir: 결과 JSON을 저장할 폴더 (기본값: data)
        threshold: 순종 판단 임계값

    Returns:
        저장된 JSON 파일의 절대 경로
    """
    img_path = Path(image_path)
    result = predict(Image.open(img_path), threshold=threshold)
    record = {"filename": img_path.name, "created_at": datetime.now().isoformat(), **result}

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / (img_path.stem + ".json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return str(output_path.resolve())

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="강아지 종 분류 결과를 JSON 파일로 저장")
    parser.add_argument("path", help="이미지 파일 또는 폴더 경로")
    parser.add_argument("--threshold", type=float, default=MIXED_THRESHOLD,
                        help=f"순종 판단 임계값 (기본값: {MIXED_THRESHOLD})")
    parser.add_argument("--outdir", default="data",
                        help="결과 JSON을 저장할 폴더 (기본값: data)")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_file():
        image_paths = [target]
    elif target.is_dir():
        image_paths = [p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    else:
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {target}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _load_resources()
    print(f"총 {len(image_paths)}개 이미지 처리 시작\n")

    success, fail = 0, 0
    for img_path in image_paths:
        try:
            result = predict(Image.open(img_path), threshold=args.threshold)
            record = {"filename": img_path.name, "created_at": datetime.now().isoformat(), **result}

            output_path = out_dir / (img_path.stem + ".json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            purity = "순종" if result["is_purebred"] else "잡종"
            print(f"[OK] {img_path.name}  →  {purity} | {result['breed_en']} | {result['breed_ko']} | {result['size']}")
            print(f"     저장: {output_path.resolve()}")
            success += 1
        except Exception as e:
            print(f"[FAIL] {img_path.name}: {e}")
            fail += 1

    print(f"\n완료: 성공 {success}개 / 실패 {fail}개")


if __name__ == "__main__":
    main()
