import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# breed 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'breed'))

# ImageNet 기준 개 품종 클래스 인덱스 범위
DOG_CLASS_START = 151
DOG_CLASS_END = 268

model = None


def load_model():
    global model
    if model is None:
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model


def is_dog(image_path: str) -> bool:
    """
    이미지 경로를 받아 강아지 여부를 반환합니다.

    Args:
        image_path: 이미지 파일 경로

    Returns:
        True if dog, False otherwise
    """
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(np.array(img), axis=0)
    )

    preds = load_model().predict(x, verbose=0)
    top_class = np.argmax(preds[0])

    return DOG_CLASS_START <= top_class <= DOG_CLASS_END


def is_dog_from_bytes(image_bytes: bytes) -> bool:
    """
    이미지 바이트를 받아 강아지 여부를 반환합니다.

    Args:
        image_bytes: 이미지 바이트 데이터

    Returns:
        True if dog, False otherwise
    """
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(np.array(img), axis=0)
    )

    preds = load_model().predict(x, verbose=0)
    top_class = np.argmax(preds[0])

    return DOG_CLASS_START <= top_class <= DOG_CLASS_END


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dog_detector.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    result = is_dog(path)
    print("강아지 O" if result else "강아지 X")

    if result:
        from savetojson import save_to_json
        saved_path = save_to_json(path)
        print(f"JSON 저장 완료: {saved_path}")
