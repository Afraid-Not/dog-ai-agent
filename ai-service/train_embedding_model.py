"""
Stanford Dogs 분류 모델(model_1.h5)을 백본으로 사용하여
임베딩 벡터를 추출하는 PyTorch 모델.

구조:
  [MobileNetV2 백본 (frozen)] -> [GAP] -> [FC 1280->120 (from h5)] 
  -> [Embedding Layer 1: 120->256]
  -> [Embedding Layer 2: 256->128]
  -> [Embedding Layer 3: 128->128 (L2 normalized)]
"""

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


# ======================================================================
# 모델 정의
# ======================================================================

class DogEmbeddingModel(nn.Module):
    """
    학습된 Stanford Dogs 분류기를 백본으로 사용하고,
    그 위에 3개의 임베딩 레이어를 추가한 모델.

    출력: L2 정규화된 128차원 임베딩 벡터
    """

    def __init__(self, embedding_dim=128, num_breeds=120):
        super(DogEmbeddingModel, self).__init__()
        # ---- 1) MobileNetV2 백본 (ImageNet 사전학습 가중치) ----
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = mobilenet.features  # feature extractor 부분만 사용
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ---- 2) 분류 헤드 (h5에서 가중치를 로드) ----
        self.classifier = nn.Linear(1280, num_breeds)

        # 백본 + 분류헤드 동결 (pretrained 부분)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False

        # ---- 3) 임베딩 레이어 3개 (새로 학습할 부분) ----
        self.embedding = nn.Sequential(
            # Layer 1: 120 -> 256
            nn.Linear(num_breeds, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Layer 2: 256 -> 128
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),

            # Layer 3: 128 -> 128 (최종 임베딩)
            nn.Linear(embedding_dim, embedding_dim),
        )

        self._init_embedding_weights()

    def _init_embedding_weights(self):
        """임베딩 레이어 가중치를 Xavier 초기화"""
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224) 입력 이미지 텐서
        Returns:
            embedding: (batch, 128) L2 정규화된 임베딩 벡터
        """
        # 백본: 특징 추출
        x = self.backbone(x)           # (batch, 1280, 7, 7)
        x = self.pool(x)               # (batch, 1280, 1, 1)
        x = torch.flatten(x, 1)        # (batch, 1280)

        # 분류 헤드 (학습된 가중치)
        x = self.classifier(x)         # (batch, 120)

        # 임베딩 레이어
        x = self.embedding(x)          # (batch, 128)

        # L2 정규화 (코사인 유사도 사용 시 필수)
        x = nn.functional.normalize(x, p=2, dim=1)

        return x

    def extract_backbone_features(self, x):
        """백본 특징만 추출 (1280차원)"""
        with torch.no_grad():
            x = self.backbone(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
        return x

    def extract_breed_features(self, x):
        """품종 분류 확률 벡터 추출 (120차원)"""
        with torch.no_grad():
            x = self.backbone(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x


# ======================================================================
# Keras .h5 -> PyTorch 가중치 변환
# ======================================================================

def load_keras_weights(model, h5_path):
    """
    Keras로 학습된 model_1.h5에서 분류 헤드(Dense 레이어)의
    가중치를 읽어 PyTorch 모델에 로드합니다.
    
    - Keras Dense kernel: (in_features, out_features) = (1280, 120)
    - PyTorch Linear weight: (out_features, in_features) = (120, 1280)
    -> 전치(transpose)가 필요합니다.
    """
    with h5py.File(h5_path, 'r') as f:
        # Dense 레이어 가중치 추출
        kernel = np.array(f['model_weights/dense/dense/kernel:0'])  # (1280, 120)
        bias = np.array(f['model_weights/dense/dense/bias:0'])      # (120,)

    # Keras -> PyTorch 변환 (kernel 전치)
    weight_tensor = torch.from_numpy(kernel.T).float()  # (120, 1280)
    bias_tensor = torch.from_numpy(bias).float()        # (120,)
    model.classifier.weight.data = weight_tensor
    model.classifier.bias.data = bias_tensor
    print(f"[OK] Keras 가중치 로드 완료: {h5_path}")
    print(f"     - kernel shape: {kernel.shape} -> weight shape: {weight_tensor.shape}")
    print(f"     - bias shape:   {bias.shape}")

    return model


# ======================================================================
# 이미지 전처리 (PyTorch용)
# ======================================================================

def get_transform():
    """MobileNetV2에 맞는 이미지 전처리 파이프라인"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ======================================================================
# 메인: 모델 생성 및 테스트
# ======================================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    H5_PATH = os.path.join(BASE_DIR, 'trained_models', 'model_1.h5')

    # 1. 모델 생성
    print("=" * 60)
    print("[1] DogEmbeddingModel 생성")
    print("=" * 60)
    model = DogEmbeddingModel(embedding_dim=128, num_breeds=120)

    # 2. Keras 가중치 로드
    print()
    print("=" * 60)
    print("[2] Keras 가중치 로드 (분류 헤드)")
    print("=" * 60)
    model = load_keras_weights(model, H5_PATH)

    # 3. 모델 구조 출력
    print()
    print("=" * 60)
    print("[3] 모델 구조")
    print("=" * 60)
    print(model)

    # 4. 파라미터 통계
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print()
    print("=" * 60)
    print("[4] 파라미터 통계")
    print("=" * 60)
    print(f"  전체 파라미터:     {total_params:>10,}")
    print(f"  학습 가능 파라미터: {trainable_params:>10,} (임베딩 레이어)")
    print(f"  동결 파라미터:     {frozen_params:>10,} (백본 + 분류헤드)")

    # 5. 더미 입력으로 테스트
    print()
    print("=" * 60)
    print("[5] 추론 테스트 (랜덤 이미지)")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224)  # 배치 크기 2

        # 임베딩 추출
        embeddings = model(dummy_input)
        print(f"  입력 shape:   {dummy_input.shape}")
        print(f"  임베딩 shape: {embeddings.shape}")
        print(f"  임베딩 L2 norm: {torch.norm(embeddings, dim=1)}")

        # 두 이미지 간 코사인 유사도
        cos_sim = nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        )
        print(f"  두 이미지 코사인 유사도: {cos_sim.item():.4f}")

    # 6. 모델 저장
    save_path = os.path.join(BASE_DIR, 'trained_models', 'embedding_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': 128,
        'num_breeds': 120,
    }, save_path)
    print()
    print("=" * 60)
    print(f"[6] 모델 저장 완료: {save_path}")
    print("=" * 60)
