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
<<<<<<< Updated upstream
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# =========================
# 설정
# =========================
DATA_DIR = "./data/Images"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
EMBEDDING_DIM = 128
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# train / val / test 비율
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Early stopping
EARLY_STOP_PATIENCE = 3   # val_loss가 개선되지 않는 연속 에폭 수
EARLY_STOP_MIN_DELTA = 0.0  # 개선으로 인정할 최소 감소량

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# 데이터 전처리 및 로드
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

full_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = full_ds.classes
num_classes = len(class_names)
print("Classes:", class_names)

n = len(full_ds)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
n_test = n - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    full_ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

# =========================
# 모델 정의
# =========================
class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = base.features  # classifier 제외, embedding만
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Global pool 후 채널 수 (MobileNetV2 last channel = 1280)
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = x.mean([2, 3])  # global average pooling
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# =========================
# Supervised Contrastive Loss (PyTorch)
# =========================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
=======
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
>>>>>>> Stashed changes
        super().__init__()

<<<<<<< Updated upstream
    def forward(self, features, labels):
        labels = labels.reshape(-1, 1)
        batch_size = features.shape[0]

        similarity = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        mask = (labels == labels.T).float()
        logits_mask = 1 - torch.eye(batch_size, device=features.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss


# =========================
# 학습
# =========================
model = EmbeddingModel(embedding_dim=EMBEDDING_DIM).to(DEVICE)
criterion = SupConLoss(temperature=0.07)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train_epoch(loader):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        features = model(images)
        loss = criterion(features, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        features = model(images)
        loss = criterion(features, labels)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches else 0.0


print("\nTraining...")
best_val_loss = float("inf")
epochs_no_improve = 0
best_state = None

for epoch in range(EPOCHS):
    train_loss = train_epoch(train_loader)
    val_loss = eval_epoch(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

    if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (val_loss {EARLY_STOP_PATIENCE} epochs no improve)")
            break

if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

# 최종 test 평가
test_loss = eval_epoch(test_loader)
print(f"\nTest loss: {test_loss:.4f}")

# =========================
# 모델 저장
# =========================
os.makedirs("embedding_output", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "embedding_dim": EMBEDDING_DIM,
}, "embedding_output/model.pt")

# =========================
# centroid 생성 (train set 기준)
# =========================
model.eval()
embeddings_by_class = {i: [] for i in range(num_classes)}

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(DEVICE)
        embs = model(images)
        for e, l in zip(embs.cpu().numpy(), labels.numpy()):
            embeddings_by_class[int(l)].append(e)

centroids = []
for i in range(num_classes):
    vecs = np.array(embeddings_by_class[i])
    centroid = vecs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    centroids.append(centroid)

centroids = np.array(centroids)
np.save("embedding_output/centroids.npy", centroids)
=======
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
>>>>>>> Stashed changes

    # 모델에 로드
    model.classifier.weight.data = weight_tensor
    model.classifier.bias.data = bias_tensor

<<<<<<< Updated upstream
print("\n✅ Training 완료 (train/val/test 분할) + centroid 생성 완료")
=======
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
>>>>>>> Stashed changes
