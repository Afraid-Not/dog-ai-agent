"""
임베딩 모델 학습 스크립트
- Stanford Dogs 데이터셋 사용 (tensorflow_datasets에서 로드 후 PyTorch로 변환)
- Triplet Loss (Online Hard Mining) 으로 임베딩 레이어 학습
- 백본(MobileNetV2) + 분류헤드는 동결, 임베딩 레이어만 학습
"""

import os
import sys
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 프로젝트 모듈
from embedding_model import DogEmbeddingModel, load_keras_weights


# ======================================================================
# 설정
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(BASE_DIR, 'trained_models', 'model_1.h5')
SAVE_PATH = os.path.join(BASE_DIR, 'trained_models', 'embedding_model_trained.pth')

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
MARGIN = 0.3          # Triplet loss margin
EMBEDDING_DIM = 128
NUM_BREEDS = 120
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ======================================================================
# 데이터셋 (tensorflow_datasets -> PyTorch)
# ======================================================================

class StanfordDogsDataset(Dataset):
    """
    tensorflow_datasets에서 Stanford Dogs를 로드하여
    PyTorch Dataset으로 변환합니다.
    """

    def __init__(self, split='train', transform=None):
        super().__init__()
        self.transform = transform

        print(f"[데이터] Stanford Dogs ({split}) 로딩 중...")

        # TF 데이터셋 로드
        import tensorflow_datasets as tfds
        ds, info = tfds.load(
            'stanford_dogs',
            split=split,
            as_supervised=False,
            with_info=True,
            data_dir='data/tfds'
        )

        # numpy로 변환
        self.images = []
        self.labels = []
        for sample in ds:
            img = sample['image'].numpy()  # (H, W, 3) uint8
            label = sample['label'].numpy()
            self.images.append(img)
            self.labels.append(int(label))

        self.labels = np.array(self.labels)
        print(f"  -> {len(self.images)}장 로드 완료 ({len(set(self.labels))} 클래스)")

        # 클래스별 인덱스 매핑 (triplet 생성용)
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# ======================================================================
# Triplet 배치 샘플러
# ======================================================================

class TripletBatchSampler:
    """
    각 배치에 P개 클래스 x K개 샘플이 포함되도록 생성.
    Online Hard Triplet Mining에 필수적입니다.
    """

    def __init__(self, labels, p_classes=8, k_samples=4):
        self.labels = labels
        self.p_classes = p_classes
        self.k_samples = k_samples

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        # K개 이상 샘플이 있는 클래스만 사용
        self.valid_classes = [
            c for c, idxs in self.class_to_indices.items()
            if len(idxs) >= k_samples
        ]

        if len(self.valid_classes) < p_classes:
            print(f"  [경고] 유효 클래스({len(self.valid_classes)})가 "
                  f"P({p_classes})보다 적어 P를 {len(self.valid_classes)}로 조정")
            self.p_classes = len(self.valid_classes)

    def __iter__(self):
        random.shuffle(self.valid_classes)
        num_batches = len(self.valid_classes) // self.p_classes

        for b in range(num_batches):
            batch_indices = []
            selected_classes = self.valid_classes[
                b * self.p_classes: (b + 1) * self.p_classes
            ]

            for cls in selected_classes:
                idxs = self.class_to_indices[cls]
                selected = random.sample(idxs, min(self.k_samples, len(idxs)))
                batch_indices.extend(selected)

            yield batch_indices

    def __len__(self):
        return len(self.valid_classes) // self.p_classes


# ======================================================================
# Online Hard Triplet Loss
# ======================================================================

def pairwise_distances(embeddings):
    """배치 내 모든 임베딩 간 L2 거리 행렬 계산"""
    dot = torch.mm(embeddings, embeddings.t())
    sq_norms = torch.diag(dot)
    distances = sq_norms.unsqueeze(0) - 2 * dot + sq_norms.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0)
    return torch.sqrt(distances + 1e-16)


def batch_hard_triplet_loss(embeddings, labels, margin):
    """
    Batch Hard Triplet Loss:
    각 anchor에 대해 가장 먼 positive와 가장 가까운 negative를 선택
    """
    dist_matrix = pairwise_distances(embeddings)

    labels = labels.unsqueeze(0)
    same_class = (labels == labels.t()).float()
    diff_class = 1.0 - same_class

    # 가장 먼 positive (hardest positive)
    pos_dist = dist_matrix * same_class
    hardest_pos, _ = pos_dist.max(dim=1)

    # 가장 가까운 negative (hardest negative)
    # diff_class가 0인 곳(같은 클래스)에 큰 값을 넣어서 무시
    neg_dist = dist_matrix + same_class * 1e6
    hardest_neg, _ = neg_dist.min(dim=1)

    # Triplet loss
    loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)
    return loss.mean()


# ======================================================================
# 학습용 Transform
# ======================================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ======================================================================
# 평가 함수 (Recall@K)
# ======================================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    """Recall@1, Recall@5 평가"""
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        emb = model(images)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 거리 행렬 계산
    dist_matrix = pairwise_distances(all_embeddings)

    # 자기 자신 제외
    dist_matrix.fill_diagonal_(float('inf'))

    # Recall@K
    recalls = {}
    for k in [1, 5]:
        _, topk_indices = dist_matrix.topk(k, largest=False, dim=1)
        topk_labels = all_labels[topk_indices]
        match = (topk_labels == all_labels.unsqueeze(1)).any(dim=1).float()
        recalls[k] = match.mean().item() * 100

    return recalls


# ======================================================================
# 메인 학습 루프
# ======================================================================

def train():
    print("=" * 60)
    print("  Stanford Dogs - Embedding Model Training")
    print("=" * 60)
    print(f"  Device:       {DEVICE}")
    print(f"  Epochs:       {NUM_EPOCHS}")
    print(f"  Batch Size:   P*K = 8*4 = 32")
    print(f"  LR:           {LEARNING_RATE}")
    print(f"  Margin:       {MARGIN}")
    print(f"  Embedding:    {EMBEDDING_DIM}D")
    print("=" * 60)

    # ---- 1) 데이터 로드 ----
    print("\n[1/5] 데이터셋 로딩")
    train_dataset = StanfordDogsDataset(split='train', transform=train_transform)
    test_dataset = StanfordDogsDataset(split='test', transform=val_transform)

    # Triplet 배치 샘플러 (P=8 클래스, K=4 샘플)
    train_sampler = TripletBatchSampler(
        train_dataset.labels, p_classes=8, k_samples=4
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ---- 2) 모델 생성 ----
    print("\n[2/5] 모델 생성 및 가중치 로드")
    model = DogEmbeddingModel(embedding_dim=EMBEDDING_DIM, num_breeds=NUM_BREEDS)
    model = load_keras_weights(model, H5_PATH)
    model = model.to(DEVICE)

    # ---- 3) 옵티마이저 (임베딩 레이어만 학습) ----
    print("\n[3/5] 옵티마이저 설정")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    print(f"  전체 파라미터: {total:,} / 학습 대상: {trainable:,}")

    # ---- 4) 학습 전 평가 ----
    print("\n[4/5] 학습 전 평가")
    recalls = evaluate(model, test_loader, DEVICE)
    print(f"  [평가] Recall@1: {recalls[1]:.2f}%  |  Recall@5: {recalls[5]:.2f}%")

    # ---- 5) 학습 ----
    print("\n[5/5] 학습 시작")
    print("-" * 60)

    best_recall = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        non_zero_triplets = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward
            embeddings = model(images)

            # Triplet Loss
            loss = batch_hard_triplet_loss(embeddings, labels, MARGIN)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if loss.item() > 0:
                non_zero_triplets += 1

            # 진행률 출력
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                avg_loss = total_loss / num_batches
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                      f"Batch [{batch_idx+1}/{len(train_sampler)}] "
                      f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f})")

        scheduler.step()

        # 에폭 종료 평가
        avg_loss = total_loss / max(num_batches, 1)
        recalls = evaluate(model, test_loader, DEVICE)

        print(f"\n  >> Epoch {epoch+1} 완료: "
              f"Loss={avg_loss:.4f} | "
              f"Recall@1={recalls[1]:.2f}% | "
              f"Recall@5={recalls[5]:.2f}% | "
              f"LR={scheduler.get_last_lr()[0]:.6f}")
        print("-" * 60)

        # Best 모델 저장
        if recalls[1] > best_recall:
            best_recall = recalls[1]
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'recall_at_1': recalls[1],
                'recall_at_5': recalls[5],
                'embedding_dim': EMBEDDING_DIM,
                'num_breeds': NUM_BREEDS,
            }, SAVE_PATH)
            print(f"  ** Best 모델 저장! Recall@1: {best_recall:.2f}% -> {SAVE_PATH}")
            print("-" * 60)

    print()
    print("=" * 60)
    print(f"  학습 완료! Best Recall@1: {best_recall:.2f}%")
    print(f"  저장 경로: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
