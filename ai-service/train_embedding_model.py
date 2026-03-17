import os
import numpy as np
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
        super().__init__()
        self.temperature = temperature

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

with open("embedding_output/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("\n✅ Training 완료 (train/val/test 분할) + centroid 생성 완료")
