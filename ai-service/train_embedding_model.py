import os
import numpy as np
import tensorflow as tf

# =========================
# 설정
# =========================
DATA_DIR = "dataset/images"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
EMBEDDING_DIM = 128

# =========================
# 데이터 로드
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)

# =========================
# 모델 정의
# =========================
def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(EMBEDDING_DIM)(x)
    outputs = tf.math.l2_normalize(x, axis=1)

    return tf.keras.Model(inputs, outputs)

# =========================
# contrastive loss
# =========================
class SupConLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def call(self, labels, features):
        labels = tf.reshape(labels, [-1, 1])

        similarity = tf.matmul(features, features, transpose_b=True)
        similarity = similarity / self.temperature

        logits_max = tf.reduce_max(similarity, axis=1, keepdims=True)
        logits = similarity - logits_max

        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

        logits_mask = tf.ones_like(mask) - tf.eye(tf.shape(labels)[0])
        mask = mask * logits_mask

        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-9)

        mask_sum = tf.reduce_sum(mask, axis=1)
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (mask_sum + 1e-9)

        loss = -tf.reduce_mean(mean_log_prob_pos)
        return loss

# =========================
# 학습
# =========================
model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=SupConLoss()
)

model.summary()

model.fit(
    train_ds,
    epochs=EPOCHS
)

# =========================
# 모델 저장
# =========================
os.makedirs("embedding_output", exist_ok=True)
model.save("embedding_output/model.keras")

# =========================
# centroid 생성
# =========================
embeddings_by_class = {i: [] for i in range(len(class_names))}

for images, labels in train_ds:
    embs = model.predict(images, verbose=0)
    labels = labels.numpy()

    for e, l in zip(embs, labels):
        embeddings_by_class[int(l)].append(e)

centroids = []
for i in range(len(class_names)):
    vecs = np.array(embeddings_by_class[i])
    centroid = vecs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    centroids.append(centroid)

centroids = np.array(centroids)

np.save("embedding_output/centroids.npy", centroids)

with open("embedding_output/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("\n✅ Training 완료 + centroid 생성 완료")