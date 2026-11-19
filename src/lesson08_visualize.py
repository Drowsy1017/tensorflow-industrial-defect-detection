
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf


# ==========================================
# 1. 基本設定
# ==========================================

tf.keras.utils.set_random_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "abnormal"
    / "group_00_sample_00.png"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. 載入原始影像
# ==========================================

raw = tf.io.read_file(str(IMAGE_PATH))

image = tf.io.decode_png(
    raw,
    channels=3
)

image = tf.image.resize(
    image,
    (224, 224)
)

image = tf.cast(
    image,
    tf.float32
) / 255.0

# 加上 Batch 維度
image_batch = tf.expand_dims(
    image,
    axis=0
)

print("Image shape:", image.shape)
print("Batch shape:", image_batch.shape)


# ==========================================
# 3. 建立四個增強 Layer
# ==========================================

flip = tf.keras.layers.RandomFlip(
    mode="horizontal",
    seed=42
)

rotation = tf.keras.layers.RandomRotation(
    factor=0.03,
    fill_mode="reflect",
    seed=42
)

brightness = tf.keras.layers.RandomBrightness(
    factor=0.08,
    value_range=(0.0, 1.0),
    seed=42
)

contrast = tf.keras.layers.RandomContrast(
    factor=0.1,
    seed=42
)


# ==========================================
# 4. 執行影像增強
# ==========================================

flipped = flip(
    image_batch,
    training=True
)

rotated = rotation(
    image_batch,
    training=True
)

brightened = brightness(
    image_batch,
    training=True
)

contrasted = contrast(
    image_batch,
    training=True
)


# ==========================================
# 5. 建立綜合增強流程
# ==========================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        "horizontal"
    ),
    tf.keras.layers.RandomRotation(
        0.03,
        fill_mode="reflect"
    ),
    tf.keras.layers.RandomBrightness(
        0.08,
        value_range=(0.0, 1.0)
    ),
    tf.keras.layers.RandomContrast(
        0.1
    )
])

combined = augmentation(
    image_batch,
    training=True
)


# ==========================================
# 6. 繪製比較圖
# ==========================================

images = [
    image,
    flipped[0],
    rotated[0],
    brightened[0],
    contrasted[0],
    combined[0]
]

titles = [
    "Original",
    "RandomFlip",
    "RandomRotation",
    "RandomBrightness",
    "RandomContrast",
    "Combined"
]

fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 8)
)

for ax, img, title in zip(
    axes.flat,
    images,
    titles
):

    # 限制顯示數值到 0~1
    img = tf.clip_by_value(
        img,
        0.0,
        1.0
    )

    ax.imshow(img.numpy())
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

output_path = (
    OUTPUT_DIR / "lesson08_augmentation.png"
)

plt.savefig(
    output_path,
    dpi=160
)

plt.close(fig)

print("Saved:", output_path)
