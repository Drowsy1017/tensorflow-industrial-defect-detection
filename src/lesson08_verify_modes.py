
from pathlib import Path
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

ROOT = Path(__file__).resolve().parents[1]

image_path = (
    ROOT / "dataset/train/abnormal"
    / "group_00_sample_00.png"
)

# 1. 讀取影像
raw = tf.io.read_file(str(image_path))
image = tf.io.decode_png(raw, channels=3)

# 2. Resize 與 Normalize
image = tf.image.resize(image, (224, 224))
image = tf.cast(image, tf.float32) / 255.0

# 3. 建立 Batch 維度
image = tf.expand_dims(image, axis=0)

# 4. 建立增強模型
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(
        factor=0.1,
        seed=42
    ),
    tf.keras.layers.RandomBrightness(
        factor=0.2,
        value_range=(0.0, 1.0),
        seed=43
    )
])

# 5. 分別執行訓練與推論模式
train_output = augmentation(
    image, training=True
)

eval_output = augmentation(
    image, training=False
)

# 6. 計算與原圖的絕對差異
train_diff = tf.reduce_max(
    tf.abs(train_output - image)
)

eval_diff = tf.reduce_max(
    tf.abs(eval_output - image)
)

print("Training difference:", train_diff.numpy())
print("Evaluation difference:", eval_diff.numpy())

# 7. 驗證推論模式是否維持原圖
tf.debugging.assert_near(
    eval_output,
    image
)

print("Inference mode verification passed.")
