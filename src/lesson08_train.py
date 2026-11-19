
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

# 匯入第 7 課的資料管線
from lesson07_image_pipeline import build_dataset


# ==========================================
# 1. 基本設定
# ==========================================

tf.keras.utils.set_random_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("TensorFlow:", tf.__version__)
print(
    "GPU:",
    tf.config.list_physical_devices("GPU")
)


# ==========================================
# 2. 載入 Dataset
# ==========================================

train_ds = build_dataset(
    "train",
    training=True
)

val_ds = build_dataset("val")
test_ds = build_dataset("test")


# ==========================================
# 3. 建立 Data Augmentation
# ==========================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        "horizontal",
        seed=42
    ),

    tf.keras.layers.RandomRotation(
        factor=0.03,
        fill_mode="reflect",
        seed=43
    ),

    tf.keras.layers.RandomBrightness(
        factor=0.08,
        value_range=(0.0, 1.0),
        seed=44
    ),

    tf.keras.layers.RandomContrast(
        factor=0.1,
        seed=45
    )
], name="data_augmentation")


# ==========================================
# 4. 建立 CNN 模型
# ==========================================

model = tf.keras.Sequential([

    tf.keras.Input(shape=(224, 224, 3)),

    augmentation,

    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.summary()


# ==========================================
# 5. 編譯模型
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        )
    ]
)


# ==========================================
# 6. 訓練模型
# ==========================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    shuffle=False,
    verbose=1
)


# ==========================================
# 7. 評估模型
# ==========================================

results = model.evaluate(
    test_ds,
    return_dict=True,
    verbose=0
)

print("Test results:", results)


# ==========================================
# 8. 儲存模型
# ==========================================

model_path = (
    PROJECT_ROOT
    / "models"
    / "lesson08_augmented_cnn.keras"
)

model_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

model.save(model_path)

print("Saved model:", model_path)


# ==========================================
# 9. 繪製訓練曲線
# ==========================================

history_data = history.history

train_loss = history_data["loss"]
val_loss = history_data["val_loss"]

train_acc = history_data["accuracy"]
val_acc = history_data["val_accuracy"]

epochs = range(1, len(train_loss) + 1)

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4)
)

# Loss 曲線
axes[0].plot(
    epochs,
    train_loss,
    label="Training Loss"
)

axes[0].plot(
    epochs,
    val_loss,
    label="Validation Loss"
)

axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

# Accuracy 曲線
axes[1].plot(
    epochs,
    train_acc,
    label="Training Accuracy"
)

axes[1].plot(
    epochs,
    val_acc,
    label="Validation Accuracy"
)

axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()

output_path = (
    PROJECT_ROOT
    / "outputs"
    / "lesson08_history.png"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    output_path,
    dpi=160
)

plt.close(fig)

print("Saved chart:", output_path)
