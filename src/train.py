
import json

from pathlib import Path

import tensorflow as tf

from .config import CFG
from .dataset import make_dataset
from .model import build_model


# ==========================================
# 1. 基本設定
# ==========================================

tf.keras.utils.set_random_seed(CFG.seed)

print("TensorFlow:", tf.__version__)
print(
    "GPU:",
    tf.config.list_physical_devices("GPU")
)


# ==========================================
# 2. 建立 Dataset
# ==========================================

train_ds = make_dataset("train")
val_ds = make_dataset("val")


# ==========================================
# 3. 處理類別不平衡
# ==========================================

NORMAL_COUNT = 1772
DEFECT_COUNT = 209

TOTAL = NORMAL_COUNT + DEFECT_COUNT

normal_weight = TOTAL / (2 * NORMAL_COUNT)
defect_weight = TOTAL / (2 * DEFECT_COUNT)

print("Normal weight:", normal_weight)
print("Defect weight:", defect_weight)


def add_sample_weight(images, labels):

    weights = tf.where(
        labels > 0.5,
        tf.cast(defect_weight, tf.float32),
        tf.cast(normal_weight, tf.float32)
    )

    return images, labels, weights


weighted_train_ds = train_ds.map(
    add_sample_weight,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ==========================================
# 4. 建立 MobileNetV2
# ==========================================

model = build_model()


# ==========================================
# 5. Compile
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.AUC(
            curve="PR",
            name="pr_auc"
        )
    ]
)


# ==========================================
# 6. 訓練回呼
# ==========================================

model_path = (
    CFG.model_dir
    / "mobilenetv2_stage1.keras"
)

CFG.model_dir.mkdir(
    parents=True,
    exist_ok=True
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]


# ==========================================
# 7. 訓練模型
# ==========================================

history = model.fit(
    weighted_train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)


# ==========================================
# 8. 儲存訓練紀錄
# ==========================================

CFG.output_dir.mkdir(
    parents=True,
    exist_ok=True
)

history_path = (
    CFG.output_dir
    / "stage1_history.json"
)

with history_path.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history.history,
        f,
        indent=2
    )


# ==========================================
# 9. 顯示結果
# ==========================================

print("Training completed.")
print("Epochs:", len(history.history["loss"]))
print("Best model:", model_path)
print("History:", history_path)
