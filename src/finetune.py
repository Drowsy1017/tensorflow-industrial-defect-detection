
import json

import tensorflow as tf

from .config import CFG
from .dataset import make_dataset, read_manifest


# =====================================
# 1. 基本設定
# =====================================

tf.keras.utils.set_random_seed(CFG.seed)

STAGE1_MODEL = (
    CFG.model_dir / "mobilenetv2_stage1.keras"
)

FINETUNED_MODEL = (
    CFG.model_dir / "mobilenetv2_finetuned.keras"
)

if not STAGE1_MODEL.is_file():
    raise FileNotFoundError(STAGE1_MODEL)


# =====================================
# 2. 載入第一階段模型
# =====================================

model = tf.keras.models.load_model(
    STAGE1_MODEL,
    compile=False
)

# 取得 MobileNetV2 Backbone
backbone = next(
    layer for layer in model.layers
    if isinstance(layer, tf.keras.Model)
)

print("Backbone:", backbone.name)


# =====================================
# 3. 解凍部分 Backbone
# =====================================

backbone.trainable = True

# 只解凍後面 30 層
fine_tune_at = len(backbone.layers) - 30

for i, layer in enumerate(backbone.layers):

    if i < fine_tune_at:
        layer.trainable = False

    else:
        layer.trainable = True

    # BatchNormalization 維持凍結
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


print("Total backbone layers:", len(backbone.layers))
print("Fine-tune starts at:", fine_tune_at)


# =====================================
# 4. 建立 Dataset 與類別權重
# =====================================

train_ds = make_dataset("train")
val_ds = make_dataset("val")

_, labels = read_manifest("train")

normal_count = labels.count(0)
defect_count = labels.count(1)
total = len(labels)

normal_weight = total / (2 * normal_count)
defect_weight = total / (2 * defect_count)


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


# =====================================
# 5. 重新 Compile
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
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


# =====================================
# 6. 設定 Callbacks
# =====================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(FINETUNED_MODEL),
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=5,
        restore_best_weights=True
    )
]


# =====================================
# 7. Fine-tuning
# =====================================

history = model.fit(
    weighted_train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
    shuffle=False,
    verbose=1
)


# =====================================
# 8. 儲存訓練紀錄
# =====================================

CFG.output_dir.mkdir(
    parents=True,
    exist_ok=True
)

history_path = (
    CFG.output_dir / "finetune_history.json"
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


print("Fine-tuning completed.")
print("Saved:", FINETUNED_MODEL)
print("History:", history_path)
