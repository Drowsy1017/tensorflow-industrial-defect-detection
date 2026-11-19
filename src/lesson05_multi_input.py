
import numpy as np
import tensorflow as tf
from pathlib import Path

tf.keras.utils.set_random_seed(42)

# 1. 建立兩組輸入
sensor_input = tf.keras.Input(
    shape=(3,),
    name="sensor_input"
)

status_input = tf.keras.Input(
    shape=(2,),
    name="status_input"
)

# 2. 感測器分支
sensor_features = tf.keras.layers.Dense(
    8,
    activation="relu",
    name="sensor_dense"
)(sensor_input)

# 3. 運轉狀態分支
status_features = tf.keras.layers.Dense(
    4,
    activation="relu",
    name="status_dense"
)(status_input)

# 4. 特徵融合
merged = tf.keras.layers.Concatenate(
    axis=-1,
    name="feature_fusion"
)([sensor_features, status_features])

# 5. 融合後的隱藏層
hidden = tf.keras.layers.Dense(
    8,
    activation="relu",
    name="fusion_dense"
)(merged)

# 6. 二元分類輸出
output = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="abnormal_probability"
)(hidden)

# 7. 建立 Functional Model
model = tf.keras.Model(
    inputs={
        "sensor_input": sensor_input,
        "status_input": status_input
    },
    outputs=output,
    name="industrial_multi_input"
)

model.summary()

# 8. 建立合成資料
rng = np.random.default_rng(42)

sensor_data = rng.normal(
    size=(1000, 3)
).astype("float32")

status_data = rng.normal(
    size=(1000, 2)
).astype("float32")

# 9. 人為定義異常標籤
signal = (
    sensor_data[:, 0]
    + 0.8 * sensor_data[:, 1]
    - 0.5 * sensor_data[:, 2]
    + 0.7 * status_data[:, 0]
    + 0.4 * status_data[:, 1]
)

# 加入少量隨機擾動
signal += rng.normal(0, 0.2, size=1000)

labels = (signal > 0.5).astype("float32")
labels = labels.reshape(-1, 1)

# 10. 建立 Train/Validation/Test 索引
indices = rng.permutation(1000)

train_idx = indices[:700]
val_idx = indices[700:850]
test_idx = indices[850:]

def make_inputs(idx):
    return {
        "sensor_input": sensor_data[idx],
        "status_input": status_data[idx]
    }

# 11. 設定模型訓練方式
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

# 12. 訓練模型
history = model.fit(
    make_inputs(train_idx),
    labels[train_idx],
    validation_data=(
        make_inputs(val_idx),
        labels[val_idx]
    ),
    epochs=40,
    batch_size=32,
    verbose=1
)

# 13. 最終測試
results = model.evaluate(
    make_inputs(test_idx),
    labels[test_idx],
    return_dict=True,
    verbose=0
)

print("Test results:", results)

# 14. 儲存模型
path = Path("models/lesson05_multi_input.keras")
path.parent.mkdir(parents=True, exist_ok=True)

model.save(path)
print("Model saved:", path)

# 15. 重新載入並推論
loaded = tf.keras.models.load_model(path)

pred = loaded.predict(
    make_inputs(test_idx[:3]),
    verbose=0
)

print("Predictions:", pred.flatten())
print("Actual labels:", labels[test_idx[:3]].flatten())
