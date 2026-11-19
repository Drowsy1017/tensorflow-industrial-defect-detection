
import numpy as np
import tensorflow as tf

from pathlib import Path

# 固定隨機種子，提高重現性
tf.keras.utils.set_random_seed(42)

# 1. 建立模擬感測器資料
rng = np.random.default_rng(42)

X = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(1200, 3)
).astype("float32")

# 2. 模擬品質指標的產生規則
noise = rng.normal(
    0.0, 0.3, 1200
).astype("float32")

y = (
    0.6 * X[:, 0]
    + 1.4 * X[:, 1]
    - 0.8 * X[:, 2]
    + noise
).astype("float32")

# 3. 隨機排列資料索引
indices = rng.permutation(len(X))

# 4. 切分資料集
train_idx = indices[:800]
val_idx = indices[800:1000]
test_idx = indices[1000:]

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# 5. 建立標準化層
normalizer = tf.keras.layers.Normalization(
    axis=-1
)

# 只使用訓練資料建立統計參數
normalizer.adapt(X_train)

print("Normalization ready.")


# 6. 建立模型
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    normalizer,
    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),
    tf.keras.layers.Dense(
        8,
        activation="relu"
    ),
    tf.keras.layers.Dense(1)
])

model.summary()


# 7. 設定訓練方式
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)


# 8. 設定 EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)


# 9. 訓練模型
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print(
    "Epochs trained:",
    len(history.history["loss"])
)


# 10. 評估測試集
results = model.evaluate(
    X_test,
    y_test,
    verbose=0,
    return_dict=True
)

print("Test MSE:", results["loss"])
print("Test MAE:", results["mae"])


# 11. 儲存模型
model_path = Path("models/lesson04_sensor.keras")
model_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

model.save(model_path)

print("Saved:", model_path)

# 12. 重新載入模型
loaded_model = tf.keras.models.load_model(
    model_path
)

# 13. 預測一筆測試資料
sample = X_test[:1]

prediction = loaded_model.predict(
    sample,
    verbose=0
)

print("Prediction:", prediction[0, 0])
print("Actual:", y_test[0])
