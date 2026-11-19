
import numpy as np
import tensorflow as tf

# 1. 固定隨機種子
tf.keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)

# 2. 建立 1000 筆模擬感測器資料
X = rng.normal(
    size=(1000, 3)
).astype(np.float32)

# 3. 建立合成二元分類標籤
signal = (
    1.2 * X[:, 0]
    - 0.8 * X[:, 1]
    + 0.5 * X[:, 2]
)

noise = rng.normal(
    0.0, 0.2, size=1000
)

y = (signal + noise > 0).astype(np.float32)
y = y.reshape(-1, 1)

# 4. 建立固定的資料切分
indices = rng.permutation(len(X))

train_idx = indices[:700]
val_idx = indices[700:850]
test_idx = indices[850:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# 5. 只使用 Training Data 計算統計值
mean = X_train.mean(axis=0).astype(np.float32)
std = X_train.std(axis=0).astype(np.float32)
std = np.maximum(std, np.float32(1e-6))

mean_tf = tf.constant(mean)
std_tf = tf.constant(std)

# 6. 建立前處理函式
def preprocess(features, label):

    features = (features - mean_tf) / std_tf

    return (
        tf.cast(features, tf.float32),
        tf.cast(label, tf.float32)
    )

# 7. 建立 tf.data Pipeline
def make_dataset(features, labels, training=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (features, labels)
    )

    ds = ds.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 本次合成資料很小，允許快取
    ds = ds.cache()

    if training:
        ds = ds.shuffle(
            buffer_size=len(features),
            seed=42,
            reshuffle_each_iteration=True
        )

    ds = ds.batch(32)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# 8. 分別建立三個 Dataset
train_ds = make_dataset(
    X_train, y_train, training=True
)

val_ds = make_dataset(
    X_val, y_val, training=False
)

test_ds = make_dataset(
    X_test, y_test, training=False
)

# 9. 檢查 Batch Shape
for batch_x, batch_y in train_ds.take(1):
    print("Batch X:", batch_x.shape)
    print("Batch Y:", batch_y.shape)

# 10. 建立 Keras 模型
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 11. 使用 Dataset 訓練
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    verbose=1
)

# 12. 最後測試
results = model.evaluate(
    test_ds,
    verbose=0,
    return_dict=True
)

print("Test results:", results)

# 13. 查看模型推論
for batch_x, batch_y in test_ds.take(1):
    predictions = model(
        batch_x,
        training=False
    )

    print("Predictions:", predictions[:5].numpy())
    print("Labels:", batch_y[:5].numpy())
    break
