
import tensorflow as tf

# 1. 定義輸入
inputs = tf.keras.Input(
    shape=(3,),
    name="sensor_input"
)

# 2. 第一個隱藏層
x = tf.keras.layers.Dense(
    units=16,
    activation="relu",
    name="hidden_1"
)(inputs)

# 3. 第二個隱藏層
x = tf.keras.layers.Dense(
    units=8,
    activation="relu",
    name="hidden_2"
)(x)

# 4. 輸出層
outputs = tf.keras.layers.Dense(
    units=1,
    activation="sigmoid",
    name="prediction"
)(x)

# 5. 建立模型
model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="sensor_classifier"
)

# 6. 檢查架構
model.summary()

# 7. 模擬輸入
sample = tf.constant([
    [0.5, 0.2, 0.8],
    [0.1, 0.9, 0.4]
], dtype=tf.float32)

# 8. 執行 Forward
prediction = model(sample, training=False)

print("Output shape:", prediction.shape)
print("Predictions:", prediction.numpy())
