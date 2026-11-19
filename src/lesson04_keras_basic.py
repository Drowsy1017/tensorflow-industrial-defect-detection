
import tensorflow as tf

print("TensorFlow:", tf.__version__)

# 1. 建立資料
x_train = tf.constant([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y_train = tf.constant([
    [3.0],
    [5.0],
    [7.0],
    [9.0]
])

# 2. 建立 Keras 模型
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(
        units=1
    )
])

# 3. 顯示模型架構
model.summary()

# 4. 設定訓練方式
model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.05
    ),
    loss="mse",
    metrics=["mae"]
)

# 5. 開始訓練
history = model.fit(
    x_train,
    y_train,
    epochs=200,
    batch_size=4,
    verbose=0
)

# 6. 顯示訓練結果
print("Final Loss:", history.history["loss"][-1])

# 7. 查看學習到的參數
weights, bias = model.layers[0].get_weights()

print("Weight:", weights)
print("Bias:", bias)

# 8. 預測新資料
prediction = model.predict(
    tf.constant([[5.0]]),
    verbose=0
)

print("Prediction for x=5:", prediction)
