
import tensorflow as tf

# 建立資料
x_train = tf.constant(
    [1., 2., 3., 4.],
    dtype=tf.float32
)

y_train = tf.constant(
    [3., 5., 7., 9.],
    dtype=tf.float32
)

# 初始化參數
w = tf.Variable(0.0)
b = tf.Variable(0.0)

# Optimizer
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.05
)

# 建立訓練 Step
@tf.function
def train_step(x, y):

    # 1. Forward + Loss
    with tf.GradientTape() as tape:

        prediction = w * x + b

        loss = tf.reduce_mean(
            tf.square(prediction - y)
        )

    # 2. Backward
    gradients = tape.gradient(
        loss, [w, b]
    )

    # 3. Update
    optimizer.apply_gradients(
        zip(gradients, [w, b])
    )

    return loss

# 訓練 100 次
for epoch in range(100):

    loss = train_step(
        x_train,
        y_train
    )

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1} | "
            f"Loss {loss.numpy():.6f}"
        )

print("Final Weight:", w.numpy())
print("Final Bias:", b.numpy())
