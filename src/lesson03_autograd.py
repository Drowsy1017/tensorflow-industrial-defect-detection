
import tensorflow as tf

print("=== Lesson 03: Variables ===")

# 固定 Tensor
x = tf.constant(3.0)

# 可訓練變數
w = tf.Variable(2.0, trainable=True)

print("x:", x.numpy())
print("w:", w.numpy())

# 更新變數
w.assign(5.0)

print("Updated w:", w.numpy())

# 在原有數值上加 2
w.assign_add(2.0)

print("After add:", w.numpy())


print("\n=== GradientTape ===")

# 建立模型權重
w = tf.Variable(0.0)

# 開始記錄可微分運算
with tf.GradientTape() as tape:

    # Forward：計算 Loss
    loss = (w - 5.0) ** 2

# Backward：計算梯度
gradient = tape.gradient(loss, w)

print("Weight:", w.numpy())
print("Loss:", loss.numpy())
print("Gradient:", gradient.numpy())


print("\n=== Chain Rule ===")

x = tf.constant(3.0)

w = tf.Variable(2.0)
b = tf.Variable(1.0)

with tf.GradientTape() as tape:
    z = w * x + b
    loss = z ** 2

grad_w, grad_b = tape.gradient(
    loss, [w, b]
)

print("z:", z.numpy())
print("Loss:", loss.numpy())
print("Gradient w:", grad_w.numpy())
print("Gradient b:", grad_b.numpy())

print("\n=== Linear Regression Training ===")

# 1. 建立訓練資料
x_train = tf.constant(
    [1.0, 2.0, 3.0, 4.0],
    dtype=tf.float32
)

y_train = tf.constant(
    [3.0, 5.0, 7.0, 9.0],
    dtype=tf.float32
)

# 2. 初始化模型參數
w = tf.Variable(0.0)
b = tf.Variable(0.0)

# 3. 設定學習率
learning_rate = 0.05

# 4. 訓練 100 個 Epoch
for epoch in range(100):

    # Forward + Loss
    with tf.GradientTape() as tape:

        predictions = w * x_train + b

        error = predictions - y_train

        loss = tf.reduce_mean(
            tf.square(error)
        )

    # Backward
    gradients = tape.gradient(
        loss, [w, b]
    )

    grad_w, grad_b = gradients

    # 更新參數
    w.assign_sub(learning_rate * grad_w)
    b.assign_sub(learning_rate * grad_b)

    # 每 10 個 Epoch 顯示一次
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss: {loss.numpy():.6f} | "
            f"w: {w.numpy():.4f} | "
            f"b: {b.numpy():.4f}"
        )

print("Final Weight:", w.numpy())
print("Final Bias:", b.numpy())


print("\n=== tf.function ===")

@tf.function
def calculate(x):
    return x * x + 2.0

a = tf.constant(3.0)
b = tf.constant(5.0)

result_a = calculate(a)
result_b = calculate(b)

print("Result A:", result_a.numpy())
print("Result B:", result_b.numpy())
