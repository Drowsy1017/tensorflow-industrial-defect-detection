
import tensorflow as tf

print("=== TensorFlow Lesson 02 ===")

# 1. 建立 Tensor
scalar = tf.constant(10)

vector = tf.constant([1, 2, 3])

matrix = tf.constant([
    [1, 2, 3],
    [4, 5, 6]
])

# 2. 觀察資料
print("Scalar:", scalar.numpy())
print("Vector:", vector.numpy())
print("Matrix:\n", matrix.numpy())

# 3. Shape / Rank / Size
print("Shape:", matrix.shape)
print("Rank:", tf.rank(matrix).numpy())
print("Size:", tf.size(matrix).numpy())

# 4. Dtype
print("Dtype:", matrix.dtype)

# 5. Reshape
reshaped = tf.reshape(matrix, (3, 2))

print("Reshaped:\n", reshaped.numpy())

# 6. Transpose
transposed = tf.transpose(matrix)

print("Transposed:\n", transposed.numpy())


print("\n=== Broadcasting ===")

# A 是 3 列 1 欄
a = tf.constant([
    [1],
    [2],
    [3]
])

# B 是 1 列 3 欄
b = tf.constant([
    [10, 20, 30]
])

# 使用 Broadcasting 相加
c = a + b

print("A Shape:", a.shape)
print("B Shape:", b.shape)
print("C Shape:", c.shape)
print("Result:\n", c.numpy())


print("\n=== Matrix Multiplication ===")

# 建立矩陣 A
a = tf.constant([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# 建立矩陣 B
b = tf.constant([
    [7.0, 8.0],
    [9.0, 10.0],
    [11.0, 12.0]
])

# 進行矩陣乘法
c = tf.matmul(a, b)

print("A Shape:", a.shape)
print("B Shape:", b.shape)
print("C Shape:", c.shape)

print("Result:\n", c.numpy())


print("\n=== GPU Matrix Test ===")

# 取得 TensorFlow 可使用的 GPU
gpus = tf.config.list_physical_devices("GPU")

print("Available GPUs:", gpus)

if gpus:

    # 指定第一張 GPU
    with tf.device("/GPU:0"):

        # 建立兩個大型浮點矩陣
        a = tf.random.normal((1024, 1024))
        b = tf.random.normal((1024, 1024))

        # GPU 矩陣乘法
        result = tf.matmul(a, b)

    print("Result Shape:", result.shape)
    print("Device:", result.device)

else:
    print("No GPU detected")
