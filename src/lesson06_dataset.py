
import tensorflow as tf
import numpy as np

print("=== Lesson 06: tf.data ===")

# 建立 5 筆資料，每筆 2 個特徵
X = tf.constant([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
    [9.0, 10.0]
], dtype=tf.float32)

# 建立對應標籤
y = tf.constant(
    [0, 1, 0, 1, 0],
    dtype=tf.int32
)

# 轉成 TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (X, y)
)

# 逐筆讀取
for features, label in dataset:
    print(
        "Features:", features.numpy(),
        "Label:", label.numpy()
    )
