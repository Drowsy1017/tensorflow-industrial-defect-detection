import tensorflow as tf

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)

if not gpus:
    raise RuntimeError("GPU not detected!")

with tf.device("/GPU:0"):
    a = tf.random.normal((1024, 1024))
    b = tf.random.normal((1024, 1024))
    result = tf.matmul(a, b)

print("Result shape:", result.shape)
print("Device:", result.device)
print("First value:", result[0, 0].numpy())
