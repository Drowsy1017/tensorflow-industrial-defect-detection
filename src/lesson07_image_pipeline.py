
from pathlib import Path

import tensorflow as tf


# ==========================================
# 1. 基本設定
# ==========================================

DATA_ROOT = Path("dataset")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

CLASS_TO_ID = {
    "normal": 0,
    "abnormal": 1
}


# ==========================================
# 2. 收集影像路徑及 Label
# ==========================================

def collect_files(split):

    paths = []
    labels = []

    for class_name, label_id in CLASS_TO_ID.items():

        folder = DATA_ROOT / split / class_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        image_files = sorted(
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in (
                ".png", ".jpg", ".jpeg"
            )
        )

        for path in image_files:

            paths.append(str(path))
            labels.append(label_id)

    if not paths:
        raise ValueError(
            f"No images found in {split}"
        )

    return paths, labels


# ==========================================
# 3. 讀取圖片與前處理
# ==========================================

def load_image(path, label):

    # 3-1. 讀取檔案 Bytes
    raw = tf.io.read_file(path)

    # 3-2. 解碼成 RGB uint8 Tensor
    image = tf.io.decode_image(
        raw,
        channels=3,
        expand_animations=False
    )

    # 3-3. 固定通道 Shape
    image = tf.ensure_shape(
        image,
        (None, None, 3)
    )

    # 3-4. Resize 圖片
    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # 3-5. 轉成 float32
    image = tf.cast(
        image,
        tf.float32
    )

    # 3-6. Normalize 到 0~1
    image = image / 255.0

    # 3-7. 固定最終 Shape
    image = tf.ensure_shape(
        image,
        (224, 224, 3)
    )

    # 3-8. Label 轉成 float32
    label = tf.cast(
        label,
        tf.float32
    )

    return image, label


# ==========================================
# 4. 建立 tf.data Pipeline
# ==========================================

def build_dataset(split, training=False):

    paths, labels = collect_files(split)

    print(
        f"{split}: {len(paths)} images"
    )

    # 4-1. 由檔案路徑與標籤建立 Dataset
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    # 4-2. 只打亂訓練資料
    if training:

        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=42,
            reshuffle_each_iteration=True
        )

    # 4-3. 平行讀取與前處理
    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 4-4. 組成 Batch
    ds = ds.batch(BATCH_SIZE)

    # 4-5. 預先載入
    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


# ==========================================
# 5. 檢查 Dataset
# ==========================================

def inspect_dataset(name, dataset):

    print(f"\n=== {name} ===")

    total = 0
    label_counts = {0: 0, 1: 0}

    for images, labels in dataset:

        batch_count = int(
            tf.shape(images)[0].numpy()
        )

        total += batch_count

        for label in labels.numpy().astype(int):
            label_counts[int(label)] += 1

        # 只印出第一個 Batch 的資訊
        if total == batch_count:

            print("Image Shape:", images.shape)
            print("Image Dtype:", images.dtype)

            print("Label Shape:", labels.shape)
            print("Labels:", labels.numpy())

            print(
                "Pixel Min:",
                tf.reduce_min(images).numpy()
            )

            print(
                "Pixel Max:",
                tf.reduce_max(images).numpy()
            )

            # 確保數值正常
            tf.debugging.assert_all_finite(
                images,
                "Image contains NaN or Inf"
            )

            tf.debugging.assert_greater_equal(
                images,
                0.0
            )

            tf.debugging.assert_less_equal(
                images,
                1.0
            )

    print("Total decoded images:", total)
    print("Label counts:", label_counts)


# ==========================================
# 6. 主程式
# ==========================================

def main():

    print("TensorFlow:", tf.__version__)
    print(
        "GPU:",
        tf.config.list_physical_devices("GPU")
    )

    train_ds = build_dataset(
        "train",
        training=True
    )

    val_ds = build_dataset(
        "val"
    )

    test_ds = build_dataset(
        "test"
    )

    inspect_dataset(
        "Training",
        train_ds
    )

    inspect_dataset(
        "Validation",
        val_ds
    )

    inspect_dataset(
        "Test",
        test_ds
    )

    print("\nImage pipeline ready.")


if __name__ == "__main__":
    main()
