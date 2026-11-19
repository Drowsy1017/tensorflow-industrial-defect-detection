
from pathlib import Path
import csv

import tensorflow as tf

from .config import CFG


# =====================================
# 1. 讀取 CSV Manifest
# =====================================

def read_manifest(split):

    csv_path = CFG.manifest_dir / f"{split}.csv"

    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split: {split}")

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    paths = []
    labels = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            path = Path(row["path"])
            label = int(row["label"])

            if not path.is_file():
                raise FileNotFoundError(path)

            if label not in (0, 1):
                raise ValueError(
                    f"Invalid label: {label}"
                )

            paths.append(str(path))
            labels.append(label)

    if not paths:
        raise ValueError(
            f"Empty manifest: {csv_path}"
        )

    return paths, labels


# =====================================
# 2. 影像讀取與前處理
# =====================================

def preprocess_image(path, label):

    # 讀取 PNG Bytes
    raw = tf.io.read_file(path)

    # 解碼成 RGB 圖片
    image = tf.io.decode_png(
        raw,
        channels=3
    )

    # Resize
    image = tf.image.resize(
        image,
        CFG.image_size
    )

    # 轉換成 float32
    image = tf.cast(
        image,
        tf.float32
    )

    # MobileNetV2 正規化：0~255 → -1~1
    image = image / 127.5 - 1.0

    # 固定 Shape
    image = tf.ensure_shape(
        image,
        (
            CFG.image_height,
            CFG.image_width,
            CFG.channels
        )
    )

    # Label Shape: (1,)
    label = tf.reshape(
        tf.cast(label, tf.float32),
        (1,)
    )

    return image, label


# =====================================
# 3. 建立 TensorFlow Dataset
# =====================================

def make_dataset(split):

    paths, labels = read_manifest(split)

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    # 只打亂訓練集
    if split == "train":
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=CFG.seed,
            reshuffle_each_iteration=True
        )

    # 平行載入與處理圖片
    ds = ds.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 組成 Batch
    ds = ds.batch(
        CFG.batch_size
    )

    # 預先準備下一批
    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


# =====================================
# 4. 測試 Dataset
# =====================================

def main():

    print("TensorFlow:", tf.__version__)

    print(
        "GPU:",
        tf.config.list_physical_devices("GPU")
    )

    for split in ("train", "val", "test"):

        paths, labels = read_manifest(split)

        print(f"\n=== {split.upper()} ===")
        print("Images:", len(paths))
        print("Normal:", labels.count(0))
        print("Defect:", labels.count(1))

        ds = make_dataset(split)

        # 取第一個 Batch
        images, batch_labels = next(iter(ds))

        print("Image Shape:", images.shape)
        print("Label Shape:", batch_labels.shape)
        print("Image Dtype:", images.dtype)

        print(
            "Pixel Min:",
            float(tf.reduce_min(images).numpy())
        )

        print(
            "Pixel Max:",
            float(tf.reduce_max(images).numpy())
        )

        # 檢查有沒有 NaN / Inf
        tf.debugging.assert_all_finite(
            images,
            "Invalid image values"
        )

        # 檢查正規化範圍
        tf.debugging.assert_greater_equal(
            images,
            -1.0
        )

        tf.debugging.assert_less_equal(
            images,
            1.0
        )

    print("\nDataset smoke test passed.")


if __name__ == "__main__":
    main()
