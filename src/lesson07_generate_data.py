
from pathlib import Path

import numpy as np
import tensorflow as tf

# 資料集根目錄
DATA_ROOT = Path("dataset")

# 固定隨機種子
rng = np.random.default_rng(42)

# 用不同群組切分 Train / Val / Test
SPLITS = {
    "train": range(0, 8),
    "val": range(8, 10),
    "test": range(10, 12),
}

CLASSES = ("normal", "abnormal")

def create_image(label, group_id, sample_id):
    """建立一張 RGB uint8 示範影像。"""

    height, width = 160, 160

    # 建立灰色背景
    image = np.full(
        (height, width, 3),
        100 + group_id,
        dtype=np.uint8
    )

    # 模擬工件：中央矩形
    image[30:130, 25:135, :] = 175

    # 模擬工件內部結構
    image[45:115, 40:120, :] = 205

    # 正常與異常的特徵差異
    if label == "abnormal":
        image[75:94, 70:90, :] = [230, 35, 35]

    # 增加少量影像雜訊
    noise = rng.integers(
        -8, 9,
        size=image.shape,
        dtype=np.int16
    )

    image = np.clip(
        image.astype(np.int16) + noise,
        0,
        255
    ).astype(np.uint8)

    return image


# 建立資料集
total = 0

for split, groups in SPLITS.items():

    for label in CLASSES:

        folder = DATA_ROOT / split / label
        folder.mkdir(parents=True, exist_ok=True)

        # 每組每類別產生 3 張影像
        for group_id in groups:
            for sample_id in range(3):

                image = create_image(
                    label,
                    group_id,
                    sample_id
                )

                filename = (
                    f"group_{group_id:02d}_"
                    f"sample_{sample_id:02d}.png"
                )

                path = folder / filename

                # NumPy -> TensorFlow Tensor
                tensor = tf.convert_to_tensor(image)

                # 將 RGB 像素編碼成 PNG
                encoded = tf.io.encode_png(tensor)

                # 將 PNG Bytes 寫入硬碟
                tf.io.write_file(
                    str(path),
                    encoded
                )

                total += 1

print("Dataset generated.")
print("Total images:", total)
