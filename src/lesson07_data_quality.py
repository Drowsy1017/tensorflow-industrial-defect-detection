
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import re

import tensorflow as tf


# ==========================================
# 1. 基本設定
# ==========================================

DATA_ROOT = Path("dataset")

SPLITS = ("train", "val", "test")
CLASSES = ("normal", "abnormal")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ==========================================
# 2. 計算檔案 SHA-256
# ==========================================

def file_sha256(path):
    """計算檔案的 SHA-256 雜湊值。"""

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


# ==========================================
# 3. 檢查單張影像
# ==========================================

def check_image(path):
    """驗證圖片能否正常解碼。"""

    raw = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        raw,
        channels=3,
        expand_animations=False
    )

    shape = tf.shape(image).numpy()

    height = int(shape[0])
    width = int(shape[1])
    channels = int(shape[2])

    if height <= 0 or width <= 0:
        raise ValueError("Invalid image size")

    if channels != 3:
        raise ValueError("Image is not RGB")

    return height, width, channels


# ==========================================
# 4. 掃描 Dataset
# ==========================================

def main():

    all_hashes = defaultdict(list)
    all_groups = {}
    total_images = 0

    for split in SPLITS:

        print(f"\n=== {split.upper()} ===")

        counts = Counter()
        group_ids = set()

        for class_name in CLASSES:

            folder = DATA_ROOT / split / class_name

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Folder not found: {folder}"
                )

            paths = sorted(
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            for path in paths:

                # 4-1. 圖片解碼檢查
                try:
                    h, w, c = check_image(path)

                except (tf.errors.OpError, ValueError) as exc:
                    raise RuntimeError(
                        f"Corrupted image: {path}"
                    ) from exc

                # 4-2. 檔案雜湊檢查
                digest = file_sha256(path)
                all_hashes[digest].append(
                    (split, class_name, str(path))
                )

                # 4-3. 從檔名取得 Group ID
                match = re.fullmatch(
                    r"group_(\d+)_sample_(\d+)",
                    path.stem
                )

                if not match:
                    raise ValueError(
                        f"Unexpected filename: {path}"
                    )

                group_id = match.group(1)
                group_ids.add(group_id)

                counts[class_name] += 1
                total_images += 1

        all_groups[split] = group_ids

        print("Class counts:", dict(counts))
        print("Total:", sum(counts.values()))
        print("Groups:", sorted(group_ids))

        if counts["normal"] != counts["abnormal"]:
            print("WARNING: Class imbalance detected")


    # ======================================
    # 5. 檢查跨資料集重複圖片
    # ======================================

    print("\n=== Duplicate Check ===")

    duplicate_count = 0
    leakage_found = False

    for digest, records in all_hashes.items():

        if len(records) <= 1:
            continue

        duplicate_count += 1

        print("Duplicate files:")
        for record in records:
            print(" ", record[2])

        involved_splits = {
            record[0] for record in records
        }

        if len(involved_splits) > 1:
            leakage_found = True
            print("WARNING: Cross-split duplicate")


    # ======================================
    # 6. 檢查 Group Leakage
    # ======================================

    print("\n=== Group Leakage Check ===")

    for i, split_a in enumerate(SPLITS):

        for split_b in SPLITS[i + 1:]:

            overlap = (
                all_groups[split_a]
                & all_groups[split_b]
            )

            if overlap:
                leakage_found = True

                print(
                    f"WARNING: {split_a} / {split_b}",
                    overlap
                )

    print("\n=== Summary ===")
    print("Total images:", total_images)
    print("Duplicate hash groups:", duplicate_count)
    print("Leakage detected:", leakage_found)

    if leakage_found:
        raise RuntimeError(
            "Data leakage was detected!"
        )

    print("Data quality checks completed.")


if __name__ == "__main__":
    main()
