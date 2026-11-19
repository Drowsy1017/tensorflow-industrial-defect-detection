
from pathlib import Path
import csv
import random

import tensorflow as tf

try:
    from .config import CFG
except ImportError:
    from config import CFG


def read_label(gt_path: Path) -> int:
    """根據 GT 遮罩是否含非零像素建立標籤。"""

    raw = tf.io.read_file(str(gt_path))

    mask = tf.io.decode_png(
        raw,
        channels=1
    )

    has_defect = tf.reduce_any(mask > 0)

    return int(has_defect.numpy())


def scan_split(split: str):
    """讀取原始圖片及對應 GT 遮罩。"""

    folder = CFG.raw_dir / split

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    records = []

    for image_path in sorted(folder.glob("*.png")):

        # 跳過 Ground Truth 遮罩
        if image_path.stem.endswith("_GT"):
            continue

        gt_path = folder / f"{image_path.stem}_GT.png"

        if not gt_path.is_file():
            raise FileNotFoundError(
                f"Missing GT: {gt_path}"
            )

        label = read_label(gt_path)

        records.append({
            "path": str(image_path.resolve()),
            "label": label
        })

    if not records:
        raise RuntimeError(f"No images found: {folder}")

    return records


def stratified_split(records, validation_ratio, seed):
    """依照類別比例切出 Train 和 Validation。"""

    random_gen = random.Random(seed)

    groups = {0: [], 1: []}

    for record in records:
        groups[record["label"]].append(record)

    train_records = []
    val_records = []

    for label, items in groups.items():

        random_gen.shuffle(items)

        val_count = round(
            len(items) * validation_ratio
        )

        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    random_gen.shuffle(train_records)
    random_gen.shuffle(val_records)

    return train_records, val_records


def save_csv(records, path: Path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["path", "label"]
        )

        writer.writeheader()
        writer.writerows(records)


def summarize(name, records):

    normal = sum(
        record["label"] == 0
        for record in records
    )

    defect = sum(
        record["label"] == 1
        for record in records
    )

    print(
        f"{name}: total={len(records)}, "
        f"normal={normal}, defect={defect}"
    )


def main():

    print("Scanning KolektorSDD2...")

    official_train = scan_split("train")
    official_test = scan_split("test")

    # 檢查官方資料數量
    if len(official_train) != 2331:
        raise RuntimeError(
            f"Unexpected train count: "
            f"{len(official_train)}"
        )

    if len(official_test) != 1004:
        raise RuntimeError(
            f"Unexpected test count: "
            f"{len(official_test)}"
        )

    train_records, val_records = stratified_split(
        official_train,
        CFG.validation_ratio,
        CFG.seed
    )

    save_csv(
        train_records,
        CFG.manifest_dir / "train.csv"
    )

    save_csv(
        val_records,
        CFG.manifest_dir / "val.csv"
    )

    save_csv(
        official_test,
        CFG.manifest_dir / "test.csv"
    )

    summarize("Train", train_records)
    summarize("Validation", val_records)
    summarize("Test", official_test)

    print("Dataset manifests created successfully.")


if __name__ == "__main__":
    main()
