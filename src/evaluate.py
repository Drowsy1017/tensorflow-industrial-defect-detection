
import json

import numpy as np
import tensorflow as tf

from .config import CFG
from .dataset import make_dataset


# =====================================
# 1. 評估設定
# =====================================

MODEL_PATH = (
    CFG.model_dir / "mobilenetv2_stage1.keras"
)

THRESHOLD = 0.5


# =====================================
# 2. 計算分類指標
# =====================================

def calculate_metrics(y_true, y_prob):

    y_pred = (
        y_prob >= THRESHOLD
    ).astype(np.int32)

    y_true = y_true.astype(np.int32)

    tp = int(np.sum(
        (y_true == 1) & (y_pred == 1)
    ))

    tn = int(np.sum(
        (y_true == 0) & (y_pred == 0)
    ))

    fp = int(np.sum(
        (y_true == 0) & (y_pred == 1)
    ))

    fn = int(np.sum(
        (y_true == 1) & (y_pred == 0)
    ))

    total = len(y_true)

    accuracy = (tp + tn) / total

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0.0
    )

    pr_auc_metric = tf.keras.metrics.AUC(
        curve="PR",
        num_thresholds=200
    )

    pr_auc_metric.update_state(
        y_true,
        y_prob
    )

    return {
        "threshold": THRESHOLD,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "pr_auc": float(
            pr_auc_metric.result().numpy()
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total
    }


# =====================================
# 3. 評估指定資料集
# =====================================

def evaluate_split(model, split):

    dataset = make_dataset(split)

    all_labels = []
    all_probabilities = []

    for images, labels in dataset:

        probabilities = model(
            images,
            training=False
        )

        all_labels.append(
            labels.numpy().reshape(-1)
        )

        all_probabilities.append(
            probabilities.numpy().reshape(-1)
        )

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probabilities)

    metrics = calculate_metrics(
        y_true,
        y_prob
    )

    print(f"\n=== {split.upper()} ===")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nConfusion Matrix:")
    print("               Pred Normal  Pred Defect")
    print(
        f"Actual Normal  {metrics['tn']:11d}"
        f"  {metrics['fp']:11d}"
    )
    print(
        f"Actual Defect  {metrics['fn']:11d}"
        f"  {metrics['tp']:11d}"
    )

    return metrics


# =====================================
# 4. 主程式
# =====================================

def main():

    print("TensorFlow:", tf.__version__)

    print(
        "GPU:",
        tf.config.list_physical_devices("GPU")
    )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    results = {}

    # 先查看 Validation
    results["val"] = evaluate_split(
        model,
        "val"
    )

    # 再查看保留的 Test Set
    results["test"] = evaluate_split(
        model,
        "test"
    )

    CFG.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        CFG.output_dir
        / "stage1_evaluation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print("\nEvaluation completed.")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
