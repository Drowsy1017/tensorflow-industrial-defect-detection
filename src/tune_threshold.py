
import json
import numpy as np
import tensorflow as tf

from .config import CFG
from .dataset import make_dataset


MODEL_PATH = CFG.model_dir / "mobilenetv2_finetuned.keras"
OUTPUT_PATH = CFG.output_dir / "finetuned_threshold_result.json"

TARGET_RECALL = 0.80


def calculate_metrics(y_true, y_prob, threshold):

    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def main():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    val_ds = make_dataset("val")

    labels = []
    probabilities = []

    for images, batch_labels in val_ds:

        output = model(images, training=False)

        labels.extend(batch_labels.numpy().reshape(-1))
        probabilities.extend(output.numpy().reshape(-1))

    y_true = np.asarray(labels, dtype=np.int32)
    y_prob = np.asarray(probabilities, dtype=np.float32)

    if not np.all(np.isfinite(y_prob)):
        raise ValueError("Invalid prediction scores")

    if np.sum(y_true == 1) == 0:
        raise ValueError("Validation set has no defects")

    # 使用實際預測分數作為候選閾值
    thresholds = np.unique(
        np.concatenate(([0.0], y_prob, [1.0]))
    )

    results = [
        calculate_metrics(y_true, y_prob, t)
        for t in thresholds
    ]

    eligible = [
        r for r in results
        if r["recall"] >= TARGET_RECALL
    ]

    best = max(
        eligible,
        key=lambda r: (
            r["f1"],
            r["precision"],
            r["threshold"]
        )
    )

    baseline = calculate_metrics(
        y_true, y_prob, 0.5
    )

    print("\n=== BASELINE (0.5) ===")
    print(json.dumps(baseline, indent=2))

    print("\n=== SELECTED THRESHOLD ===")
    print(json.dumps(best, indent=2))

    CFG.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": str(MODEL_PATH),
                "dataset": "validation",
                "target_recall": TARGET_RECALL,
                "baseline": baseline,
                "selected": best
            },
            f,
            indent=2
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
