
import json

import numpy as np
import tensorflow as tf

from .config import CFG
from .dataset import make_dataset


MODEL_PATHS = {
    "stage1": CFG.model_dir / "mobilenetv2_stage1.keras",
    "finetuned": CFG.model_dir / "mobilenetv2_finetuned.keras",
}

THRESHOLD = 0.5


def calculate_metrics(y_true, y_prob):

    y_pred = (y_prob >= THRESHOLD).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    pr_auc = tf.keras.metrics.AUC(
        curve="PR",
        num_thresholds=200
    )
    pr_auc.update_state(y_true, y_prob)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc.result().numpy()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():

    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("Dataset: Validation")
    print("Threshold:", THRESHOLD)

    val_ds = make_dataset("val")

    results = {}

    for name, model_path in MODEL_PATHS.items():

        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        print(f"\nLoading {name}: {model_path}")

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        y_true_parts = []
        y_prob_parts = []

        for images, labels in val_ds:

            probabilities = model(
                images,
                training=False
            )

            y_true_parts.append(
                labels.numpy().reshape(-1)
            )
            y_prob_parts.append(
                probabilities.numpy().reshape(-1)
            )

        y_true = np.concatenate(y_true_parts).astype(np.int32)
        y_prob = np.concatenate(y_prob_parts)

        metrics = calculate_metrics(y_true, y_prob)
        results[name] = metrics

        print(f"\n=== {name.upper()} ===")
        print(json.dumps(metrics, indent=2))

    print("\n=== DIFFERENCE (FINETUNED - STAGE1) ===")

    for metric in (
        "accuracy", "precision", "recall",
        "f1", "pr_auc"
    ):
        difference = (
            results["finetuned"][metric]
            - results["stage1"][metric]
        )
        print(f"{metric}: {difference:+.4f}")

    CFG.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = CFG.output_dir / "model_comparison.json"

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nComparison completed.")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
