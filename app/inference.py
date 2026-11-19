
import json
import sys
from pathlib import Path

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "models" / "inference_config.json"


class DefectPredictor:

    def __init__(self):

        # 1. 讀取推論設定
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.height = int(self.config["input_height"])
        self.width = int(self.config["input_width"])

        self.threshold = float(
            self.config["defect_threshold"]
        )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")

        if self.config["input_channels"] != 3:
            raise ValueError("Only RGB images are supported")

        if self.config["normalization"] != "mobilenet_v2_minus1_to1":
            raise ValueError("Unsupported normalization")

        self.class_names = self.config["class_names"]

        if self.class_names != ["normal", "defect"]:
            raise ValueError("Unexpected class ordering")

        # 2. 載入 Keras 模型
        model_path = (
            PROJECT_ROOT
            / "models"
            / self.config["model_filename"]
        )

        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print(f"Model loaded: {model_path.name}")
        
        print(
    "TensorFlow GPUs:",
    tf.config.list_physical_devices("GPU")
)

    def preprocess(self, image_path: str):

        # 3. 讀取影像
        raw = tf.io.read_file(image_path)

        # 支援 PNG 與 JPEG
        image = tf.io.decode_image(
            raw,
            channels=3,
            expand_animations=False
        )

        # 4. Resize
        image = tf.image.resize(
            image,
            (self.height, self.width)
        )

        # 5. MobileNetV2 正規化
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1.0

        # 6. 增加 Batch 維度
        image = tf.expand_dims(
            image,
            axis=0
        )

        return image

    def predict(self, image_path: str):

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(path)

        image = self.preprocess(str(path))

        # 7. Forward 推論
        prediction = self.model(
            image,
            training=False
        )

        probability = float(
            prediction.numpy()[0][0]
        )

        # 8. Threshold 判斷
        predicted_index = (
            1 if probability >= self.threshold else 0
        )

        return {
            "filename": path.name,
            "predicted_class": self.class_names[predicted_index],
            "defect_score": probability,
            "threshold": self.threshold,
            "is_defect": bool(predicted_index == 1)
        }


def main():

    if len(sys.argv) != 2:
        print(
            "Usage: python -m app.inference "
            "<image_path>"
        )
        sys.exit(1)

    predictor = DefectPredictor()

    result = predictor.predict(
        sys.argv[1]
    )

    print("\n=== PREDICTION RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
