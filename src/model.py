
import tensorflow as tf

from .config import CFG
from .dataset import make_dataset


def build_model():
    # 1. 定義輸入
    inputs = tf.keras.Input(
        shape=(
            CFG.image_height,
            CFG.image_width,
            CFG.channels
        ),
        name="image_input"
    )

    # 2. 載入 ImageNet 預訓練模型
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(
            CFG.image_height,
            CFG.image_width,
            CFG.channels
        ),
        include_top=False,
        weights="imagenet",
        alpha=1.0
    )

    # 3. 凍結預訓練權重
    backbone.trainable = False

    # 4. 擷取影像特徵
    x = backbone(inputs, training=False)

    # 5. 將 Feature Map 轉成特徵向量
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # 6. 降低過擬合風險
    x = tf.keras.layers.Dropout(0.3)(x)

    # 7. 二元分類輸出
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="defect_probability"
    )(x)

    # 8. 建立完整模型
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="industrial_defect_mobilenetv2"
    )

    return model


def main():
    print("TensorFlow:", tf.__version__)
    print(
        "GPU:",
        tf.config.list_physical_devices("GPU")
    )

    model = build_model()
    model.summary()

    # 取一批真實工業影像
    train_ds = make_dataset("train")
    images, labels = next(iter(train_ds))

    # 只執行 Forward，不更新權重
    predictions = model(
        images,
        training=False
    )

    print("Input shape:", images.shape)
    print("Output shape:", predictions.shape)
    print("Predictions:", predictions.numpy()[:5])

    print(
        "Backbone trainable:",
        model.get_layer("mobilenetv2_1.00_192").trainable
        if "mobilenetv2_1.00_192" in
        [layer.name for layer in model.layers]
        else "Check model summary"
    )

    print("Model forward test passed.")


if __name__ == "__main__":
    main()
