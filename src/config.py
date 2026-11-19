
from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:

    # 資料路徑
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    manifest_dir: Path = PROJECT_ROOT / "data" / "manifests"

    # 輸出路徑
    model_dir: Path = PROJECT_ROOT / "models"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # 影像設定
    image_height: int = 192
    image_width: int = 512
    channels: int = 3

    # 訓練設定
    batch_size: int = 8
    seed: int = 42
    validation_ratio: float = 0.15

    # 分類設定
    class_names: tuple[str, str] = ("normal", "defect")

    # 模型
    model_name: str = "mobilenet_v2"

    @property
    def image_size(self):
        return (self.image_height, self.image_width)


CFG = Config()


if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Raw dataset:", CFG.raw_dir)
    print("Image size:", CFG.image_size)
    print("Batch size:", CFG.batch_size)
    print("Classes:", CFG.class_names)
    print("Model:", CFG.model_name)
