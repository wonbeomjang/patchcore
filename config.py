import dataclasses
import enum
import os.path


class InferenceEngine(enum.Enum):
    trt = enum.auto()
    onnx = enum.auto()


@dataclasses.dataclass
class Backbone:
    model: str = "resnet50"
    weight_url: str | None = "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
    inference_engin: str = InferenceEngine.onnx
    onnx_path: str = "wide_resnet50_2.onnx"
    trt_path: str = "wide_resnet50_2.trt"
    return_layer: list[str] = dataclasses.field(
        default_factory=lambda: ["layer2", "layer3"]
    )


@dataclasses.dataclass
class PatchCoreConfig:
    backbone: Backbone = Backbone()
    num_neighbors: int = 9
    sampling_ratio: float = 0.01


@dataclasses.dataclass
class DataConfig:
    base_dir: str = os.path.join("..", "datasets", "mvtec")
    category: str = "bottle"
    defect_type: str = "broken_large"
    image_size: int = 256
    center_crop: int = 224
    batch_size: int = 32
    num_workers: int = 16
