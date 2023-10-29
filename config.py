import dataclasses
import os.path


@dataclasses.dataclass
class Backbone:
    model: str = "wide_resnet50_2"
    weight_url: str | None = None
    return_layer: list[str] = dataclasses.field(default_factory=lambda: ["layer2", "layer3"])


@dataclasses.dataclass
class PatchCoreConfig:
    backbone: Backbone = Backbone()
    num_neighbors: int = 9
    sampling_ratio: float = 0.1


@dataclasses.dataclass
class DataConfig:
    base_dir: str = os.path.join("..", "datasets", "mvtec")
    category: str = "bottle"
    defect_type: str = "broken_large"
    image_size: int = 256
    center_crop: int = 224
    batch_size: int = 32
    num_workers: int = 16
