import dataclasses
import os.path


@dataclasses.dataclass
class Backbone:
    model_id: str = "resnet50"
    return_node: list[str] = dataclasses.field(
        default_factory=lambda: [
            "conv1",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]
    )


@dataclasses.dataclass
class PatchCoreConfig:
    backbone: Backbone = Backbone()
    layer_num: list[int] = dataclasses.field(default_factory=lambda: [2, 3])
    num_neighbors: int = 9


@dataclasses.dataclass
class DataConfig:
    base_dir: str = os.path.join("..", "datasets", "mvtec")
    category: str = "bottle"
    defect_type: str = "broken_large"
    image_size: int = 256
    batch_size: int = 64
    num_workers: int = 0
