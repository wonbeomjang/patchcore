import copy
import dataclasses
import os.path

from torchvision.models import WeightsEnum
from torchvision.models import resnet


@dataclasses.dataclass
class Backbone:
    # model: str = "dino_resnet50"
    # repo_or_dir: str = "facebookresearch/dino:main"
    #
    # model: str = "resnet50"
    model: str = "wide_resnet50_2"
    repo_or_dir: str = "pytorch/vision:v0.10.0"

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
    sampling_ratio: float = 0.01


@dataclasses.dataclass
class DataConfig:
    base_dir: str = os.path.join("..", "datasets", "mvtec")
    category: str = "bottle"
    defect_type: str = "broken_large"
    image_size: int = 256
    batch_size: int = 64
    num_workers: int = 8
