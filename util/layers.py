import timm
import torch
from torch import Tensor, nn


class TimmFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: list[str], weight_url: str | None = None):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.idx = self._map_layer_to_idx()
        self.feature_extractor: torch.nn.Module = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            exportable=True,
            out_indices=self.idx,
        )
        self.feature_pool: nn.Module = torch.nn.AvgPool2d(3, 1, 1)

        if weight_url is not None:
            state_dict = torch.hub.load_state_dict_from_url(weight_url)
            self.feature_extractor.load_state_dict(state_dict)

    def _map_layer_to_idx(self, offset: int = 3) -> list[int]:
        """Maps set of layer names to indices of model.

        Args:
            offset (int) `timm` ignores the first few layers when indexing please update offset based on need

        Returns:
            Feature map extracted from the CNN
        """
        idx = []
        features = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=False,
            exportable=True,
        )
        for i in self.layers:
            try:
                idx.append(
                    list(dict(features.named_children()).keys()).index(i) - offset
                )
            except ValueError:
                self.layers.remove(i)

        return idx

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        self.feature_extractor.eval()
        self.feature_pool.eval()
        with torch.no_grad():
            features = dict(
                zip(
                    self.layers,
                    map(self.feature_pool, (self.feature_extractor(inputs))),
                )
            )
        return features
