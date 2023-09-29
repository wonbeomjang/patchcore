import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models import get_model
from torchvision.models.feature_extraction import create_feature_extractor

from config import PatchCoreConfig
from models.projection import SparseRandomProjection
from models.sampling import KCenterGreedy
from utils.math import euclidean_dist


class PatchCore(nn.Module):
    def __init__(
        self, patch_core_config: PatchCoreConfig = PatchCoreConfig(), *args, **kwargs
    ):
        super(PatchCore, self).__init__(*args, **kwargs)

        return_node: dict[str, str] = {
            name: str(i)
            for i, name in enumerate(list(patch_core_config.backbone.return_node))
            if i in patch_core_config.layer_num
        }

        self.feature_extractor: nn.Module = create_feature_extractor(
            get_model(patch_core_config.backbone.model_id, pretrained=True),
            return_node,
        )
        self.feature_pool: nn.Module = torch.nn.AvgPool2d(3, 1, 1)
        self.projection = SparseRandomProjection()
        self.sampler = KCenterGreedy()
        self.embeddings: list[Tensor] = []

        self.num_neighbors = patch_core_config.num_neighbors

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            x: dict[str, Tensor] = self.feature_extractor(x)
            embedding = self.generate_embeddings(x)
            batch_size, _, width, height = embedding.shape
            embedding = self.reshape_embedding(embedding)
            embedding = self.project_embedding(embedding)

            if self.training:
                self.embeddings += [embedding]

                return embedding
            else:
                patch_scores, locations = self.get_nearest_neighbors(
                    embedding, num_neighbors=1
                )
                patch_scores = patch_scores.reshape((batch_size, -1))
                locations = locations.reshape((batch_size, -1))
                pred_score = self.compute_anomaly_score(
                    patch_scores, locations, embedding
                )
                # reshape to w, h
                patch_scores = patch_scores.reshape((batch_size, 1, width, height))
                # get anomaly map
                # anomaly_map = self.anomaly_map_generator(patch_scores)

                return pred_score

    def generate_embeddings(self, x: dict[str, Tensor]) -> Tensor:
        target_shape = x[min(x.keys())].shape[-2:]
        for k in x.keys():
            x[k] = self.feature_pool(x[k])
            x[k] = F.interpolate(x[k], size=target_shape, mode="bilinear")
        x: Tensor = torch.cat(list(x.values()), dim=1)
        return x

    def compute_anomaly_score(
        self, patch_scores: Tensor, locations: Tensor, embedding: Tensor
    ) -> Tensor:
        if self.num_neighbors == 1:
            return patch_scores.amax(1)

        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(
            patch_scores, dim=1
        )  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[
            torch.arange(batch_size), max_patches
        ]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[
            torch.arange(batch_size), max_patches
        ]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.embeddings[0][nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.embeddings[0].shape[
            0
        ]  # edge case when memory bank is too small
        _, support_samples = self.get_nearest_neighbors(
            nn_sample, num_neighbors=min(self.num_neighbors, memory_bank_effective_size)
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = euclidean_dist(
            max_patches_features.unsqueeze(1), self.embeddings[0][support_samples]
        )
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        return score

    def get_nearest_neighbors(
        self, embedding: Tensor, num_neighbors: int = 9
    ) -> tuple[Tensor, Tensor]:
        assert len(self.embeddings) == 1
        distance = euclidean_dist(embedding, self.embeddings[0])
        patch_scores, location = torch.topk(
            distance, k=num_neighbors, largest=False, dim=1
        )
        return patch_scores, location

    def project_embedding(self, embedding: Tensor) -> Tensor:
        self.projection.fit(embedding)
        embedding = self.projection.transform(embedding)
        return embedding

    def reshape_embedding(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
        return x

    def make_coreset(self) -> list[Tensor]:
        self.embeddings = [torch.cat(self.embeddings, dim=0)]
        indexes = self.sampler.select_index(self.embeddings[0])
        self.embeddings = [self.embeddings[0][indexes]]

        return self.embeddings


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PatchCore().to(device)
    image = torch.randn((1, 3, 224, 224)).to(device)

    model(image)
    model(image)
    model(image)
    model(image)
    model.make_coreset()

    model.eval()
    res = model(image)
    print(model(image))
    print(model(image))
    print(model(image))
    print(model(image))
