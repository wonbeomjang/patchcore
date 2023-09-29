import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor


class KCenterGreedy:
    def __init__(self, ratio: float = 0.1):
        self.ratio = ratio
        self._min_distance: Tensor = torch.zeros((0,))

    def select_index(self, embeddings: Tensor):
        coreset_indexes = []
        coreset_size = int(embeddings.size(0) * self.ratio)

        center_index = torch.randint(low=0, high=embeddings.size(0), size=(1,))

        for i in tqdm.tqdm(range(coreset_size - 1), total=coreset_size - 1):
            center_index = self.greedy_coreset_seleection(center_index, embeddings)
            coreset_indexes += [center_index]
        return coreset_indexes

    def greedy_coreset_seleection(self, center_index, embeddings):
        self.calculate_min_distance(embeddings, center_index)
        center_index = self.get_next_center_index()

        return center_index

    def calculate_min_distance(self, embeddings: Tensor, center_index: Tensor):
        center_embedding: Tensor = embeddings[center_index]
        distances = F.pairwise_distance(center_embedding, embeddings, p=2)
        self._min_distance = (
            distances
            if self._min_distance.shape != distances.shape
            else torch.minimum(self._min_distance, distances)
        )

    def get_next_center_index(self):
        center_index = int(torch.argmax(self._min_distance).item())
        self._min_distance[center_index] = 0
        return center_index

    def __call__(self, embeddings, *args, **kwargs):
        indexes = self.select_index(embeddings)
        coreset = embeddings[indexes]
        return coreset
