import os
import random

import numpy as np
import torch
from torch import Tensor
import sklearn

from config import DataConfig
from data.datasets import get_train_loader, get_val_loader
from models.patchcore import PatchCore

random_seed = 2023
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class ThresholdAdaptor:
    def __init__(self, not_defect_name: str = "good"):
        self.scores: list[float] = []
        self.is_normal: list[int] = []
        self.not_defect_name: str = not_defect_name

    def __call__(
        self, score: list[float] | Tensor, defect_types: list[str], *args, **kwargs
    ):
        if isinstance(score, Tensor):
            score = score.cpu().tolist()
        self.scores += score
        self.is_normal += [
            int(defect_type == self.not_defect_name) for defect_type in defect_types
        ]

    def calc_threshold(self):
        assert len(self.scores) == len(self.is_normal)

        score_map = sorted(list(zip(self.scores, self.is_normal)))
        self.scores, self.is_normal = list(map(list, zip(*score_map)))

        max_f1_score = 0
        target_index = 0
        for i in range(len(self.scores)):
            normal_preds = [1] * i + [0] * (len(self.scores) - i)
            f1_score = sklearn.metrics.f1_score(self.is_normal, normal_preds)
            if f1_score > max_f1_score:
                max_f1_score = f1_score
                target_index = i

        f1_score = self.scores[target_index]
        self.reset()
        return f1_score, max_f1_score

    def reset(self):
        self.scores: list[float] = []
        self.is_normal: list[bool] = []


def calc(dataloader, threshold_adaptor, model, device):
    for i, (image, defect_type) in enumerate(dataloader):
        image = image.to(device)
        res = model(image)
        score = res["score"]
        threshold_adaptor(score, defect_type)


def train(dataset_config: DataConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Start {dataset_config.category}")
    dataloader = get_train_loader(dataset_config)
    model = PatchCore().to(device)

    for i, (image, defect_type) in enumerate(dataloader):
        image = image.to(device)
        model(image)

    model.make_coreset()
    model.eval()

    threshold_adaptor = ThresholdAdaptor()

    for defect_type in os.listdir(
        os.path.join(dataset_config.base_dir, dataset_config.category, "test")
    ):
        dataset_config.defect_type = defect_type
        calc(get_val_loader(dataset_config), threshold_adaptor, model, device)

    threshold, f1_score = threshold_adaptor.calc_threshold()
    print(
        f"Category: {dataset_config.category} F1 Score: {f1_score}, Threshold: {threshold}"
    )

    return f1_score, threshold


if __name__ == "__main__":
    dataset_config = DataConfig()
    f1_score_sum = 0
    cnt = 0

    for category in os.listdir(dataset_config.base_dir):
        if os.path.isfile(os.path.join(dataset_config.base_dir, category)):
            continue
        dataset_config.category = category
        f1_score, _ = train(dataset_config)
        f1_score_sum += f1_score
        cnt += 1

    print(f"Total F1 Score: {f1_score_sum / cnt}")
