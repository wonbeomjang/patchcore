import argparse
import os

import torch
from torch import Tensor
from tqdm import tqdm
from config import DataConfig
from data.datasets import get_train_loader, get_val_loader
from model import PatchCore


class ThresholdAdaptor:
    def __init__(self, not_defect_name: str = "good"):
        self.scores: list[float] = []
        self.is_defect: list[bool] = []
        self.not_defect_name : str = not_defect_name

    def __call__(self, score: list[float] | Tensor, defect_types: list[str], *args, **kwargs):
        if isinstance(score, Tensor):
            score = score.cpu().tolist()
        self.scores += score
        self.is_defect += [defect_type == self.not_defect_name for defect_type in defect_types]

    def calc_threshold(self):
        assert len(self.scores) == len(self.is_defect)

        score_map = sorted(list(zip(self.scores, self.is_defect)))
        accuracy = 0
        num_correct = 0
        threshold = 0

        for i, (score, is_defect) in enumerate(score_map):
            num_correct += is_defect
            acc = num_correct / (i + 1)

            if accuracy < acc:
                threshold = score
                accuracy = acc

        self.reset()
        return threshold, accuracy

    def reset(self):
        self.scores: list[float] = []
        self.is_defect: list[bool] = []


def calc(dataloader, threshold_adaptor):
    for i, (image, defect_type) in enumerate(dataloader):
        image = image.to(device)
        res = model(image)
        score = res["score"]
        threshold_adaptor(score, defect_type)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--category", type=str)

    config = args.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_config = DataConfig()
    dataset_config.category = config.category

    print(f"Start {dataset_config.category}")
    dataloader = get_train_loader(dataset_config)
    model = PatchCore().to(device)

    for i, (image, defect_type) in enumerate(dataloader):
        image = image.to(device)
        model(image)

    model.make_coreset()
    model.save_coreset(f"{dataset_config.category}.pt")
    model.load_coreset(f"{dataset_config.category}.pt")
    model.eval()

    threshold_adaptor = ThresholdAdaptor()
    calc(dataloader, threshold_adaptor)

    for defect_type in os.listdir(os.path.join(dataset_config.base_dir, dataset_config.category, "test")):
        dataset_config.defect_type = defect_type
        calc(get_val_loader(dataset_config), threshold_adaptor)

    threshold, accuracy = threshold_adaptor.calc_threshold()
    print(f"Category: {dataset_config.category} Accuracy: {accuracy}, Threshold: {threshold}")