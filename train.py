import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from torch import Tensor, nn
import sklearn

from config import DataConfig, PatchCoreConfig
from data.datasets import get_train_loader, get_val_loader, download_dataset
from models.inference_engine import PatchCoreEngine
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
        self,
        score: list[float] | Tensor | np.ndarray,
        defect_types: list[str],
        *args,
        **kwargs,
    ):
        if isinstance(score, Tensor):
            score = score.cpu().tolist()
        elif isinstance(score, np.ndarray):
            score = score.tolist()

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


def calc(
    dataloader, threshold_adaptor, model: nn.Module, device: torch.device
) -> float:
    total_time = 0
    for i, (image, defect_type) in enumerate(dataloader):
        image = image.to(device)
        cur_time = time.time()
        res = model(image)
        if i != 0:
            total_time += time.time() - cur_time
        score = res["score"]
        threshold_adaptor(score, defect_type)

    return total_time / len(dataloader) * 1000


def train(dataset_config: DataConfig, model: nn.Module, device: torch.device):
    print(f"Start {dataset_config.category}")
    model.train()
    dataloader = get_train_loader(dataset_config)

    for i, (image, defect_type) in enumerate(dataloader):
        image = image.to(device)
        model(image)

    model.make_coreset()


def eval(
    dataset_config: DataConfig, model: nn.Module | PatchCoreEngine, device: torch.device
):
    model.eval()
    threshold_adaptor = ThresholdAdaptor()
    inference_time = 0
    for defect_type in os.listdir(
        os.path.join(dataset_config.base_dir, dataset_config.category, "test")
    ):
        dataset_config.defect_type = defect_type
        inference_time = calc(
            get_val_loader(dataset_config), threshold_adaptor, model, device
        )
    threshold, f1_score = threshold_adaptor.calc_threshold()
    print(
        f"Category: {dataset_config.category} F1 Score: {f1_score:.4f}, Threshold: {threshold:.2f}, "
        f"Inference Time {inference_time:.2f}ms"
    )
    return f1_score, threshold, inference_time


if __name__ == "__main__":
    dataset_config = DataConfig()
    f1_score_sum = 0
    cnt = 0
    patchcore_config = PatchCoreConfig()

    result = {"category": [], "f1_score": [], "inference_time": []}

    if not os.path.exists(dataset_config.base_dir):
        download_dataset(dataset_config.base_dir)

    for category in os.listdir(dataset_config.base_dir):
        if os.path.isfile(os.path.join(dataset_config.base_dir, category)):
            continue

        dataset_config.category = category
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = PatchCore(patchcore_config).to(device)
        train(dataset_config, model, device)
        f1_score, threshold, inference_time = eval(dataset_config, model, device)

        result["category"] += [category]
        result["f1_score"] += [f1_score]
        result["inference_time"] += [inference_time]

        # x = torch.randn((1, 3, 224, 224)).to(device)
        # torch.onnx.export(model,  # 실행될 모델
        #                   x,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
        #                   f"./serve/{dataset_config.category}.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
        #                   export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
        #                   opset_version=12,  # 모델을 변환할 때 사용할 ONNX 버전
        #                   do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
        #                   input_names=['input'],  # 모델의 입력값을 가리키는 이름
        #                   output_names=['output'],  # 모델의 출력값을 가리키는 이름
        #                   dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
        #                                 'output': {0: 'batch_size'}})
        #
        # patchcore_config.backbone.onnx_path = f"./serve/{dataset_config.category}.onnx"
        # model = PatchCoreEngine(patchcore_config)
        # eval(dataset_config, model, device)
        #
        # patchcore_config.backbone.trt_path = f"./serve/{dataset_config.category}.trt"
        # patchcore_config.backbone.inference_engin = InferenceEngine.trt
        # model = PatchCoreEngine(patchcore_config)
        # eval(dataset_config, model, device)

    if not os.path.exists("result"):
        os.makedirs("result")
    df = pandas.DataFrame.from_dict(result)
    print(df.describe())
    df.plot(kind="bar", title=patchcore_config.backbone.model)
    plt.show()
    df.to_csv(os.path.join("result", f"{patchcore_config.backbone.model}-dino.csv"))
