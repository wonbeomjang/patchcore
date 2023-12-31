import glob
import os
import tarfile

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from config import DataConfig
from util.url import download_with_progressbar


def download_dataset(path: str = os.path.join("..", "datasets")):
    url = (
        "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/"
        "420938113-1629952094/mvtec_anomaly_detection.tar.xz"
    )
    if not os.path.exists(path):
        os.makedirs(path)
    download_with_progressbar(url, os.path.join(path, "mvtec.tar.xz"))
    with tarfile.open(os.path.join(path, "mvtec.tar.xz")) as f:
        f.extractall(path)


class MVTecDataset(Dataset):
    def __init__(
        self,
        transform: A.Compose,
        base_dir: str,
        category: str,
        set_name: str = "train",
        defect_type: str = "good",
        download: bool = True,
    ):
        if not os.path.exists(base_dir) and download:
            download_dataset(base_dir)

        self.image_paths = glob.glob(
            os.path.join(base_dir, category, set_name, defect_type, "*")
        )
        self.transforms = transform
        self.defect_type = defect_type

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]

        return image, self.defect_type

    def __len__(self):
        return len(self.image_paths)


def get_train_loader(dataset_config: DataConfig):
    transform = A.Compose(
        [
            A.Resize(
                height=dataset_config.image_size,
                width=dataset_config.image_size,
                always_apply=True,
            ),
            A.CenterCrop(
                height=dataset_config.center_crop,
                width=dataset_config.center_crop,
                always_apply=True,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset = MVTecDataset(transform, dataset_config.base_dir, dataset_config.category)
    dataloader = DataLoader(
        dataset,
        dataset_config.batch_size,
        False,
        num_workers=dataset_config.num_workers,
    )

    return dataloader


def get_val_loader(dataset_config: DataConfig):
    transform = A.Compose(
        [
            A.Resize(
                height=dataset_config.image_size,
                width=dataset_config.image_size,
                always_apply=True,
            ),
            A.CenterCrop(
                height=dataset_config.center_crop,
                width=dataset_config.center_crop,
                always_apply=True,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset = MVTecDataset(
        transform,
        dataset_config.base_dir,
        dataset_config.category,
        "test",
        dataset_config.defect_type,
    )
    dataloader = DataLoader(
        dataset,
        1,
        False,
        num_workers=dataset_config.num_workers,
    )

    return dataloader
