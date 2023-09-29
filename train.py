from tqdm import tqdm
from config import DataConfig
from data.datasets import get_train_loader
from model import PatchCore

if __name__ == "__main__":
    dataset_config = DataConfig()
    dataloader = get_train_loader(dataset_config)
    model = PatchCore()

    pbar = tqdm(enumerate(dataloader))
    for i, (image, defect_type) in pbar:
        model(image)

    print(model.embeddings[0].shape)
