import torch
from tqdm import tqdm
from config import DataConfig
from data.datasets import get_train_loader, get_val_loader
from model import PatchCore

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_config = DataConfig()
    dataloader = get_train_loader(dataset_config)
    model = PatchCore().to(device)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (image, defect_type) in pbar:
        image = image.to(device)
        model(image)

    model.make_coreset()
    model.save_coreset("temp.pt")
    model.load_coreset("temp.pt")
    model.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (image, defect_type) in pbar:
        image = image.to(device)
        print(model(image))

    dataloader = get_val_loader(dataset_config)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (image, defect_type) in pbar:
        image = image.to(device)
        print(model(image))