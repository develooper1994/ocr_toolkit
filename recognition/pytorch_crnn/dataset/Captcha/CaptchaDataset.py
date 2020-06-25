import os

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CaptchaDataset(Dataset):
    def __init__(self, img_dir: str):
        pathes = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir)
        self.img_dir = img_dir
        self.pathes = [os.path.join(abspath, path) for path in pathes]
        self.list_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.pathes)

    def __getitem__(self, idx):
        path = self.pathes[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        text = self.get_filename(path)
        return img, text

    def get_filename(self, path: str) -> str:
        return os.path.basename(path).split('.')[0].lower().strip()

    def transform(self, img) -> torch.Tensor:
        return self.list_transforms(img)