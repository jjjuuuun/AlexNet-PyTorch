import torch
from torch import nn
from torch.utils.data import Dataset
from abc import abstractmethod
from PIL import Image

class BaseDataset(Dataset):
    """Data loader를 만들기 위한 base class"""

    def __init__(self, data_path):
        # self.config = config
        self.data_path = data_path

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self):
        pass


def resize(img):
    if img.size[0] < img.size[1]:
        return img.resize((256, img.size[1]), Image.BILINEAR)
    else:
        return img.resize((img.size[0], 256), Image.BILINEAR)


def center_crop(img):
    width, height = img.size

    left = (width - 256)/2
    top = (height - 256)/2
    right = (width + 256)/2
    bottom = (height + 256)/2

    return img.crop((left, top, right, bottom))
