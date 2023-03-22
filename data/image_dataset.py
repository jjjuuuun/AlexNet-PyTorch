from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms 
import torch
from base import BaseDataset
from utils import read_zip, read_img

class TrainDataset(BaseDataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.imgs = []
        self.labels= []
        print('Loading Image ...')
        if self.data_path.suffix == '.zip':
            self.imgs, self.labels = read_zip(self.data_path)
        else:
            self.imgs, self.labels = read_img(self.data_path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        self.transform = transforms.Compose([
            # transforms.RandomAffine(degrees=0, translate=[0.2, 0.2]),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop((227, 227)),
            transforms.CenterCrop(227),
            # Instead of Color PCA Augmentation
            # transforms.ColorJitter(brightness=0.5,
            #                        contrast=0.5,
            #                        saturation=0.5,
            #                        hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor_img = self.transform(self.imgs[idx])

        tensor_label = torch.LongTensor(self.labels[idx]).squeeze()

        return tensor_img, tensor_label
    
    

