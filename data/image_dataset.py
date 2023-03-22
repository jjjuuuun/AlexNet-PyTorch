from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms 
import torch
from base import BaseDataset
from utils import read_zip, read_img

# try:
#     from base import BaseDataset
#     from utils import read_zip, read_img
# except:
#     import os
#     import sys
#     sys.path.append(os.getcwd())
#     from base import BaseDataset
#     from utils import read_zip, read_img
#     from torch.utils.data import DataLoader

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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    base_path = Path(Path.cwd())
    img_path = base_path / 'datasets' / 'dogs_and_cats' / 'train_subsample'
    
    train_set = TrainDataset(img_path)

    train_iter = DataLoader(train_set,
                        batch_size=4,
                        shuffle = True)

    cnt = 1
    for img, label in train_iter:
        for im in img:
            if im.shape[-1] < 224 or im.shape[-2]  < 224:
                print(cnt)
                cnt += 1

    # print(next(iter(train_iter)))
    # img, label = train_set[1]
    # print(label)
    
    # img = transforms.Normalize(
    #     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    #     std=[1/0.229, 1/0.224, 1/0.255])((img))
    # img = transforms.ToPILImage()(img)
    # plt.imshow(np.array(img))
    # plt.show()

    
    

