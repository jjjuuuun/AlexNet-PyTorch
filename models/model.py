import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from base import BaseModel

class AlexNet(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        
        self.Conv1 = nn.Sequential(
            # Input Shape   >>  (Batch, 3, 227, 227)
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.Conv2 = nn.Sequential(
            # Input Shape   >>  (Batch, 96, 27, 27)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.Conv3 = nn.Sequential(
            # Input Shape   >>  (Batch, 256, 27, 27)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.Conv4 = nn.Sequential(
            # Input Shape   >>  (Batch, 384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.Conv5 = nn.Sequential(
            # Input Shape   >>  (Batch, 384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.Dense1 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU()
        )

        self.Dense2 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )

        self.Dense3 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        # self.init_weight()
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.Dense1(x)
        x = self.Dense2(x)
        x = self.Dense3(x)
        return x

    ########## Original paper code ##########
    # def init_weight(self):
    #     for name, module in self.named_modules():        
    #         if isinstance(module, nn.Conv2d):
    #             nn.init.normal_(module.weight, mean=0, std=0.01)
    #             if name in ['Conv2.0', 'Conv4.0', 'Conv5.0']:
    #                 nn.init.ones_(module.bias)
    #             else:
    #                 nn.init.zeros_(module.bias)
    #         elif isinstance(module, nn.Linear):
    #             nn.init.normal_(module.weight, mean=0, std=0.01)
    #             nn.init.ones_(module.bias)
    ########################################