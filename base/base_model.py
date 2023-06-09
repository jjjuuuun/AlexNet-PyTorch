import torch
from torch import nn
import numpy as np
from abc import abstractmethod



class BaseModel(nn.Module):

    @abstractmethod
    def forward(self, input):
        pass

    
