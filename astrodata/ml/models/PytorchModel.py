import torch
from BaseModel import BaseModel
from torch.utils.data import TensorDataset, DataLoader
from typing import Type, Optional

class PyTorchModel(BaseModel):

    def __init__(self):
        super().__init__()
    