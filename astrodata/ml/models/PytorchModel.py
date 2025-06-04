from typing import Optional, Type

import torch
from BaseModel import BaseModel
from torch.utils.data import DataLoader, TensorDataset


class PyTorchModel(BaseModel):

    def __init__(self):
        super().__init__()
