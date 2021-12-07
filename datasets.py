"""
Author: Said Kaloev
Exercise 5: Machine Learning Project
"""

import torch
import pickle as pkl
import gzip
import numpy as np
from torch.utils.data import Dataset

class ImageData(Dataset):
    """Get data from prepared pickels file"""
    def __init__(self):
        with gzip.open('prep_files.pklz', 'rb') as something:
            self.dataset = pkl.load(something)
            self.dataset = self.dataset['array']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file_data = self.dataset[idx]
        return file_data
