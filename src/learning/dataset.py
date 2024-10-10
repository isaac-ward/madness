import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
import warnings
import utils.logging
import utils.general
from tqdm import tqdm
import numpy as np

class DummyDataset(Dataset):

    def __init__(self, episode_length, stage, log_folder):
        super().__init__()
        self.episode_length = episode_length
        self.stage = stage
        self.log_folder = log_folder

    def __len__(self):
        return self.episode_length

    def __getitem__(self, idx):
        data = {}
        return data
    
