import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import copy
import os

from learning.dataset import DummyDataset

class DummyDataModule(pl.LightningDataModule):

    def __init__(self, episode_length, log_folder, batch_size=1):
        super().__init__()
        self.episode_length = episode_length
        self.log_folder = log_folder
        self.batch_size = batch_size

        # For speeding up dataloaders
        #self.num_workers = os.cpu_count()//2 - 1
        # NOTE this has to be one for the RL paradigm to work
        self.num_workers = 1
    
    def setup(self, stage=None):
        # Create the training and validation datasets
        self.dataset_train = DummyDataset(
            episode_length=self.episode_length,
            stage="train",
            log_folder=None, # Don't log during training
        )
        self.dataset_val = DummyDataset(
            episode_length=self.episode_length,
            stage="val",
            log_folder=self.log_folder,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
