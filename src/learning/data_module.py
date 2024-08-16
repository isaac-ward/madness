import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import copy
import os

from learning.dataset import EnvironmentDataset

# TODO create a dataset that is on-policy and informed by the
# in-training policy's predicted actions, and then return those

class EnvironmentDataModule(pl.LightningDataModule):
    """
    Wraps a sequential decision-making environment into a
    data module that we can fit to using Lightning AI
    """

    def __init__(self, environment, agent, log_folder, batch_size=1):
        super().__init__()
        self.environment = environment
        self.agent = agent
        self.log_folder = log_folder
        # Note that this is a little different to the notion of a
        # traditional supervised learning batch_size. Since we're
        # on-policy, we're going to be using the agent to determine
        # the next action, and so the batch size is the number of
        # steps we take before updating the policy
        self.batch_size = batch_size

        # For speeding up dataloaders
        #self.num_workers = os.cpu_count()//2 - 1
        # NOTE this has to be one for the RL paradigm to work
        self.num_workers = 1
    
    def setup(self, stage=None):
        # Create the training and validation datasets
        self.dataset_train = EnvironmentDataset(
            environment=copy.deepcopy(self.environment),
            agent=copy.deepcopy(self.agent),
            stage="train",
            log_folder=None, # Don't log during training
        )
        self.dataset_val = EnvironmentDataset(
            environment=copy.deepcopy(self.environment),
            agent=copy.deepcopy(self.agent),
            stage="val",
            log_folder=self.log_folder,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
