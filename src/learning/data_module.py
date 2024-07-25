import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from learning.data_loader import SDPDataLoader

class EnvironmentDataModule(pl.LightningDataModule):
    """
    Wraps a sequential decision-making environment into a
    data module that we can fit to using Lightning AI
    """

    def __init__(self, environment):
        super().__init__()
        self.environment = environment
    
    def setup(self, stage=None):
        self.environment.reset()
    
    def next_state(self, action):
        return self.environment.step(action)