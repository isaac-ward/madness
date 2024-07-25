import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from learning.data_loader import SDPDataLoader

# TODO create a dataset that is on-policy and informed by the
# in-training policy's predicted actions, and then return those

class EnvironmentDataModule(pl.LightningDataModule):
    """
    Wraps a sequential decision-making environment into a
    data module that we can fit to using Lightning AI
    """

    def __init__(self, environment):
        super().__init__()
        self.environment = environment
    
    def setup(self, stage=None):
        # TODO compute a random start and goal state
        # and use it to reset the environment
        self.environment.reset()
    
    def step(self, action):
        return self.environment.step(action)
    
