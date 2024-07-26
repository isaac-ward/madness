import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

class EnvironmentDataset(Dataset):
    """
    Wraps a sequential decision-making environment into a
    torch dataset. It works by returning items with respect
    to some policy. If we exceed an episode length, or are
    otherwise done, then we reset the environment and start 
    again

    To avoid spikes in the learning curves, and variability 
    in performance across episodes, we can: 
        1) use parallel environments,
        2) use experience replay 
        3) learn on batches of experiences before updating
           the policy
    """

    def __init__(self, environment, policy):
        super().__init__()
        self.environment = environment
        self.policy = policy

        # Ensure the environment is reset

    def __len__(self):
        return self.environment.episode_length

    def __getitem__(self, idx):
        """
        An on-policy approach to getting the next learning instance
        """

        # Without gradients, we determine the next action fm the policy

        # We step the environment with the action

        # We reset the environment as needed

        # We return the information required for learning
        
        pass
    
