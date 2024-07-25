import torch.optim as optim
import numpy as np
import pytorch_lightning as pl

from nflows import transforms, distributions, flows

class FlowMPPIModule(pl.LightningModule):
    """
    Learns the optimal action distribution for MPPI using normalizing flows
    """

    def __init__(self, policy_network, lr=1e-3):
        # Initialize the parent class (pl.LightningModule)
        super(FlowMPPIModule, self).__init__()

        # Save the learning rate
        self.lr = lr

        # Create the learning networks, which include
        # a normalizing flow
        self.flow_network = self._flow_network()
        # the context embedder
        self.context_network = self._context_network()

    def _flow_network(self):
        """
        Define a normalizing flow network that takes a sample from a 
        standard normal distribution and transforms it into a sample
        from the learned optimal action distribution

        When we sample from the learned optimal action distribution,
        we get a tensor of shape (H, |A|), where H is the horizon
        and |A| is the dimension of the action space
        """
        pass

    def _context_network(self):
        """
        Takes some information of interest and embeds it into a fixed
        length context vector that can be used to parameterize the 
        normalizing flow
        """
        pass
    
    def forward(self, state_current, state_goal):
        # Get the context vector, then draw K samples from the normalizing flow
        pass
    
    def training_step(self, batch, batch_idx):
        """
        Generate K samples from the learned optimal action distribution,
        and compute a loss that encourages the samples to be close to
        the optimal action distribution
        """

        # Start by getting the current state from the data module

        # And get the goal state from the data module

        # Generate a context vector from the desired contextual information
        # and parameterize the normalizing flow with it

        # Draw K samples from the normalizing flow

        # Compute the probabilities of optimality from the reward function

        # Compute the likelihood of the control sequences wrt the context vector
        # using the reverse mode of the normalizing flow

        # Compute the loss weight for each sample

        # Assemble the total loss

        # Reset the environment if we're done

        # And return the loss
        
    def configure_optimizers(self):
        return optim.Adam(self.network.parameters(), lr=self.lr)