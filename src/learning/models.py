import torch.optim as optim
import numpy as np
import pytorch_lightning as pl

from nflows.distributions.normal import StandardNormal
from nflows.flows import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform, ReversePermutation

from policies.rewards import batch_reward

class FlowActionDistribution(pl.LightningModule):
    """
    Learns the optimal action distribution for MPPI using normalizing flows

    When used, an object of this class will be used in tandem with a MPPI policy 
    object and this will be used to generate optimal actions
    """

    def __init__(
        self, 
        state_size,
        action_size,
        K,
        H,
        context_input_size,
        context_output_size,
        num_flow_layers,
        learning_rate,
    ):
        # Initialize the parent class (pl.LightningModule)
        super(FlowActionDistribution, self).__init__()

        # Save the learning rate
        self.learning_rate = learning_rate

        # Define the internal shapes
        self.state_size = state_size
        self.action_size = action_size
        self.K = K
        self.H = H
        self.context_input_size = context_input_size
        self.context_output_size = context_output_size
        self.num_flow_layers = num_flow_layers

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
        we get a tensor with enough material in it to reshape into
        (H, |A|), where H is the horizon and |A| is the dimension of 
        the action space

        The flow should be parameterized by a fixed length context vector
        """
        transforms = []
        for _ in range(self.num_flow_layers):
            # Reverses the elements of a 1d input
            # https://github.com/bayesiains/nflows/blob/3b122e5bbc14ed196301969c12d1c2d94fdfba47/nflows/transforms/permutations.py#L56
            transforms.append(ReversePermutation(features=self.H * self.action_size))
            # https://github.com/bayesiains/nflows/blob/3b122e5bbc14ed196301969c12d1c2d94fdfba47/nflows/transforms/autoregressive.py#L64
            transforms.append(MaskedAffineAutoregressiveTransform(
                features=self.H * self.action_size, 
                hidden_features=2 * self.H * self.action_size,
                context_features=self.context_output_size
            ))
        transform = CompositeTransform(transforms)
        distribution = StandardNormal(shape=[self.H * self.action_size])
        flow = Flow(transform, distribution)
        return flow

    def _context_network(self):
        """
        Takes some information of interest and embeds it into a fixed
        length context vector that can be used to parameterize the 
        normalizing flow

        A MLP is normally a good choice here if nothing is known about
        the input contextual data
        """
        return nn.Sequential(
            nn.Linear(self.context_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.context_output_size)
        )
    
    def forward(self, state_current, state_goal):
        # Assemble the context vector, which includes the
        # current state
        # goal state
        context_input = torch.cat((state_current, state_goal), dim=0)

        # Generate a context vector from the desired contextual information
        # and parameterize the normalizing flow with it
        context = self.context_network(context_input)

        # Draw K samples from the normalizing flow
        actions = self.flow_network.sample(num_samples=self.K, context=context)
        actions = actions.view(self.K, self.H, self.action_size)
        
        return actions
    
    def training_step(self, batch, batch_idx):
        """
        Generate K samples from the learned optimal action distribution,
        and compute a loss that encourages the samples to be close to
        the optimal action distribution
        """

        # Start by getting the contextual variables
        state_current = batch["state_current"]
        state_goal = batch["state_goal"]

        # Run a forward pass, getting the action samples
        actions = self.forward(state_current, state_goal)
        
        # Reward the samples based on the reward function
        rewards = batch_reward(
            state_trajectory_plans,
            action_trajectory_plans,
            state_goal,
            map_,
        )

        # Assemble the optimal action sequence from the 
        # reward information

        # Compute the probabilities of optimality from the reward function

        # Compute the likelihood of the control sequences wrt the context vector
        # using the reverse mode of the normalizing flow

        # Compute the loss weight for each sample

        # Assemble the total loss

        # Reset the environment if we're done

        # And return the loss
        pass
        
    def configure_optimizers(self):
        learnable_parameters = self.flow_network.parameters() + self.context_network.parameters()
        return optim.Adam(learnable_parameters, lr=self.lr)