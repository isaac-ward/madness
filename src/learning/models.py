import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from nflows.distributions.normal import StandardNormal
from nflows.flows import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform, ReversePermutation

from policies.rewards import batch_reward
from policies.mppi import MPPIComputer
import policies.samplers

class PolicyFlowActionDistribution(pl.LightningModule):
    """
    Learns the optimal action distribution for MPPI using normalizing flows

    When used, an object of this class will be used in tandem with a MPPI policy 
    object and this will be used to generate optimal actions
    """
    def __init__(
        self, 
        dynamics,
        K,
        H,
        context_input_size,
        context_output_size,
        num_flow_layers,
        learning_rate,
    ):
        # Initialize the parent class (pl.LightningModule)
        super(PolicyFlowActionDistribution, self).__init__()

        # Don't defaultly set a goal, this needs to be set 
        self.state_goal = None

        # Save the learning rate
        self.learning_rate = learning_rate

        # Will need access to an MPPI computer
        self.mppi_computer = MPPIComputer(
            dynamics=dynamics,
            K=K,
            H=H,
            lambda_=None,
            map_=None,
        )

        # Define the internal shapes
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
        
    def update_state_goal(
        self,
        state_goal,
    ):
        self.state_goal = state_goal

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
    
    def forward(self, state_history, action_history, state_goal):
        # Assemble the context vector, which includes the
        # current state
        # goal state
        state_current = state_history[-1]
        context_input = torch.cat((state_current, state_goal), dim=0)

        # Generate a context vector from the desired contextual information
        # and parameterize the normalizing flow with it
        context = self.context_network(context_input)

        # Draw K samples from the normalizing flow
        actions = self.flow_network.sample(num_samples=self.K, context=context)
        actions = actions.view(self.K, self.H, self.dynamic.action_size)

        # Compute the likelihood of the control sequences wrt the context vector
        # using the reverse mode of the normalizing flow
        # TODO unecessary during validation
        log_likelihoods = self.flow_network.log_prob(actions, context)

        # Compute the optimal future using MPPI
        state_plans, action_plans, rewards, optimal_state_plan, optimal_action_plan = self.mppi_computer.compute(
            state_history,
            action_history,
            state_goal,
            # The MPPI computer needs to take a 'sampler'
            policies.samplers.FixedSampler(actions),
        )        
        
        return state_plans, action_plans, rewards, optimal_state_plan, optimal_action_plan, log_likelihoods
    
    def act(
        self,
        state_history,
        action_history,
    ):
        # Check if we have a goal
        if self.state_goal is None:
            raise ValueError("A goal state must be set before acting")
        
        # Return the optimal action
        _, _, _, _, optimal_action_plan = self.forward(state_history, action_history, self.state_goal)
        return optimal_action_plan[0]
    
    def generic_step(self, batch, batch_idx, stage):
        pass
    
    def training_step(self, batch, batch_idx):
        """
        Generate K samples from the learned optimal action distribution,
        and compute a loss that encourages the samples to be close to
        the optimal action distribution
        """

        # Start by getting the contextual variables
        state_history = batch["state_history"]
        action_history = batch["action_history"]
        state_goal = batch["state_goal"]

        # Run a forward pass, getting all the required information 
        # from the MPPI computer
        state_plans, action_plans, rewards, optimal_state_plan, optimal_action_plan, log_likelihoods = self.forward(state_history, action_history, state_goal)

        # Compute the probabilities of optimality from the reward function
        # This is just the exponential of the rewards
        p_opt = torch.exp(rewards)

        # Compute the likelihood of the control sequences wrt the context vector
        # using the reverse mode of the normalizing flow
        p_likelihood = torch.exp(log_likelihoods)

        # TODO hyperparameter weighting of opt/entropy

        # Compute the loss weight for each sample
        # Combine probabilities elementwise and then get the mean
        p_combined = torch.mean(p_opt * p_likelihood)
        # What fraction of each samples combined probability does
        # each sample take up?
        weights = (p_opt * p_likelihood) / p_combined

        # Assemble the total loss
        loss = -torch.mean(weights * log_likelihoods)

        # Do custom logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # And return the loss
        return loss

    def validation_step(self, batch, batch_idx):
        pass
        
    def configure_optimizers(self):
        learnable_parameters = self.flow_network.parameters() + self.context_network.parameters()
        return optim.Adam(learnable_parameters, lr=self.lr)