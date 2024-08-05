import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

import nflows.nn.nets
from nflows.distributions.normal import StandardNormal
from nflows.flows import Flow
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

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

        # Create the learning network
        self.model = self._create_model()
        
    def update_state_goal(
        self,
        state_goal,
    ):
        self.state_goal = state_goal

    def _create_transform_parameters_block(self, num_features_identity, num_features_transform):
        """
        A valid transform parameters block will be the same shape as the to-be-transformed-input, as this get's
        called in the nflows code:
        https://github.com/bayesiains/nflows/blob/3b122e5bbc14ed196301969c12d1c2d94fdfba47/nflows/transforms/coupling.py 
        z, jac = self.transformer(inputs, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
        """
        
        class TransformParametersBlock(nn.Module):
            def __init__(self, num_features_identity, num_features_transform, num_features_context):
                super().__init__()
                # Save constructor variables
                self.num_features_identity = num_features_identity
                self.num_features_transform = num_features_transform
                self.num_features_context = num_features_context
                # Compute inputs and ouputs
                num_inputs = num_features_identity + num_features_context
                num_outputs = num_features_transform
                print(f"{self.__class__.__name__}:")
                print(f"\t-num_features_identity={num_features_identity}")
                print(f"\t-num_features_transform={num_features_transform}")
                print(f"\t-num_features_context={num_features_context}")
                print(f"\t-num_inputs={num_inputs}")
                print(f"\t-num_outputs={num_outputs}")
                # Simple big then small MLP
                self.transform = nn.Sequential(
                    nn.Linear(num_inputs, 2 * num_inputs),
                    nn.ReLU(),
                    nn.Linear(2 * num_inputs, num_outputs),
                )

            def forward(self, x, context):
                """
                x shape -> (K*C, identity_features), the identity features repeated for every sample-context pair
                context shape -> (K*C), the context vector repeated for every sample
                """

                # The context is repeated for every sample, so we need to reshape it
                # from (K*C,) to (K,C)
                context = context.view(-1, self.num_features_context)

                # Reshape x, to (K, C, I)
                x = x.view(-1, self.num_features_context, self.num_features_identity)
                x = x[:, 0, :]

                # Need to put in something of shape (K, I+C)
                network_input = torch.cat((x, context), dim=-1)
                print(f"network_input: {network_input.shape}")

                return self.transform(network_input)
            
        return TransformParametersBlock(
            num_features_identity=num_features_identity, 
            num_features_transform=num_features_transform,
            num_features_context=self.context_output_size
        )

    def _create_model(self):
        """
        Define a normalizing flow network that takes a sample from a 
        standard normal distribution and transforms it into a sample
        from the learned optimal action distribution

        When we sample from the learned optimal action distribution,
        we get a tensor with enough material in it to reshape into
        (H, |A|), where H is the horizon and |A| is the dimension of 
        the action space

        The flow should be parameterized by a fixed length context vector
        which is generated by a separate context network
        """

        # Convienience and readability
        A = self.mppi_computer.dynamics.action_size()
        H = self.H
        C = self.context_output_size
        #print(f"Flow network: A={A}, H={H}, C={C}")

        def create_net(in_features, out_features):
            # Make a little baby resnet
            net = nflows.nn.nets.ResidualNet(
                in_features, 
                out_features, 
                hidden_features=128, 
                num_blocks=3,
                use_batch_norm=False,
            )
            return net

        transforms = []
        for _ in range(self.num_flow_layers):
            # https://github.com/bayesiains/nflows/blob/3b122e5bbc14ed196301969c12d1c2d94fdfba47/nflows/transforms/coupling.py#L212
            affine_coupling_transform = AffineCouplingTransform(
                # A simple alternating binary mask stating which features will be transformed
                # via an affine transformation (mask > 0) and which will inform the creation of the
                # the transformation parameters (mask <= 0, identity)
                mask=torch.arange(H*A) % 2,  
                # A function that creates a transformation object that which is constructed with two
                # constructor variables: the number of identity features, and the number of transform features
                # Moreover, it must have a forward function which accepts the parts of the input that are to 
                # be transformed, and the context vector, and produces some transformation parameters that will
                # be a
                transform_net_create_fn=self._create_transform_parameters_block,
                #transform_net_create_fn=create_net,
            )
            transforms.append(affine_coupling_transform)

            # Reverses the elements of a 1d input
            # https://github.com/bayesiains/nflows/blob/3b122e5bbc14ed196301969c12d1c2d94fdfba47/nflows/transforms/permutations.py#L56
            reverse_permutation = ReversePermutation(features=H*A)
            transforms.append(reverse_permutation)

        # Need a context network
        context_network = nn.Sequential(
            nn.Linear(self.context_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.context_output_size)
        )

        # Convert into a single composite transform
        transform = CompositeTransform(transforms)
        distribution = StandardNormal(shape=[H*A])
        flow = Flow(transform, distribution, embedding_net=context_network)
        return flow
    
    def forward(self, state_history, action_history, state_goal):
        # Assemble the context vector, which includes the
        # current state
        # goal state
        # Ensure everything is a torch tensor
        state_current = torch.tensor(state_history[-1], dtype=torch.float32)
        state_goal    = torch.tensor(state_goal, dtype=torch.float32)
        context_input = torch.cat((state_current, state_goal), dim=0)

        # Sample from the normalizing flow and get log probs (this is more
        # efficient than sampling and then computing the log probs, though log
        # likelihoods are not needed during validation)
        # We'll get actions as (K,K,H*A)
        # We'll get log_likelihoods as (K,K)
        # Read this as 'generate K samples for each context vector', where we only have one
        # context vector
        samples, log_likelihoods = self.model.sample_and_log_prob(self.K, context=context_input)

        print(f"actions: {actions.shape}")
        print(f"log_likelihoods: {log_likelihoods.shape}")

        # Reshape the samples into K, H-long sequences of action vectors of shape K,H,A
        actions = samples.view(self.K, self.H, self.mppi_computer.dynamics.action_size())

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

        # What was the average and best reward?
        reward_best = torch.max(rewards)
        reward_mean = torch.mean(rewards)

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
        self.log(f'{stage}/loss',        loss,        on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}/reward/best', reward_best, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}/reward/mean', reward_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # And return the loss
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, 'val')
        
    def configure_optimizers(self):
        learnable_parameters = self.model.parameters()
        return optim.Adam(learnable_parameters, lr=self.learning_rate)