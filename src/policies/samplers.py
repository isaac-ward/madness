import numpy as np
from tqdm import tqdm
import os
import cupy as cp
import shutil

from scipy.stats.qmc import Sobol

#from learning.models import PolicyFlowActionDistribution

class Sampler:
    def __init__(
        self,
        K,
        H,
        action_ranges,
    ):
        self.K = K
        self.H = H
        self.action_ranges = np.array(action_ranges)
        self.action_size = self.action_ranges.shape[0]

    def sample(self):
        """
        Sample K action plans, each with H actions
        """
        raise NotImplementedError
    
# ----------------------------------------------------------------
    
class FixedSampler(Sampler):
    """
    Not a sampler, just a convienence class to pass in a fixed set of actions
    """
    def __init__(self, actions):
        self.actions = actions
    
    def sample(self):
        return self.actions
    
# ----------------------------------------------------------------

class RandomActionSampler(Sampler):
    def __init__(self, K, H, action_ranges):
        super().__init__(K, H, action_ranges)

    def sample(self):
        return np.random.uniform(self.action_ranges[:,0], self.action_ranges[:,1], (self.K, self.H, self.action_size))
    
# ----------------------------------------------------------------

class SobolActionSampler(Sampler):
    def __init__(self, K, H, action_ranges):
        super().__init__(K, H, action_ranges)
                
    def sample(self):
        dim = self.action_size * self.H
        sampler = Sobol(dim)
        higher_power_of_two = np.ceil(np.log2(self.K)).astype(int)
        samples = sampler.random_base2(higher_power_of_two) # 2^m samples
        samples = samples[:self.K] # Take only K samples
        samples = samples.reshape((self.K, self.H, self.action_size))
        # Make sure they're in range
        samples = samples * (self.action_ranges[:,1] - self.action_ranges[:,0]) + self.action_ranges[:,0]
        return samples
    
# ----------------------------------------------------------------
    
class RolloverGaussianActionSampler(Sampler):
    def __init__(self, K, H, action_ranges):
        super().__init__(K, H, action_ranges)
        # When initialized, the previous optimal action plan is the mean action
        # value, or just all zeros
        self.previous_optimal_action_plan = np.zeros((H, action_ranges.shape[0]))
        # for i in range(action_ranges.shape[0]):
        #     self.previous_optimal_action_plan[:,i] = (action_ranges[i,0] + action_ranges[i,1]) / 2
    
    def sample(self):
        """
        Create a gaussian centered around the previous best action plan
        and sample from it with some standard deviation (based off action_ranges)
        """

        # I essentially need K lots of (H, action_size) sized samples, where
        # the standard devation of each element in the sample is according to 
        # the std_dev vector

        # Create a gaussian centered around the previous best action plan
        # (H, action_size)
        mean = self.previous_optimal_action_plan 
        # Perform the shift operation - we center around the best actions
        # shifted forward by one, with zero padding at the end
        shifted_mean = np.zeros_like(mean)
        shifted_mean[:-1] = mean[1:]

        # Variance too low and we'll narrow in, variance too high and we'll 
        # hit the ends of our action ranges constantly. Too high in particular
        # will result in a MAX/MIN type action plan which will almost always
        # lead to failure. Too low makes it hard to quickly change behavior
        # and adequately explore the state space
        # (action_size)
        std_dev = np.abs(self.action_ranges[:,1] - self.action_ranges[:,0]) / 4 

        # This will draw K * H * action_size samples around the shifted mean
        samples = np.random.normal(shifted_mean, std_dev, (self.K, self.H, self.action_size))
        
        # Clip into the action ranges
        samples = np.clip(samples, self.action_ranges[:,0], self.action_ranges[:,1])
        return samples
    
    def update_previous_optimal_action_plan(self, optimal_action_plan):
        self.previous_optimal_action_plan = optimal_action_plan

# ----------------------------------------------------------------

# TODO: Implement a sampler that samples actions from a learned policy
# TODO: reconsider how MPPI computer is being used in policy - may need to have a sampler model
# that is imported here and by the policy for training
# class FlowActionSampler(Sampler):
#     def __init__(self, K, H, action_ranges, policy):
#         super().__init__(K, H, action_ranges)
#         self.policy = policy

#     def sample(self):
#         return self.policy.sample_actions(self.K, self.H)