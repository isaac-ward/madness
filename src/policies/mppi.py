import numpy as np
from tqdm import tqdm
import os
import cupy as cp
#from numba import njit, prange
import shutil

from scipy.stats.qmc import Sobol

import utils.geometric
import utils.general
import policies.rewards

class MPPIComputer:
    """
    An MPPI computer handles the sampling of actions, rollouts of actions
    wrt to some dynamics model, optimal action production (wrt some reward),
    and logging
    """
    pass

class PolicyMPPI:
    def __init__(
        self,
        state_size,
        action_size,
        dynamics,
        K,
        H,
        action_ranges,
        lambda_,
        map_,
        use_gpu_if_available=True,
    ):
        """
        Roll out a bunch of random actions and select the best one
        """
        self.state_size = state_size
        self.action_size = action_size
        self.dynamics = dynamics
        self.K = K
        self.H = H
        self.action_ranges = action_ranges

        # Lambda is the temperature of the softmax
        # infinity selects the best action plan, 0 selects uniformly
        self.lambda_ = lambda_

        # We need a map to plan paths against (e.g. collision checking)
        self.map_ = map_

        # Typically we'll sample about the previous best action plan
        self._previous_optimal_action_plan = np.zeros((H, action_size))

        # Will need a path to follow, but we want it to be
        # updated separately (changeable)
        self.state_goal = None

        # Are we going to have logging? Defaultly no
        self.log_folder = None

        # If we're using a GPU, we'll need to move some things over
        self.use_gpu_if_available = use_gpu_if_available

    def update_state_goal(
        self,
        state_goal,
    ):
        """
        Update the path to follow
        """
        self.state_goal = state_goal

    def enable_logging(
        self,
        run_folder,
    ):
        """
        Enable logging to a folder
        """
        self.log_folder = os.path.join(run_folder, "policy", "mppi")

    def delete_logs(self):
        """
        Delete all logs
        """
        if self.log_folder is not None:
            shutil.rmtree(self.log_folder)

    # ----------------------------------------------------------------

    def _random_uniform_sample_actions(self):
        return np.random.uniform(self.action_ranges[:,0], self.action_ranges[:,1], (self.K, self.H, self.action_size))
    
    def _sobol_sample_actions(self):
        dim = self.action_size * self.H
        sampler = Sobol(dim)
        higher_power_of_two = np.ceil(np.log2(self.K)).astype(int)
        samples = sampler.random_base2(higher_power_of_two) # 2^m samples
        samples = samples[:self.K] # Take only K samples
        samples = samples.reshape((self.K, self.H, self.action_size))
        # Make sure they're in range
        samples = samples * (self.action_ranges[:,1] - self.action_ranges[:,0]) + self.action_ranges[:,0]
        return samples
    
    def _rollover_mean_gaussian(self):
        """
        Create a gaussian centered around the previous best action plan
        and sample from it with some standard deviation (based off action_ranges)
        """

        # I essentially need K lots of (H, action_size) sized samples, where
        # the standard devation of each element in the sample is according to 
        # the std_dev vector

        # Create a gaussian centered around the previous best action plan
        # (H, action_size)
        mean = self._previous_optimal_action_plan 
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

    # ----------------------------------------------------------------

    def act(
        self,
        state_history,
        action_history,
    ):

        # Check if we have a path to follow
        if self.state_goal is None:
            raise ValueError(f"{self.__class__.__name__} requires a goal state to follow")

        # For convenience
        K = self.K
        H = self.H

        # Sample actions from some distribution
        #action_plans = self._random_uniform_sample_actions()
        #action_plans = self._sobol_sample_actions()
        action_plans = self._rollover_mean_gaussian()

        # We'll simulate those actions using dynamics and figure
        # out the states
        state_plans = np.zeros((K, H, self.state_size))

        # We will compute rewards for each future
        rewards = np.zeros(K)

        # If GPU is available and we desire it, then we'll use it
        using_gpu = self.use_gpu_if_available and cp.cuda.is_available()
        # Move everything to the GPU if we're using it
        if using_gpu:
            state_history   = cp.array(state_history)
            action_history  = cp.array(action_history)
            state_plans     = cp.array(state_plans)
            action_plans    = cp.array(action_plans)
            rewards         = cp.array(rewards)
            self.state_goal = cp.array(self.state_goal)

        # What module to use? cupy or numpy?
        xp = cp.get_array_module(state_history)

        # Roll out futures in parallel (needs to be serial because we need to compute
        # the state at t=0 before we can compute the state at t=1)
        for h in tqdm(range(H), desc="Rolling out futures", leave=False, disable=True):
            # Compute the next states
            state_plans[:, h] = self.dynamics.step(
                # If it's our first computation, start at our last known state, otherwise
                # start at the last computed state
                xp.tile(state_history[-1], (K, 1)) if h == 0 else state_plans[:, h - 1],
                action_plans[:, h],
            )
            
        # Compute all rewards
        rewards = policies.rewards.batch_reward(
            state_plans, 
            action_plans,
            self.state_goal,
            self.map_,
        )

        # TODO
        # # Normalize rewards between 0-1 so that they don't blow up when exponentiated
        # min_reward = np.min(rewards)
        # max_reward = np.max(rewards)
        # normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
        # # Use softmax style weighting to compute the best action plan
        # # Rewards is shape (K,)
        # # Weights should be shape (K,)
        # print(normalized_rewards)
        # weights = np.exp(+ self.lambda_ * normalized_rewards)
        # # Must check for divide by zero TODO
        # weights = weights / np.sum(weights)
        # print(np.sum(weights))
        # # Compute optimal action plan 
        # optimal_action_plan = np.sum(weights[:, np.newaxis, np.newaxis] * action_plans, axis=0) 
        # optimal_action = optimal_action_plan[0]    

        # Select the best plan and return the immediate action from that plan
        optimal_action_plan = action_plans[xp.argmax(rewards)]
        optimal_state_plan  = state_plans[xp.argmax(rewards)]
        optimal_action = optimal_action_plan[0]

        # Convert everything back to CPU if necessary
        if using_gpu:
            state_history       = cp.asnumpy(state_history)
            action_history      = cp.asnumpy(action_history)
            state_plans         = cp.asnumpy(state_plans)
            action_plans        = cp.asnumpy(action_plans)
            rewards             = cp.asnumpy(rewards)
            self.state_goal     = cp.asnumpy(self.state_goal)
            optimal_state_plan  = cp.asnumpy(optimal_state_plan)
            optimal_action_plan = cp.asnumpy(optimal_action_plan)
            optimal_action      = cp.asnumpy(optimal_action)

        # Update the model that we're sampling from
        self._previous_optimal_action_plan = optimal_action_plan

        # ----------------------------------------------------------------
        # Logging from here on
        # ----------------------------------------------------------------

        # Log the state and action plans alongside the reward, 
        # if we're logging
        if self.log_folder is not None:
            # Create a subfolder for this step
            folder = os.path.join(self.log_folder, f"step_{utils.general.get_timestamp(ultra_precise=True)}")
            os.makedirs(folder, exist_ok=True)
            # Save the state and action plans
            utils.logging.save_state_and_action_trajectories(
                folder,
                state_plans,
                action_plans,
            )
            # Save the rewards
            utils.logging.pickle_to_filepath(
                os.path.join(folder, "rewards.pkl"),
                rewards,
            )
            # If we're logging we will want to see what the optimal plan was
            optimal_state_plan = np.zeros((H, self.state_size))
            for h in range(H):
                optimal_state_plan[h] = self.dynamics.step(
                    state_history[-1] if h == 0 else optimal_state_plan[h - 1],
                    optimal_action_plan[h],
                )

            # Save the optimal plans
            utils.logging.save_state_and_action_trajectories(
                folder,
                optimal_state_plan,
                optimal_action_plan,
                suffix="optimal",
            )

        return optimal_action
                


