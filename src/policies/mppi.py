import numpy as np
from tqdm import tqdm
import os
#from numba import njit, prange

from scipy.stats.qmc import Sobol

import utils.geometric
import utils.general

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
        self.path_xyz = None

        # Are we going to have logging? Defaultly no
        self.log_folder = None

    def reward(
        self,
        state_trajectory_plan,
        action_trajectory_plan,
    ):
        """
        Given a potential plan, return a scalar reward (higher is better)
        """

        p = state_trajectory_plan[:, 0:3]
        r = state_trajectory_plan[:, 3:6]
        v = state_trajectory_plan[:, 6:9]
        w = state_trajectory_plan[:, 9:12]
        # Goal point
        g = self.path_xyz[-1]
        
        # The cost function (negative reward) from the published work is:
        # a * distance_from_goal_at_T + SUM b * distance_from_goal_at_t + SUM c * collision_at_t
        goal_term = 100 * np.linalg.norm(p[-1] - g)
        path_term = 10 * np.sum(np.linalg.norm(p - g, axis=1))
        collision_term = 10000 * np.sum([self.map_.is_collision(p_i, collision_radius=0.5) for p_i in p])

        # Minimize the first derivatives (low speed, angular velocity is favored)
        derivative_term = 1 * np.sum(np.linalg.norm(v, axis=1)) + 1 * np.sum(np.linalg.norm(w, axis=1))

        # Assemble, and note we're using a reward paradigm
        cost = goal_term + path_term + collision_term #+ derivative_term
        reward = -cost
        return reward     

    def update_path_xyz(
        self,
        path_xyz,
    ):
        """
        Update the path to follow
        """
        self.path_xyz = path_xyz

    def enable_logging(
        self,
        run_folder,
    ):
        """
        Enable logging to a folder
        """
        self.log_folder = os.path.join(run_folder, "policy", "mppi")

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
        # (action_size)\
        # Variance too low and we'll narrow in, variance too high and we'll 
        # hit the ends of our action ranges constantly. Too high in particular
        # will result in a MAX/MIN type action plan which will almost always
        # lead to failure. Too low makes it hard to quickly change behavior
        std_dev = np.abs(self.action_ranges[:,1] - self.action_ranges[:,0]) / 4 # 4
        # This will draw K * H * action_size samples
        samples = np.random.normal(mean, std_dev, (self.K, self.H, self.action_size))
        #print(samples)
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
        if self.path_xyz is None:
            raise ValueError(f"{self.__class__.__name__} requires a path to follow")

        # For convenience
        K = self.K
        H = self.H

        # Sample actions from some distribution
        #action_plans = self._random_uniform_sample_actions()
        #action_plans = self._sobol_sample_actions()
        action_plans = self._rollover_mean_gaussian()
        #print(action_plans)

        # We'll simulate those actions using dynamics and figure
        # out the states
        state_plans = np.zeros((K, H, self.state_size))

        # We will compute rewards for each future
        rewards = np.zeros(K)

        # Roll out futures in parallel
        for h in tqdm(range(H), desc="Rolling out futures", leave=False, disable=True):
            # Compute the next states
            state_plans[:, h] = self.dynamics.step(
                np.tile(state_history[-1], (K, 1)) if h == 0 else state_plans[:, h - 1],
                action_plans[:, h],
            )

        # # Roll out futures NOT in parallel
        # for k in tqdm(range(K), desc="Rolling out futures", leave=False, disable=True):
        #     for h in range(H):
        #         state_plans[k, h + 1] = self.dynamics.step(
        #             state_plans[k, h],
        #             action_plans[k, h],
        #         )
            
        # Compute all rewards
        # TODO parallelize
        for k in tqdm(range(K), desc="Computing rewards", leave=False, disable=True):
            rewards[k] = self.reward(
                state_plans[k],
                action_plans[k],
            )

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
        optimal_action_plan = action_plans[np.argmax(rewards)]
        optimal_state_plan  = state_plans[np.argmax(rewards)]
        self._previous_optimal_action_plan = optimal_action_plan
        optimal_action = optimal_action_plan[0]

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
                


