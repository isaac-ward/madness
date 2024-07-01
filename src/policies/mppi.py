import numpy as np
from tqdm import tqdm
import os
from numba import njit, prange

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

        # Typically we'll sample about the previous best action plan
        self._previous_optimal_action_plan = np.zeros((H, action_size))

        # Will need a path to follow, but we want it to be
        # updated separately (changeable)
        self.path_xyz = None

        # Are we going to have logging? Defaultly no
        self.log_folder = None

        # Lambda is the temperature of the softmax
        # infinity selects the best action plan, 0 selects uniformly
        self.lambda_ = lambda_

    def reward(
        self,
        state_trajectory_plan,
        action_trajectory_plan,
    ):
        """
        Given a potential plan, return a scalar reward (higher is better)
        """

        p = state_trajectory_plan[:, 0:3]
        q = state_trajectory_plan[:, 3:7]
        v = state_trajectory_plan[:, 7:10]
        w = state_trajectory_plan[:, 10:]

        # We compute the shortest distance between every point in
        # our plan and the path we want to follow, and add this distance
        # as a cost (ideally they'd be aligned)
        path_deviation = 0
        # Do an average so its interpretable across different path
        # lengths
        decay = 1
        path_deviation = np.mean([
            (decay ** i) * utils.geometric.shortest_distance_between_path_and_point(
                self.path_xyz,
                p[i],
            ) for i in range(self.H)
        ])

        # Stay upright
        desired_xy_euler_angles = np.array([0, 0])
        xy_euler_angles = utils.geometric.quaternion_to_euler_angles_rad(
            q[:, 0],
            q[:, 1],
            q[:, 2],
            q[:, 3],
        )[:, :2]
        upright_deviation = np.mean(np.linalg.norm(xy_euler_angles - desired_xy_euler_angles, axis=1))

        # Minimize angular velocity, especially in the z
        penalty = np.array([0, 0, 1])
        angular_velocity_deviation = np.mean(np.linalg.norm(w * penalty, axis=1))

        # Stay at the origin
        origin_deviation = np.mean(np.linalg.norm(p - np.array([0, 0, 10]), axis=1))

        # Goal state
        goal_state = np.array([0, 0, 10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        goal_deviation = np.sum([
            (decay ** i) * np.linalg.norm(state_trajectory_plan[i] - goal_state) for i in range(self.H)
        ])

        # Maximize velocity 
        # TODO doesn't work - can just fall very far
        mean_velocity = np.mean(np.linalg.norm(v, axis=1))

        # Punish action discontinuities
        action_discontinuity = np.mean(np.linalg.norm(np.diff(action_trajectory_plan, axis=0), axis=1))

        # Are we going int he right direction along the path? This being positive
        # should be rewarded
        forwardness = utils.geometric.forwardness_of_path_a_wrt_path_b(p, self.path_xyz)

        # Try to keep the velocity magnitude under control
        desired_velocity = 1
        velocity_deviation = np.mean(np.abs(np.linalg.norm(v, axis=1) - desired_velocity))

        # We prefer to take no actions (control inputs are expensive)
        action_magnitude = np.mean(np.linalg.norm(action_trajectory_plan, axis=1))

        # Assemble the reward
        # If we only minimize path deviation, we get a straight line, but the angular velocity becomes massive
        # The angular velocity deviation can be small
        # The forwardness can be small - just enough to nudge us in the right direction
        reward = -5*path_deviation - 0.025*angular_velocity_deviation #- 1*velocity_deviation #- 0.05*angular_velocity_deviation - 1*velocity_deviation #+ 0.2*forwardness   # +   #- action_magnitude
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
        std_dev = (self.action_ranges[:,1] - self.action_ranges[:,0]) / 3
        # This will draw K * H * action_size samples
        samples = np.random.normal(mean, std_dev, (self.K, self.H, self.action_size))
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

        # We'll simulate those actions using dynamics and figure
        # out the states
        state_plans  = np.zeros((K, H, self.state_size))
        # Start every future in the current state
        #state_plans[:, 0] = np.array(state_history[-1])

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

        # Actually don't want the first state (it's the current state)
        # state_plans = state_plans[:, 1:]
            
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
                


