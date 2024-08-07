import numpy as np
from tqdm import tqdm
import os
import cupy as cp
#from numba import njit, prange
import shutil

import utils.geometric
import utils.general
import policies.costs
import policies.samplers

class MPPIComputer:
    """
    An MPPI computer handles the sampling of actions, rollouts of actions
    wrt to some dynamics model, optimal action production (wrt some reward/costs),
    and logging
    """
    def __init__(
        self,
        dynamics,
        K,
        H,
        lambda_,
        map_,
        use_gpu_if_available=False,
    ):
        # Save the parameters
        self.dynamics = dynamics
        self.K = K
        self.H = H

        # Lambda is the temperature of the softmax
        # infinity selects the best action plan, 0 selects uniformly
        self.lambda_ = lambda_

        # We need a map to plan paths against (e.g. collision checking)
        self.map_ = map_

        # If we're using a GPU, we'll need to move some things over
        self.use_gpu_if_available = use_gpu_if_available

    def compute(
        self,
        state_history,
        action_history,
        state_goal,
        action_sampler,
    ):
        """
        Returns as follows: state_plans, action_plans, costs, optimal_state_plan, optimal_action_plan
        """

        # Sample actions from the action sampler
        action_plans = action_sampler.sample()

        # We'll simulate those actions using dynamics and figure
        # out the states
        state_plans = np.zeros((self.K, self.H, self.dynamics.state_size()))

        # We will compute costs for each future
        costs = np.zeros((self.K,))

        # Roll out futures in parallel (needs to be serial because we need to compute
        # the state at t=0 before we can compute the state at t=1)
        for h in tqdm(range(self.H), desc="Rolling out futures", leave=False, disable=True):
            # Compute the next states
            state_plans[:, h] = self.dynamics.step(
                # If it's our first computation, start at our last known state, otherwise
                # start at the last computed state
                np.tile(state_history[-1], (self.K, 1)) if h == 0 else state_plans[:, h - 1],
                action_plans[:, h],
            )
            
        # Compute all costs
        costs = policies.costs.batch_cost(
            state_plans, 
            action_plans,
            state_goal,
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
        optimal_index = np.argmin(costs)
        optimal_action_plan = action_plans[optimal_index]
        optimal_state_plan  = state_plans[optimal_index]

        # If we're using a action sampler that uses the previous optimal action plan
        # then we'll update it here
        if isinstance(action_sampler, policies.samplers.RolloverGaussianActionSampler):
            action_sampler.update_previous_optimal_action_plan(optimal_action_plan)

        return state_plans, action_plans, costs, optimal_state_plan, optimal_action_plan

class PolicyMPPI:
    def __init__(
        self,
        dynamics,
        action_sampler,
        K,
        H,
        lambda_,
        map_,
        use_gpu_if_available=False,
    ):
        # Use an MPPI computer to do the heavy lifting
        self.computer = MPPIComputer(
            dynamics=dynamics,
            K=K,
            H=H,
            lambda_=lambda_,
            map_=map_,
            use_gpu_if_available=use_gpu_if_available,
        )

        # What are we sampling actions from?
        self.action_sampler = action_sampler

        # Defaultly no logging
        self.log_folder = None

        # Defaultly no goal
        self.state_goal = None

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

    def act(
        self,
        state_history,
        action_history,
    ):

        # Check if we have a path to follow
        if self.state_goal is None:
            raise ValueError(f"{self.__class__.__name__} requires a goal state to follow")
        
        # Get the optimal action and other logging information
        state_plans, action_plans, costs, optimal_state_plan, optimal_action_plan = self.computer.compute(
            state_history,
            action_history,
            self.state_goal,
            self.action_sampler,
        )
        optimal_action = optimal_action_plan[0]

        # ----------------------------------------------------------------
        # Logging from here on
        # ----------------------------------------------------------------

        # Log the state and action plans alongside the costs, 
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
            # Save the costs
            utils.logging.pickle_to_filepath(
                os.path.join(folder, "costs.pkl"),
                costs,
            )
            # If we're logging we will want to see what the optimal plan was
            optimal_state_plan = np.zeros((self.computer.H, self.computer.dynamics.state_size()))
            for h in range(self.computer.H):
                optimal_state_plan[h] = self.computer.dynamics.step(
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
                


