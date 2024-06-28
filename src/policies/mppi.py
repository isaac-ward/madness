import numpy as np

from tqdm import tqdm

import utils.geometric

class PolicyMPPI:
    def __init__(
        self,
        state_size,
        action_size,
        dynamics,
    ):
        """
        Roll out a bunch of random actions and select the best one
        """
        self.state_size = state_size
        self.action_size = action_size
        self.dynamics = dynamics
        self.path_xyz = None

    def reward(
        self,
        state_trajectory_plan,
        action_trajectory_plan,
    ):
        """
        Given a potential plan, return a scalar reward (higher is better)
        """

        # We compute the shortest distance between every point in
        # our plan and the path we want to follow, and add this distance
        # as a cost (ideally they'd be aligned)
        path_deviation = 0
        path_xyz_plan = state_trajectory_plan[:, 0:3]
        # Do an average so its interpretable across different path
        # lengths
        path_deviation = np.mean([
            utils.geometric.shortest_distance_between_path_and_point(
                self.path_xyz,
                path_xyz_plan[i],
            ) for i in range(len(path_xyz_plan))
        ])

        # Assemble the reward
        reward = -path_deviation
        return reward     

    def update_path_xyz(
        self,
        path_xyz,
    ):
        """
        Update the path to follow
        """
        self.path_xyz = path_xyz

    def act(
        self,
        state_history,
        action_history,
    ):

        # Check if we have a path to follow
        if self.path_xyz is None:
            raise ValueError(f"{self.__class__.__name__} requires a path to follow")

        K = 100
        H = 1

        # Sample random actions
        actions = np.random.randn(K, H, self.action_size)
        rewards = np.zeros(K)

        # Roll out the actions
        # TODO need batching
        for k in tqdm(range(K), desc="Rolling out actions", leave=False):
            state_trajectory_plan  = np.zeros((H, self.state_size))
            action_trajectory_plan = actions[k]
            curr_state = state_history[-1]
            for h in range(H):
                curr_state = self.dynamics.step(curr_state, action_trajectory_plan[h])
                state_trajectory_plan[h] = curr_state
        
            # Compute the reward
            rewards[k] = self.reward(state_trajectory_plan, action_trajectory_plan)

        # Select the best action
        best_action_trajectory_plan = actions[np.argmax(rewards)]
        best_action = best_action_trajectory_plan[0]
        return best_action
                


