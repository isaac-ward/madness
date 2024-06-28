import numpy as np

from tqdm import tqdm

import utils.geometric

class PolicyMPPI:
    def __init__(
        self,
        state_size,
        action_size,
        dynamics,
        K,
        H,
    ):
        """
        Roll out a bunch of random actions and select the best one
        """
        self.state_size = state_size
        self.action_size = action_size
        self.dynamics = dynamics
        self.K = K
        self.H = H

        self.path_xyz = None

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
        path_deviation = np.mean([
            utils.geometric.shortest_distance_between_path_and_point(
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

        # Minimize angular velocity
        angular_velocity_deviation = np.mean(np.linalg.norm(w, axis=1))

        # Assemble the reward
        reward = -1*path_deviation -5*upright_deviation - 10*angular_velocity_deviation
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

        # For convenience
        K = self.K
        H = self.H

        # Sample actions from some distribution
        action_plans = np.random.randn(K, H, self.action_size) * 3

        # We'll simulate those actions using dynamics and figure
        # out the states
        state_plans  = np.zeros((K, H + 1, self.state_size))
        # Start every future in the current state
        state_plans[:, 0] = np.array(state_history[-1])

        # We will compute rewards for each future
        rewards = np.zeros(K)

        # Roll out futures in parallel
        for h in tqdm(range(H), desc="Rolling out futures", leave=False, disable=True):
            # Compute the next states
            state_plans[:, h + 1] = self.dynamics.step(
                state_plans[:, h],
                action_plans[:, h],
            )
            
        # Compute all rewards
        # TODO parallelize
        for k in range(K):
            rewards[k] = self.reward(
                state_plans[k],
                action_plans[k],
            )

        # Select the best plan and return the immediate action from that plan
        best_action_plan = action_plans[np.argmax(rewards)]
        best_action = best_action_plan[0]
        return best_action
                


