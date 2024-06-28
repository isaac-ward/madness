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

    def reward(
        self,
        state_trajectory_plan,
        action_trajectory_plan,
    ):
        """
        Given a potential plan, return a scalar reward (higher is better)
        """

        # We want the quadcopter to hover at x=10, y=0, z=0, in a specific orientation
        desired_position = np.array([10, 0, 0])
        desired_orientation_euler_angles = np.array([0, 0, 0])

        # Look at the final state in the plan, is it our goal state?
        final_state = state_trajectory_plan[-1]
        position = final_state[:3]
        orientation = final_state[3:7]
        orientation_euler_angles = utils.geometric.quaternion_to_euler_angles_rad(*orientation)

        # But its especially bad to go below z=0
        position_penalty = np.array([1, 1, 10])

        # Compute the reward
        position_deviation = np.linalg.norm(position_penalty*(position - desired_position))
        orientation_deviation = np.linalg.norm(orientation_euler_angles - desired_orientation_euler_angles)
        reward = - 10 * position_deviation - orientation_deviation

    def act(
        self,
        state_history,
        action_history,
    ):
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
                


