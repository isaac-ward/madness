import numpy as np
import utils.logging
import os

from utils.general import ItemHistoryTracker

class Agent:
    def __init__(
        self,
        state_initial,              # initial state
        policy,                     # how to determine the optimal next action
        state_size,
        action_ranges,
        zero_pad_state,
        filter = None,
    ):
        """
        We'll start somewhere, and then we'll use the policy to determine
        the next action to take
        """

        # Shapes
        self.state_size = state_size
        self.action_size = len(action_ranges)  

        # Action bounds
        self.action_ranges = action_ranges
        self.zero_pad_state = zero_pad_state
        
        self.policy = policy
        # How many steps in the past does the policy have
        # access to?
        self.lookback = 32

        self.filter = filter

        # We'll also track the history of the states
        # Need a special zero pad item for states because quaternions can't be all zero
        self.state_history_tracker  = ItemHistoryTracker(item_shape=(self.state_size,), zero_pad_item=zero_pad_state)
        self.action_history_tracker = ItemHistoryTracker(item_shape=(self.action_size,))

        # We will track the history of the belief state/state estimate
        self.belief_history_tracker = ItemHistoryTracker(item_shape=(self.state_size,), zero_pad_item=zero_pad_state)

    def get_histories(self):
        num_states_desired  = self.lookback
        num_actions_desired = self.lookback - 1
        state_history = self.state_history_tracker.get_last_n_items_with_zero_pad(num_states_desired)
        action_history = self.action_history_tracker.get_last_n_items_with_zero_pad(num_actions_desired)
        belief_history = self.belief_history_tracker.get_last_n_items_with_zero_pad(num_states_desired)
        return state_history, action_history, belief_history

    def act(self):
        """
        Use the policy to determine the next action to take
        """
        
        state_history, action_history, belief_history = self.get_histories()

        # Different policies require different inputs
        # TODO: 

        if self.filter is None:
            # Provide the policy with the history to determine an action
            action = self.policy.act(
                state_history,
                action_history,
                len(self.state_history_tracker)
            )
        else:
            action = self.policy.act(
                belief_history,
                action_history,
                len(self.belief_history_tracker)
            )

        # Clip the action to the action ranges
        action = np.clip(action, self.action_ranges[:, 0], self.action_ranges[:, 1])

        # Save the action and return
        self.action_history_tracker.append(action)
        return action

    def observe(self, state, action):
        """
        Observe and track the latest state
        """
        # TODO partial observability
        # The environment keeps track of the state too, because they may 
        # be different if the agent is not perfectly informed
        self.state_history_tracker.append(state)

        if not(self.filter is None):
            obs = self.filter.observe(state)
            belief = self.filter.filter(action,obs)
            self.belief_history_tracker.append(belief)
            return belief

    def reset(self, state_initial, state_goal):
        """
        Reset the agent to a new state
        """
        self.state_history_tracker.reset()
        self.action_history_tracker.reset()
        self.state_history_tracker.append(state_initial)
        self.policy.update_state_goal(state_goal)

    def log(
        self,
        folder,
    ):
        utils.logging.save_state_and_action_trajectories(
            os.path.join(folder, "agent"),
            self.state_history_tracker.get_history(),
            self.action_history_tracker.get_history(),
        )


