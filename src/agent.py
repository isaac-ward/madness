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
        action_size,
    ):
        """
        We'll start somewhere, and then we'll use the policy to determine
        the next action to take
        """
        
        self.policy = policy
        # How many steps in the past does the policy have
        # access to?
        self.lookback = 32

        # We'll also track the history of the states
        self.state_history_tracker  = ItemHistoryTracker(item_shape=(state_size,))
        self.action_history_tracker = ItemHistoryTracker(item_shape=(action_size,))

        # Shapes
        self.state_size = state_size
        self.action_size = action_size  

    def get_histories(self):
        num_states_desired  = self.lookback
        num_actions_desired = self.lookback - 1
        state_history = self.state_history_tracker.get_last_n_items_with_zero_pad(num_states_desired)
        action_history = self.action_history_tracker.get_last_n_items_with_zero_pad(num_actions_desired)
        return state_history, action_history

    def act(self):
        """
        Use the policy to determine the next action to take
        """
        
        state_history, action_history = self.get_histories()

        # Different policies require different inputs
        # TODO: 

        # Provide the policy with the history to determine an action
        action = self.policy.act(
            state_history,
            action_history,
        )

        # Save the action and return
        self.action_history_tracker.append(action)
        return action

    def observe(self, state):
        """
        Observe and track the latest state
        """
        # TODO partial observability
        # The environment keeps track of the state too, because they may 
        # be different if the agent is not perfectly informed
        self.state_history_tracker.append(state)

    def reset(self, state_initial):
        """
        Reset the agent to a new state
        """
        self.state_history_tracker.reset()
        self.action_history_tracker.reset()
        self.state_history_tracker.append(state_initial)

    def log(
        self,
        folder,
    ):
        utils.logging.save_state_and_action_trajectories(
            os.path.join(folder, "agent"),
            self.state_history_tracker.get_history(),
            self.action_history_tracker.get_history(),
        )


