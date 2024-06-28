import numpy as np
import utils.logging
import os

class Agent:
    def __init__(
        self,
        state_initial,              # initial state
        policy,                     # how to determine the optimal next action
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
        self.state_history = [state_initial]
        self.action_history = []

    def act(self):
        """
        Use the policy to determine the next action to take
        """
        # Will be in the form s_0, a_0, s_1, a_1, ..., s_n
        # TODO rewards
        num_states_desired  = self.lookback  
        num_actions_desired = self.lookback - 1

        # Helper to fill a padded zeros matrix with what is available
        def fill_available_history(shape, available_history):
            num_desired = shape[0]
            num_available = len(available_history)
            if num_available > num_desired:
                return available_history[-num_desired:]
            filled = np.zeros(shape)
            if num_available > 0:
                filled[-num_available:] = available_history
            return filled

        # Fill em up
        # TODO this should not be hardcoded
        state_history = fill_available_history((num_states_desired, self.policy.state_size), self.state_history)
        action_history = fill_available_history((num_actions_desired, self.policy.action_size), self.action_history)

        # Provide the policy with the history to determine an action
        action = self.policy.act(
            state_history,
            action_history,
        )

        # Save the action and return
        self.action_history.append(action)
        return action

    def observe(self, state):
        """
        Observe and track the latest state
        """
        # TODO partial observability
        # The environment keeps track of the state too, because they may 
        # be different if the agent is not perfectly informed
        self.state_history.append(state)

    def log(
        self,
        folder,
    ):
        utils.logging.log_histories(
            os.path.join(folder, "agent"),
            self.state_history,
            self.action_history,
        )


