import numpy as np
import os

import utils.logging

class Environment:
    def __init__(
        self,
        state_initial,
        dynamics_model,
        map_,
    ):
        """
        The dynamics model, or equivalently, the transition function, of the environment
        tells us how the state evolves over time given the current state and action
        """
        
        # Save the dynamics model and the map
        self.dynamics_model = dynamics_model
        self.map = map_ # TODO must use this
        # TODO reward model

        # Although the map interaction is technically part of the 
        # dynamics model, we'll keep it separate for clarity

        # Keep track of the history of states and actions
        self.state_history = [state_initial]
        self.action_history = []

    def step(
        self,
        action,
    ):
        """
        Advance the environment by one step, and return the new state
        """
        # Get the new state
        state = self.state_history[-1]
        new_state = self.dynamics_model.step(state, action)
        # Log everything
        self.action_history.append(action)
        self.state_history.append(new_state)
        return new_state

    def log(
        self,
        folder,
    ):

        folder_environment = os.path.join(folder, "environment")

        # Save the history
        utils.logging.save_state_and_action_trajectories(
            folder_environment,
            self.state_history,
            self.action_history,
        )

        # Save the dynamics object
        utils.logging.pickle_to_filepath(
            os.path.join(folder_environment, "dynamics.pkl"),
            self.dynamics_model,
        )

        # Save the map
        utils.logging.pickle_to_filepath(
            os.path.join(folder_environment, "map.pkl"),
            self.map,
        )