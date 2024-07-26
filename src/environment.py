import numpy as np
import os

import utils.logging

class Environment:
    def __init__(
        self,
        state_initial,
        state_goal,
        dynamics_model,
        map_,
        episode_length,
    ):
        """
        The dynamics model, or equivalently, the transition function, of the environment
        tells us how the state evolves over time given the current state and action
        """
        
        # Save the dynamics model and the map
        self.dynamics_model = dynamics_model
        self.map = map_ # TODO must use this
        # TODO reward model

        # Keep track of the number of steps until the episode is done
        self.episode_length = episode_length

        # Although the map interaction is technically part of the 
        # dynamics model, we'll keep it separate for clarity

        # If we're this close to an obstacle or the goal, we're done
        self.close_enough_radius = self.dynamics_model.diameter / 2

        # Keep track of the history of states and actions
        self.reset(state_initial, state_goal)

    def step(
        self,
        action,
    ):
        """
        Advance the environment by one step, and return the new state, plus
        a done flag
        """
        # Get the new state
        state = self.state_history[-1]
        new_state = self.dynamics_model.step(state, action)
        # Log everything
        self.action_history.append(action)
        self.state_history.append(new_state)
        # Are we done? If we're out of time or in an invalid state, we're done
        done_flag = False
        done_message = ""
        if len(self.state_history) > self.episode_length:
            done_flag = True
            done_message = "Ran out of steps"
        elif self.map.is_not_valid(new_state[0:3], collision_radius=self.close_enough_radius):
            done_flag = True
            done_message = "Entered an invalid state (OOB) or collided with an obstacle"
        elif np.linalg.norm(new_state[0:3] - self.state_goal[0:3]) < self.close_enough_radius:
            # TODO should this be a full state comparison?
            done_flag = True
            done_message = "Reached the goal position"
        return new_state, done_flag, done_message
    
    def reset(
        self,
        state_initial,
        state_goal,
    ):
        """
        Reset the environment to the initial state
        """
        self.state_history = [state_initial]
        self.action_history = []
        self.state_goal = state_goal

    def _get_last_n_items(self, array, n):
        """
        Get the last n states, padding with zero if not enough states,
        and return as a numpy array
        """
        items = array[-n:]
        while len(items) < n:
            zeros = np.zeros_like(items[0])
            items = [zeros] + items
        return np.array(items)
    
    def get_last_n_states(self, n):
        return self._get_last_n_items(self.state_history, n)
    
    def get_last_n_actions(self, n):
        return self._get_last_n_items(self.action_history, n)

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