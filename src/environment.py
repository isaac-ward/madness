import numpy as np
import os
import warnings 

import utils.logging
from utils.general import ItemHistoryTracker

class Environment:
    def __init__(
        self,
        state_initial,
        state_goal,
        dynamics,
        map_,
        episode_length,
    ):
        """
        The dynamics model, or equivalently, the transition function, of the environment
        tells us how the state evolves over time given the current state and action
        """
        
        # Save the dynamics model and the map
        self.dynamics = dynamics
        self.map = map_ # TODO must use this
        # TODO reward model

        # Keep track of the number of steps until the episode is done
        self.episode_length = episode_length

        # Although the map interaction is technically part of the 
        # dynamics model, we'll keep it separate for clarity

        # If we're this close to an obstacle or the goal, we're done
        self.close_enough_radius = self.dynamics.diameter / 2

        # Keep track of the history of states and actions
        self.state_history_tracker  = ItemHistoryTracker(item_shape=(self.dynamics.state_size(),))
        self.action_history_tracker = ItemHistoryTracker(item_shape=(self.dynamics.action_size(),))
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
        state = self.state_history_tracker.get_last_n_items_with_zero_pad(1)[0]
        new_state = self.dynamics.step(state, action)
        # Log everything
        self.action_history_tracker.append(action)
        self.state_history_tracker.append(new_state)
        # Are we done? If we're out of time or in an invalid state, we're done
        done_flag = False
        done_message = ""
        # TODO Does this need to be -1?
        if len(self.state_history_tracker) == self.episode_length - 1:
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

    @staticmethod
    def get_two_states_separated_by_distance(
        map_,
        min_distance,
        rng=None,
    ):
        """
        Useful for resetting the environment with the initial state and goal
        
        Extents is a list of 3 tuples of (min, max) for each dimension
        """

        rng = np.random.default_rng() if (rng is None) else rng

        extents = map_.extents_metres_xyz
        
        # Get a random state
        state_initial = np.array([
            rng.uniform(low=extents[0][0], high=extents[0][1]),
            rng.uniform(low=extents[1][0], high=extents[1][1]),
            rng.uniform(low=extents[2][0], high=extents[2][1]),
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        ])
        
        # Get a random goal state that is at least min_distance away
        state_goal = state_initial
        attempts = 1000
        while np.linalg.norm(state_goal[0:3] - state_initial[0:3]) < min_distance and attempts > 0:
            state_goal = np.array([
                rng.uniform(low=extents[0][0], high=extents[0][1]),
                rng.uniform(low=extents[1][0], high=extents[1][1]),
                rng.uniform(low=extents[2][0], high=extents[2][1]),
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
            ])
            attempts -= 1
        
        return state_initial, state_goal
    
    def reset(
        self,
        state_initial,
        state_goal,
    ):
        """
        Reset the environment to the initial state
        """
        self.state_history_tracker.reset()
        self.state_history_tracker.append(state_initial)
        self.action_history_tracker.reset()
        self.state_goal = state_goal

    def log(
        self,
        folder,
    ):

        folder_environment = os.path.join(folder, "environment")

        # Save the path (start->goal)
        utils.logging.pickle_to_filepath(
            f"{folder_environment}/path_xyz.pkl",
            np.array([self.state_history_tracker.history[0], self.state_goal]),
        )

        # If the state history is empty then provide a warning
        if len(self.state_history_tracker) == 0:
            warnings.warn("No state history to log (saving empty npz file regardless)")

        # Save the history
        utils.logging.save_state_and_action_trajectories(
            folder_environment,
            self.state_history_tracker.get_history(),
            self.action_history_tracker.get_history(),
        )

        # Save the dynamics object
        utils.logging.pickle_to_filepath(
            os.path.join(folder_environment, "dynamics.pkl"),
            self.dynamics,
        )

        # Save the map
        utils.logging.pickle_to_filepath(
            os.path.join(folder_environment, "map.pkl"),
            self.map,
        )