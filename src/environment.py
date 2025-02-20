import numpy as np
import os
import warnings 
from tqdm import tqdm

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
        self.close_enough_position = self.dynamics.diameter # m
        self.close_enough_orientation = 10000 # radians
        self.close_enough_velocity = 0.1 # m/s
        self.close_enough_angular_velocity = 10000 # rad/s

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
        state = self.state_history_tracker.get_last_item()
        new_state = self.dynamics.step(state, action)
        # Log everything
        self.action_history_tracker.append(action)
        self.state_history_tracker.append(new_state)

        def is_goal_met(new_state):
            # Is position close enough?
            is_position_goal_met = np.linalg.norm(new_state[0:3] - self.state_goal[0:3]) < self.close_enough_position
            # Is orientation close enough?
            is_orientation_goal_met = np.linalg.norm(new_state[3:6] - self.state_goal[3:6]) < self.close_enough_orientation 
            # Is velocity close enough?
            is_velocity_goal_met = np.linalg.norm(new_state[6:9] - self.state_goal[6:9]) < self.close_enough_velocity
            # Is angular velocity close enough?
            is_angular_velocity_goal_met = np.linalg.norm(new_state[9:12] - self.state_goal[9:12]) < self.close_enough_angular_velocity
            return is_position_goal_met and is_orientation_goal_met and is_velocity_goal_met and is_angular_velocity_goal_met

        # Are we done? If we're out of time or in an invalid state, we're done
        done_flag = False
        done_message = ""
        if len(self.state_history_tracker) == (self.episode_length):
            done_flag = True
            done_message = "Ran out of steps"
        elif self.map.is_not_valid(new_state[0:3], collision_radius=self.close_enough_position):
            done_flag = True
            done_message = f"Entered an invalid state (OOB) or collided with an obstacle to within {self.close_enough_position} m"
        elif is_goal_met(new_state):
            done_flag = True
            done_message = f"Reached the goal state to within:\n"
            done_message += f"\t-position {self.close_enough_position} m\n"
            done_message += f"\t-orientation {self.close_enough_orientation} rad\n"
            done_message += f"\t-velocity {self.close_enough_velocity} m/s\n"
            done_message += f"\t-angular velocity {self.close_enough_angular_velocity} rad/s"
        return new_state, done_flag, done_message

    @staticmethod
    def get_two_states_separated_by_distance(
        map_,
        template,
        min_distance,
        rng=None,
    ):
        """
        Useful for resetting the environment with the initial state and goal

        Template is a list of items, the letter R denotes randomize, letter X Y Z
        denotes positional randomization, and the rest are fixed values

        obstacle_collision_distance is the distance at which a point should be
        considered in collision with an obstacle
        
        Extents is a list of 3 tuples of (min, max) for each dimension
        """

        if rng is None:
            #warnings.warn("No random number generator provided, using default")
            rng = np.random.default_rng()            

        extents = map_.extents_metres_xyz

        # Has to be larger than the voxelization by 2 (nyquist)
        obstacle_collision_distance = map_.voxel_per_x_metres * 2
        
        # Get a random state
        def get_random_state(extents):
            state = np.zeros(len(template))
            for i, item in enumerate(template):
                # This gets the numbers in metres
                if item == "X":
                    state[i] = rng.uniform(low=extents[i][0], high=extents[i][1])
                elif item == "Y":
                    state[i] = rng.uniform(low=extents[i][0], high=extents[i][1])
                elif item == "Z":
                    state[i] = rng.uniform(low=extents[i][0], high=extents[i][1])
                else:
                    state[i] = item
            return state
        
        allowed_attempts = 1000
        pbar = tqdm(total=allowed_attempts, desc="Finding start and goal states")
        
        # Get a random goal state that is at least min_distance away
        def _far_apart_enough(state_goal, state_initial, min_distance):
            return np.linalg.norm(state_goal[0:3] - state_initial[0:3]) > min_distance
        def _in_collision(state):
            # not valid = in collision
            #return map_.is_not_valid(state[0:3], collision_radius=obstacle_collision_distance)
            #return map_.batch_is_collision_metres_xyz(state[0:3], collision_radius=obstacle_collision_distance)
            # Convert to voxel units
            vc = map_.metres_to_voxel_coords(state[0:3])
            # print(vc)
            # print(map_.voxel_grid[vc[0], vc[1], vc[2]])
            return map_.voxel_grid[vc[0], vc[1], vc[2]] != 0
        def _satisfied(state_goal, state_initial, min_distance):
            # Go through all checks and determine failure reason
            failure_messages = []
            satisfied = True
            if _in_collision(state_initial):
                failure_messages.append(f"start collision ({state_initial[0]:.1f}, {state_initial[1]:.1f}, {state_initial[2]:.1f})")
                satisfied = False
            if _in_collision(state_goal):
                failure_messages.append(f"goal collision ({state_goal[0]:.1f}, {state_goal[1]:.1f}, {state_goal[2]:.1f})")
                satisfied = False
            if not _far_apart_enough(state_goal, state_initial, min_distance):
                failure_messages.append(f"too close ({np.linalg.norm(state_goal[0:3] - state_initial[0:3]):.2f} m)")
                satisfied = False
            if satisfied:
                desc_string = "Start points satisfied"
            else:
                desc_string = ", ".join(failure_messages)
            pbar.set_description(desc_string)
            return satisfied               
        
        # Keep trying until we get it
        state_initial = get_random_state(extents)
        state_goal = get_random_state(extents)
        while not _satisfied(state_goal, state_initial, min_distance) and pbar.n < allowed_attempts:
            state_initial = get_random_state(extents)
            state_goal = get_random_state(extents)
            pbar.update(1)

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
            np.array([self.state_history_tracker.get_first_item(), self.state_goal]),
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