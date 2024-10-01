import numpy as np
import cvxpy as cvx
import cupy as cp


import utils.geometric
import utils.general
import policies.costs
import policies.samplers

# TODO
# import sdf.sdf write and import sdf class that holds sphere locations and sizes (plus the field?)

class PolicyConvex:
    def __init__(
            self,
            dynamics,
            sdf_nearest
            ):
        self.dynamics = dynamics
        
        self.sdf = sdf_nearest
        self.state_goal = None

    def update_state_goal(
        self,
        state_goal,
    ):
        """
        Update the path to follow
        """
        self.state_goal = state_goal

    def update_sdf(
            self,
            sdf_nearest
    ):
        self.sdf = sdf_nearest
        

    def new_traj(
            self,
            state_goal,
            sdf_nearest
    ):
        self.update_state_goal(state_goal)
        self.update_sdf(sdf_nearest)

    def act(
            self
            state_history
    ):
        
        optimal_action_history, optimal_state_history = 
        return optimal_action_history, optimal_state_history