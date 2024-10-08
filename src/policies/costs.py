import numpy as np
from tqdm import tqdm
import os
import cupy as cp
#from numba import njit, prange
import shutil

from scipy.stats.qmc import Sobol

import utils.geometric
import utils.general

def batch_cost(
    state_trajectory_plans,
    action_trajectory_plans,
    state_goal,
    map_,
):
    """
    Given a batch of plans, return a scalar cost (lower is better) for each

    state_trajectory_plans: (batch_size, H, state_size)
    action_trajectory_plans: (batch_size, H, action_size)
    """

    # What module to use? cupy or numpy?
    xp = cp.get_array_module(state_trajectory_plans)
    batch_size, _, _ = xp.shape(state_trajectory_plans)
    H = xp.shape(state_trajectory_plans)[1]

    p = state_trajectory_plans[:, :, 0:3]   # batch_size, H, 3
    r = state_trajectory_plans[:, :, 3:6]
    v = state_trajectory_plans[:, :, 6:9]
    w = state_trajectory_plans[:, :, 9:12]
    # Goal point
    goal_p = state_goal[0:3]
    goal_v = state_goal[6:9]
    goal_r = state_goal[3:6]
    goal_w = state_goal[9:12]

    # The cost function (negative reward) from the published work is:
    # a * distance_from_goal_at_T + SUM b * distance_from_goal_at_t + SUM c * collision_at_t

    # Distance of final point from goal
    goal_p_terms = xp.linalg.norm(p[:,-1,:] - goal_p, axis=1)
    goal_r_terms = xp.linalg.norm(r[:,-1,:] - goal_r, axis=1)
    goal_v_terms = xp.linalg.norm(v[:,-1,:] - goal_v, axis=1)
    goal_w_terms = xp.linalg.norm(w[:,-1,:] - goal_w, axis=1)

    # Distance along path to goal
    path_towards_goal_p_terms = xp.sum(xp.linalg.norm(p - goal_p, axis=2), axis=1)
    path_towards_goal_v_terms = xp.sum(xp.linalg.norm(v - goal_v, axis=2), axis=1)

    # Collision/oob check
    # It is extremely important that this check be done in parallel, it
    # uses ckdtrees and is very slow otherwise
    flat_p = p.reshape((batch_size * H, 3))
    invalid_terms = xp.sum(map_.batch_is_not_valid(flat_p, collision_radius=1).reshape((batch_size, H)), axis=1)

    # # Minimize velocity
    # velocity_terms = xp.sum(xp.linalg.norm(v, axis=2), axis=1)

    # # Minimize angular velocity
    # angular_velocity_terms = xp.sum(xp.linalg.norm(w, axis=2), axis=1)

    # Term for minimizing control effort
    control_effort_terms = xp.sum(xp.linalg.norm(action_trajectory_plans, axis=2), axis=1)

    # Assemble
    cost = \
        1000 * goal_p_terms + \
        100 * path_towards_goal_p_terms + \
        10 * path_towards_goal_v_terms + \
        10000 * invalid_terms + \
        0 * goal_r_terms + \
        0 * goal_v_terms + \
        0 * goal_w_terms + \
        0 * control_effort_terms
    
    # Assert the correct shape
    assert xp.shape(cost) == (batch_size,), f"Cost shape is {xp.shape(cost)}, expected {(batch_size,)}"
    
    # Big numbers will be exponentiated to zero given our floating 
    # number precision
    # NOTE: choosing the division factor here is important - inspect the loss weight
    # computations and pick a number that doesn't overly flatten the log optimality plot
    return cost / 1e3