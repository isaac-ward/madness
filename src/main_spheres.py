import numpy as np
from tqdm import tqdm
import scipy
import os
import pickle
import csv
import time
import copy
import matplotlib.pyplot as plt
import cupy as cp

import utils.general
import utils.logging
import utils.geometric
import dynamics
from environment import Environment
from mapping import Map
from agent import Agent
from visual import Visual
from policies.simple import PolicyNothing, PolicyRandom, PolicyConstant
from policies.mppi import PolicyMPPI
import policies.samplers
import standard

# TODO implement wandb to allow for more efficient grid searching of parameters

def build_cube(center:np.ndarray,radius:int):
    """
    """
    # Get a list of all boundary points
    points_horizon = list()

    # Create boundary of cube
    x_up = center[0] + radius
    x_down = center[0] - radius
    y_up = center[0] + radius
    y_down = center[0] - radius
    z_up = center[0] + radius
    z_down = center[0] - radius

    # Compile list of points to check
    # Fix x
    for _i in range(2):
        # Fix x as either the upper or lower bound
        if _i % 2 == 0:
            _x = x_up
        else:
            _x = x_down
        
        # Iterate through each new point at the fixed x value
        for _y in range(y_down,y_up+1):
            for _z in range(z_down,z_up+1):
                # New point to check
                new_point = np.array([_x,_y,_z])
                # Add new point if not already checked
                if not any(np.array_equal(new_point, point) for point in points_horizon):
                    points_horizon.append(new_point)

    # Fix y
    for _i in range(2):
        # Fix y as either the upper or lower bound
        if _i % 2 == 0:
            _y = y_up
        else:
            _y = y_down
        
        # Iterate through each new point at the fixed y value
        for _x in range(x_down,x_up+1):
            for _z in range(z_down,z_up+1):
                # New point to check
                new_point = np.array([_x,_y,_z])
                # Add new point if not already checked
                if not any(np.array_equal(new_point, point) for point in points_horizon):
                    points_horizon.append(new_point)

    # Fix z
    for _i in range(2):
        # Fix z as either the upper or lower bound
        if _i % 2 == 0:
            _z = z_up
        else:
            _z = z_down
        
        # Iterate through each new point at the fixed z value
        for _x in range(x_down,x_up+1):
            for _z in range(y_down,y_up+1):
                # New point to check
                new_point = np.array([_x,_y,_z])
                # Add new point if not already checked
                if not any(np.array_equal(new_point, point) for point in points_horizon):
                    points_horizon.append(new_point)
    
    return points_horizon

def find_max_cube(center:np.ndarray):
    """
    """
    # Begin checking points
    no_collision = True
    radius = 0
    while (no_collision):
        # Increase radius by 1 grid point
        radius += 1
        print(radius)

        # Build a cube with this radius
        points_horizon = build_cube(center,radius)
        
        # Given the list of points to check, check if occupied
        # TODO batch check function
        for _i in range(len(points_horizon)):
            if map_.is_voxel_occupied(points_horizon[_i]):
                no_collision = False
                break
    
    # Reduce radius to safe occupancy
    radius -= 1

    return radius

if __name__ == "__main__":

    # Seed everything
    utils.general.random_seed(42)

    # Are we using GPU? 
    # NOTE: suggest false for now because it's slow
    use_gpu_if_available = False
    keep_policy_logs = True

    # Will log everything to here
    log_folder = utils.logging.make_log_folder(name="run")

    # The environment follows some true dynamics, and the agent
    # has an internal model of the environment
    dyn = standard.get_standard_dynamics_quadcopter_3d()

    # Create a map representation
    #map_ = standard.get_standard_map()
    map_ = standard.get_28x28x28_at_111()
    #map_ = standard.get_28x28x28_at_111_with_obstacles()

    # Start and goal states
    # NOTE: The following utility finds two random points - it doesn't check for collisions!
    # If you're using a map with invalid positions then you might need to specify the start and goal states manually
    state_initial, state_goal = Environment.get_two_states_separated_by_distance(map_, min_distance=26)

    # # Generate a path from the initial state to the goal state
    """xyz_initial = state_initial[0:3]
    xyz_goal = state_goal[0:3]
    path_xyz = np.array([xyz_initial, xyz_goal])
    path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*4) # Ultra safe
    path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz)"""
    
    # Get the current starting position
    start_point = map_.metres_to_voxel_coords(state_initial[:3])

    # Create a list to hold centers and radii
    centers = list()
    radii = list()

    # Add starting point to list
    centers.append(start_point)
    radii.append(find_max_cube(start_point))

    # Pick next point
    

    # Create the environment
    num_seconds = 16
    num_steps = int(num_seconds / dyn.dt)
    environment = Environment(
        state_initial=state_initial,
        state_goal=state_goal,
        dynamics=dyn,
        map_=map_,
        episode_length=num_steps,
    )

    # ----------------------------------------------------------------

    # Build a sphere at the start
    pbar = tqdm(total=num_steps, desc="Running simulation")
    
    pbar.close()

    # ----------------------------------------------------------------

    # Log everything of interest
    environment.log(log_folder)

    # Render visuals
    visual = Visual(log_folder)
    visual.plot_histories()
    visual.render_video(desired_fps=25)