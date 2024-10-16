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
from sdf import Environment_SDF, Sphere_SDF

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
    #state_initial, state_goal = Environment.get_two_states_separated_by_distance(map_, min_distance=26)
    state_initial = np.zeros(12)
    state_initial[:3] = 5
    state_goal = np.zeros(12)
    state_goal[:3] = 25

    # # Generate a path from the initial state to the goal state
    xyz_initial = state_initial[0:3]
    xyz_goal = state_goal[0:3]
    path_xyz = np.array([xyz_initial, xyz_goal])
    path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*4) # Ultra safe
    path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz)

    # Create a list to hold centers and radii
    sdfs = Environment_SDF(dyn)
    sdfs.characterize_env_with_spheres_perturbations(
        start_point_meters=xyz_initial,
        end_point_meters=xyz_goal,
        path_xyz=path_xyz_smooth,
        map_env=map_,
        max_spheres=500,
        randomness_deg=45
    )
    """sdfs.characterize_env_with_spheres_xyzpath(
        start_point_meters=xyz_initial,
        end_point_meters=xyz_goal,
        path_xyz=path_xyz_smooth,
        map_env=map_,
        max_spheres=50
    )"""

    print(len(sdfs.sdf_list))

    # ----------------------------------------------------------------
    # Logging from here on
    # ----------------------------------------------------------------

    # Save the cubes
    utils.logging.pickle_to_filepath(
        os.path.join(log_folder, "signed_distance_function.pkl"),
        sdfs,
    )

    utils.logging.pickle_to_filepath(
        os.path.join(os.path.join(log_folder, "environment"), "path_xyz_smooth.pkl"),
        path_xyz_smooth,
    )

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

    # No simulation

    # ----------------------------------------------------------------

    # Log everything of interest
    environment.log(log_folder)

    # Render visuals
    visual = Visual(log_folder)
    #visual.plot_histories()
    #visual.render_video(desired_fps=25)
    visual.plot_environment()
