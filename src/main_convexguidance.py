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
from sdf import Environment_SDF
from policies.cvxguidance import SCPSolver, Trajectory

if __name__ == "__main__":

    # Seed everything
    utils.general.random_seed(42)

    # Will log everything here
    log_folder = utils.logging.make_log_folder(name="run")

    # The environment follows some true dynamics, and the agent
    # has an internal model of the environment
    dyn = standard.get_standard_dynamics_quadcopter_3d()

    # Create a map representation
    #map_ = standard.get_standard_map()
    # map_ = standard.get_28x28x28_at_111()

    print("let's get that map boys")
    map_ = standard.get_28x28x28_at_111_with_obstacles()
    print("got it!")

    # Start and goal states
    # NOTE: The following utility finds two random points - it doesn't check for collisions!
    # If you're using a map with invalid positions then you might need to specify the start and goal states manually
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
    print(path_xyz_smooth.shape)
    K = path_xyz_smooth.shape[0] - 1

    trajInit = Trajectory()

    k = dyn.thrust_coef
    m = dyn.mass
    g = dyn.g
    w_trim = np.sqrt(m*g/(4*k))

    trajInit.state = np.zeros((K+1, dyn.state_size()))
    trajInit.state[:,:3] = path_xyz_smooth
    trajInit.action = w_trim * np.ones((K, dyn.action_size()))

    # Create a list to hold centers and radii
    sdfs = Environment_SDF(dyn)
    print("we are here. we are alive. and we are loving it.")
    sdfs.characterize_env_with_spheres_perturbations(
        start_point_meters=xyz_initial,
        end_point_meters=xyz_goal,
        path_xyz=path_xyz_smooth,
        map_env=map_,
        max_spheres=500,
        randomness_deg=45
    )

    # initialize SCP solver object
    scp = SCPSolver(K = K,
                    dynamics=copy.deepcopy(dyn),
                    sdf = sdfs,
                    trajInit=trajInit,
                    maxiter = 50,
                    eps_dyn=50,
                    eps_sdf=10,
                    sig = 1.,
                    rho=1.)

    # Setup SCP iterations manually until exit condition is implemented
    state_goal = np.zeros(dyn.state_size())
    state_goal[:3] = xyz_goal
    state_history = np.zeros(dyn.state_size())
    state_history[:3] = xyz_initial
    scp.solve(state_goal=state_goal,
                state_history=state_history[np.newaxis,:])
    
    # Extract entire state history from solver
    state_history = scp.state.value
    # Extract euclidean coordinates of drone path from state history
    position_history = state_history[:,:3]

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

    # Log everything of interest
    environment.log(log_folder)

    # Log the cubes
    utils.logging.pickle_to_filepath(
        os.path.join(log_folder, "signed_distance_function.pkl"),
        sdfs,
    )

    # Log the A* path
    utils.logging.pickle_to_filepath(
        os.path.join(os.path.join(log_folder, "environment"), "path_xyz_smooth.pkl"),
        path_xyz_smooth,
    )

    # Log the CVX path
    utils.logging.pickle_to_filepath(
        os.path.join(os.path.join(log_folder, "environment"), "path_xyz_cvx.pkl"),
        position_history,
    )

    # Render visuals
    visual = Visual(log_folder)
    #visual.plot_histories()
    #visual.render_video(desired_fps=25)
    visual.plot_environment()