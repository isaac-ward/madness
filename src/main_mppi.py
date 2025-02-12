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
from environment import Environment
from mapping import Map
from agent import Agent
from visual import Visual
from policies.simple import PolicyNothing, PolicyRandom, PolicyConstant
from policies.mppi import PolicyMPPI
import policies.samplers
import standard
from benchmarking.benchmarker import Benchmarker

# TODO implement wandb to allow for more efficient grid searching of parameters

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
    dyn = standard.get_standard_dynamics()
    #dyn = standard.get_standard_dynamics_linear()

    # Create a map representation
    #map_ = standard.get_28x28x28_at_111()
    #map_ = standard.get_28x28x28_at_111_with_obstacles()
    map_ = standard.get_tunnels()
    #map_ = standard.get_chamber()

    # Start and goal states
    state_initial, state_goal = Environment.get_two_states_separated_by_distance(
        map_, 
        template=dyn.state_randomization_template(),
        min_distance=26,
    )

    # # Generate a path from the initial state to the goal state
    xyz_initial = state_initial[0:3]
    xyz_goal = state_goal[0:3]
    path_xyz = np.array([xyz_initial, xyz_goal])
    #path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter)
    #path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz)

    # Create the environment
    num_seconds = 20
    num_steps = int(num_seconds / dyn.dt)
    environment = Environment(
        state_initial=state_initial,
        state_goal=state_goal,
        dynamics=dyn,
        map_=map_,
        episode_length=num_steps,
    )

    # Create the agent, which has an initial state and a policy
    K = 500
    H = 50 #int(0.5/dynamics.dt), # X second horizon
    action_sampler = policies.samplers.RolloverGaussianActionSampler(K, H, dyn.action_ranges())
    policy = PolicyMPPI(
        dynamics=copy.deepcopy(dyn),
        action_sampler=action_sampler,
        K=K,
        H=H,
        lambda_=100,
        map_=map_,
        use_gpu_if_available=use_gpu_if_available,
    )
    policy.enable_logging(log_folder)
    policy.update_state_goal(state_goal)

    # Can now create an agent
    agent = Agent(
        state_initial=state_initial,
        policy=policy,
        state_size=dyn.state_size(),
        action_ranges=dyn.action_ranges(),
        zero_pad_state=dyn.zero_state(),
    ) 

    # ----------------------------------------------------------------

    # Create a benchmarker
    Benchmarker.run_single_agent_single_environment(
        agent=agent,
        environment=environment,
        state_initial=state_initial,
        state_goal=state_goal,
        log_folder=log_folder,
        render_videos=False
    )

    # ----------------------------------------------------------------

    # Clean up stored data for MPPI
    try:
        if not keep_policy_logs:
            policy.delete_logs()
    except:
        pass