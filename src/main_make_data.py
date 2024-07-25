import numpy as np
from tqdm import tqdm
import scipy
import os
import pickle
import csv
import time
import math
import copy
import matplotlib.pyplot as plt
import cupy as cp

import utils.general
import utils.logging
import dynamics
from environment import Environment
from map import Map
from agent import Agent
from visual import Visual
from policies.simple import PolicyNothing, PolicyRandom, PolicyConstant
from policies.mppi import PolicyMPPI

# TODO implement wandb to allow for more efficient grid searching of parameters

if __name__ == "__main__":

    # Seed everything
    utils.general.seed(0 + math.floor(time.time()))

    # Are we using GPU? 
    # NOTE: suggest false for now because it's slow
    use_gpu_if_available = False
    keep_policy_logs = True

    # Will log everything to here
    log_folder = utils.logging.make_log_folder(name="data")

    # The environment follows some true dynamics, and the agent
    # has an internal model of the environment
    dyn = dynamics.DynamicsQuadcopter3D(
        diameter=0.2,
        mass=2.5,
        Ix=0.5,
        Iy=0.5,
        Iz=0.3,
        # +z is down
        g=+9.81, 
        # higher makes it easier to roll and pitch
        thrust_coef=5,      
        # higher makes it easier to yaw
        drag_yaw_coef=5,   
        # higher values lower the max velocity
        drag_force_coef=5,   
        dt=0.025,
    )

    # Define the initial state of the system
    # Positions, rotations (euler angles), velocities, body rates
    state_initial = [
        np.random.uniform(1, 29), 
        np.random.uniform(1, 29),
        np.random.uniform(1, 29), 
        0, 0, 0,    
        0, 0, 0,    
        0, 0, 0
    ]

    # Create the environment
    map_ = Map(
        map_filepath="maps/empty.csv",
        voxel_per_x_metres=0.25,
        extents_metres_xyz=[
            [0, 30], 
            [0, 30], 
            [0, 30]
        ],
    )
    environment = Environment(
        state_initial=state_initial,
        dynamics_model=dyn,
        map_=map_,
    )

    # We need a path from the initial state to the goal state
    xyz_initial = np.array(state_initial[0:3])
    xyz_goal = xyz_initial
    attempts = 1000
    while np.linalg.norm(xyz_goal - xyz_initial) < 30 and attempts > 0:
        xyz_goal = [
            np.random.uniform(1, 29), 
            np.random.uniform(1, 29),
            np.random.uniform(1, 29)
        ]
        attempts -= 1

    # Create the agent, which has an initial state and a policy
    policy = PolicyMPPI(
        state_size=dyn.state_size(),
        action_size=dyn.action_size(),
        dynamics=copy.deepcopy(dyn),
        K=1000,
        H=50, #int(0.5/dynamics.dt), # X second horizon
        action_ranges=dyn.action_ranges(),
        lambda_=100,
        map_=map_,
        # Keep this off, it's slow
        use_gpu_if_available=use_gpu_if_available,
    )
    policy.enable_logging(log_folder)
    path_xyz = np.array([xyz_initial, xyz_goal])
    policy.update_path_xyz(path_xyz)

    agent = Agent(
        state_initial=state_initial,
        policy=policy,
    ) 

    # ----------------------------------------------------------------

    # Run the simulation for some number of steps
    num_seconds = 5
    num_steps = int(num_seconds / dyn.dt)
    pbar = tqdm(total=num_steps, desc="Running simulation")
    for i in range(num_steps):
        action = agent.act()
        state = environment.step(action)
        agent.observe(state)

        # Update the pbar with the current state and action
        p_string = ", ".join([f"{x:<5.1f}" for x in state[0:3]])
        v_string = f"{np.linalg.norm(state[6:9]):<4.1f}"
        w_string = ", ".join([f"{x:<4.1f}" for x in state[9:12]])
        a_string = ", ".join([f"{x:<4.1f}" for x in action])
        pbar.set_description(
            f"t={(i+1)*dyn.dt:.2f}/{num_seconds:.2f} | p=[{p_string}] | v={v_string} | w=[{w_string}] | a=[{a_string}] | gpu={'yes' if use_gpu_if_available else 'no'}")
        pbar.update(1)
    # Close the bar
    pbar.close()

    # ----------------------------------------------------------------

    # Log everything of interest
    agent.log(log_folder)
    environment.log(log_folder)
    utils.logging.pickle_to_filepath(
        f"{log_folder}/policy/path_xyz.pkl",
        path_xyz,
    )

    # Render visuals
    visual = Visual(log_folder)
    visual.plot_histories()
    visual.render_video()

    # Clean up stored data 
    try:
        if not keep_policy_logs:
            policy.delete_logs()
    except:
        pass