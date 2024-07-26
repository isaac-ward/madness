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
    utils.general.seed(0)

    # Are we using GPU? 
    # NOTE: suggest false for now because it's slow
    use_gpu_if_available = False
    keep_policy_logs = False

    # Will log everything to here
    log_folder = utils.logging.make_log_folder(name="run")

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

    # Create a map representation
    map_ = Map(
        map_filepath="maps/empty.csv",
        voxel_per_x_metres=0.25,
        extents_metres_xyz=[
            [0, 30], 
            [0, 30], 
            [0, 30]
        ],
    )

    # Define the initial state of the system
    # Positions, rotations (euler angles), velocities, body rates
    state_initial = [2.5, 2.5, 2.5,    0, 0, 0,    0, 0, 0,    0, 0, 0]

    # We need a path from the initial state to the goal state
    xyz_initial = state_initial[0:3]
    xyz_goal = [27.5, 27.5, 27.5]
    path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*4) # Ultra safe
    path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz)

    # Create the environment
    num_seconds = 8
    num_steps = int(num_seconds / dyn.dt)
    environment = Environment(
        state_initial=state_initial,
        state_goal=[*xyz_goal, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dynamics_model=dyn,
        map_=map_,
        episode_length=num_steps,
    )

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
    policy.update_path_xyz(path_xyz_smooth)

    # Can now create an agent
    agent = Agent(
        state_initial=state_initial,
        policy=policy,
    ) 

    # ----------------------------------------------------------------

    # Run the simulation for some number of steps
    pbar = tqdm(total=num_steps, desc="Running simulation")
    for i in range(num_steps):
        # Take an action (this is based on previous observations)
        action = agent.act()
        state, done_flag, done_message = environment.step(action)
        pbar.update(1)

        # If we're done exit the loop
        if done_flag:
            pbar.set_description(done_message)
            break

        # Make new observations
        agent.observe(state)

        # Update the pbar with the current state and action
        p_string = ", ".join([f"{x:<5.1f}" for x in state[0:3]])
        v_string = f"{np.linalg.norm(state[6:9]):<4.1f}"
        w_string = ", ".join([f"{x:<4.1f}" for x in state[9:12]])
        a_string = ", ".join([f"{x:<4.1f}" for x in action])
        pbar.set_description(
            f"t={(i+1)*dyn.dt:.2f}/{num_seconds:.2f} | p=[{p_string}] | v={v_string} | w=[{w_string}] | a=[{a_string}] | gpu={'yes' if use_gpu_if_available else 'no'}")
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
    utils.logging.pickle_to_filepath(
        f"{log_folder}/policy/path_xyz_smooth.pkl",
        path_xyz_smooth,
    )

    # Render visuals
    visual = Visual(log_folder)
    visual.plot_histories()
    visual.render_video(desired_fps=4)

    # Clean up stored data 
    try:
        if not keep_policy_logs:
            policy.delete_logs()
    except:
        pass