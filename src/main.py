import numpy as np
from tqdm import tqdm
import scipy
import os
import pickle
import csv
import time
import copy

import utils.general
import utils.logging
import dynamics
from environment import Environment
from map import Map
from agent import Agent
from visual import Visual
from policies.simple import PolicyNothing, PolicyRandom, PolicyConstant
from policies.mppi import PolicyMPPI


if __name__ == "__main__":

    # Seed everything
    utils.general.seed(0)

    # Will log everything to here
    log_folder = utils.logging.make_log_folder(name="run")

    # The environment follows some true dynamics, and the agent
    # has an internal model of the environment
    dynamics = dynamics.DynamicsQuadcopter3D(
        diameter=0.5,
        mass=4.0,
        Ix=0.25,
        Iy=0.25,
        Iz=0.15,
        g=9.81,
        lift_coef=1.0,
        thrust_coef=1.0,
        drag_coef=0.1,
        dt=0.05,
    )

    # Define the initial state of the system
    # Positions, rotations (quaternions), velocities, angular velocities
    state_initial = [0, 0, 10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    # Create the environment
    map_ = Map(
        map_filepath="maps/empty.csv",
        voxel_per_x_metres=0.2,
        extents_metres_xyz=[
            [-10, 30], 
            [-10, 10], 
            [0, 20]
        ],
    )
    environment = Environment(
        state_initial=state_initial,
        dynamics_model=dynamics,
        map_=map_,
    )

    # We need a path from the initial state to the goal state
    xyz_initial = state_initial[0:3]
    xyz_goal = [20, 0, 10]
    path_xyz = map_.plan_path(xyz_initial, xyz_goal, 0.1)

    # Create the agent, which has an initial state and a policy
    # policy = PolicyRandom(
    #     state_size=dynamics.state_size(),
    #     action_size=dynamics.action_size(),
    # )
    # policy = PolicyConstant(
    #     state_size=dynamics.state_size(),
    #     action_size=dynamics.action_size(),
    #     constant_action=np.ones((4,))*2.75,
    #     perturb=True,
    # )
    policy = PolicyMPPI(
        state_size=dynamics.state_size(),
        action_size=dynamics.action_size(),
        dynamics=copy.deepcopy(dynamics),
        K=1024,
        H=int(0.5/dynamics.dt), # X second horizon
        action_ranges=dynamics.action_ranges(),
    )
    policy.enable_logging(log_folder)
    policy.update_path_xyz(path_xyz)
    agent = Agent(
        state_initial=state_initial,
        policy=policy,
    ) 

    # ----------------------------------------------------------------

    # Run the simulation for some number of steps
    num_seconds = 5
    num_steps = int(num_seconds / dynamics.dt)
    pbar = tqdm(total=num_steps, desc="Running simulation")
    for i in range(num_steps):
        action = agent.act()
        state = environment.step(action)
        agent.observe(state)

        # Update the pbar with the current state and action
        s_string = ", ".join([f"{x:.1f}" for x in state])
        a_string = ", ".join([f"{x:.1f}" for x in action])
        pbar.set_description(f"s: {s_string} | a: {a_string}")
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