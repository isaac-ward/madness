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
from agent import Agent
from visual import Visual
from policies.simple import PolicyNothing, PolicyRandom, PolicyConstant
from policies.mppi import PolicyMPPI

if __name__ == "__main__":

    # Will log everything to here
    log_folder = utils.logging.make_log_folder(name="run")

    # The environment follows some true dynamics, and the agent
    # has an internal model of the environment
    dynamics = dynamics.DynamicsQuadcopter3D(
        diameter=0.5,
        mass=3.0,
        Ix=0.1,
        Iy=0.1,
        Iz=0.1,
        g=9.81,
        lift_coef=1.0,
        thrust_coef=1.0,
        drag_coef=0.1,
        dt=0.01,
    )

    # Define the initial state of the system
    # Positions, rotations (quaternions), velocities, angular velocities
    state_initial = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    # Create the environment
    # TODO map representations
    map_ = None
    environment = Environment(
        state_initial=state_initial,
        dynamics_model=dynamics,
        map_=map_,
    )

    # Create the agent, which has an initial state and a policy
    # policy = PolicyRandom(
    #     state_size=dynamics.state_size(),
    #     action_size=dynamics.action_size(),
    # )
    policy = PolicyConstant(
        state_size=dynamics.state_size(),
        action_size=dynamics.action_size(),
        constant_action=np.ones((4,))*2.75,
        perturb=True,
    )
    # policy = PolicyMPPI(
    #     state_size=dynamics.state_size(),
    #     action_size=dynamics.action_size(),
    #     dynamics=copy.deepcopy(dynamics),
    # )
    agent = Agent(
        state_initial=state_initial,
        policy=policy,
    ) 

    # ----------------------------------------------------------------

    # Run the simulation for some number of steps
    num_steps = 300
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

    # ----------------------------------------------------------------

    # Log everything of interest
    agent.log(log_folder)
    environment.log(log_folder)

    # Render visuals
    visual = Visual(log_folder)
    visual.plot_histories()
    visual.render_video()