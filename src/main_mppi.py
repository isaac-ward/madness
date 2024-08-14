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
from mapping import Map
from agent import Agent
from visual import Visual
from policies.simple import PolicyNothing, PolicyRandom, PolicyConstant
from policies.mppi import PolicyMPPI
import policies.samplers
import standard

# TODO implement wandb to allow for more efficient grid searching of parameters

if __name__ == "__main__":

    # Seed everything
    utils.general.seed(42)

    # Are we using GPU? 
    # NOTE: suggest false for now because it's slow
    use_gpu_if_available = False
    keep_policy_logs = False

    # Will log everything to here
    log_folder = utils.logging.make_log_folder(name="run")

    # The environment follows some true dynamics, and the agent
    # has an internal model of the environment
    dyn = standard.get_standard_dynamics_quadcopter_3d()

    # Create a map representation
    map_ = standard.get_standard_map()

    # Start and goal states
    state_initial, state_goal = Environment.get_two_states_separated_by_distance(map_, min_distance=26)

    # # Generate a path from the initial state to the goal state
    xyz_initial = state_initial[0:3]
    xyz_goal = state_goal[0:3]
    path_xyz = np.array([xyz_initial, xyz_goal])
    #path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*4) # Ultra safe
    #path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz)

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

    # Create the agent, which has an initial state and a policy
    K = 1024
    H = 50 #int(0.5/dynamics.dt), # X second horizon
    #action_sampler = policies.samplers.RandomActionSampler(K, H, dyn.action_ranges())
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
        dist_to_goal_string = f"{np.linalg.norm(state[0:3] - state_goal[0:3]):<4.1f}"
        pbar.set_description(
            f"t={(i+1)*dyn.dt:.2f}/{num_seconds:.2f} | d={dist_to_goal_string} | p=[{p_string}] | v={v_string} | w=[{w_string}] | a=[{a_string}] | gpu={'yes' if use_gpu_if_available else 'no'}")
    # Close the bar
    pbar.close()

    # ----------------------------------------------------------------

    # Log everything of interest
    agent.log(log_folder)
    environment.log(log_folder)
    
    # Render visuals
    visual = Visual(log_folder)
    visual.plot_histories()
    visual.render_video(desired_fps=25)

    # Clean up stored data 
    try:
        if not keep_policy_logs:
            policy.delete_logs()
    except:
        pass