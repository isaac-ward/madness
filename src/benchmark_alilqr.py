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
from scipy.signal import savgol_filter
import cvxpy
from scipy.spatial.transform import Rotation as R

import utils.general
import utils.logging
import utils.geometric
from environment import Environment
from mapping import Map
from agent import Agent
from visual import Visual
from policies.simple import PolicyNothing, PolicyRandom, PolicyConstant
from policies.mppi import PolicyMPPI
from policies.ilqr import PolicyALiLQR
import policies.samplers
import standard
from sdf import Environment_SDF
from policies.cvxguidance import SCPSolver, Trajectory, SCvxSolver

if __name__ == "__main__":
    run_complete = 0
    try:
        while not run_complete:
            # User inputs:
            min_pt_dist = 26 # Minimum distance (m) between start and goal points
            last_entry_duplicate = 100 # Number of times to duplicate last state in path
            num_seconds = 120 # Maximum number of seconds for simulation run
            safety_factor = 2 # Factor of safety x diameter of quadrotor
            max_alilqr_iters = 300 # Maximum number of iterations for AL-iLQR

            # Seed everything
            utils.general.random_seed(42)

            # Will log everything here
            log_folder = utils.logging.make_log_folder(name="run")
            v = Visual(log_folder)

            # The environment follows some true dynamics, and the agent
            # has an internal model of the environment
            dyn = standard.get_standard_dynamics()
            # dyn = standard.get_standard_dynamics_linear()
            n = dyn.state_size()
            m = dyn.action_size()

            # Create a map representation
            # map_ = standard.get_standard_map()
            # map_ = standard.get_28x28x28_at_111()
            # map_ = standard.get_28x28x28_at_111_with_obstacles()
            map_ = standard.get_chamber()
            # map_ = standard.get_tunnels()

            while True:
                try:
                    # Start and goal states
                    state_initial, state_goal = Environment.get_two_states_separated_by_distance(
                        map_, 
                        template=dyn.state_randomization_template(),
                        min_distance=min_pt_dist,
                    )

                    # Generate a path from the initial state to the goal state
                    xyz_initial = state_initial[0:3]
                    xyz_goal = state_goal[0:3]
                    path_xyz = np.array([xyz_initial, xyz_goal])
                    path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*safety_factor)
                    break
                except:
                    print("Invalid start and end points, retrying...")
            try:
                path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz, desired_points_per_meter=20)
            except Exception as e:
                print("Error: " + e)
                path_xyz_smooth = path_xyz

            # Duplicate last entry
            last_entry_duplicated = np.tile(path_xyz_smooth[-1], (last_entry_duplicate, 1))

            # Append the duplicated rows to the original array
            path_xyz_smooth = np.vstack([path_xyz_smooth, last_entry_duplicated])

            # Get length of A* path
            K = path_xyz_smooth.shape[0] - 1

            # Get SDFs 
            # sdfs = Environment_SDF(dyn)
            # sdfs.characterize_env_with_spheres_perturbations(
            #     start_point_meters=xyz_initial,
            #     end_point_meters=xyz_goal,
            #     path_xyz=path_xyz_smooth,
            #     map_env=map_,
            #     max_spheres=500,
            #     randomness_deg=45
            # )

            # Create AL-iLQR policy
            Q = np.eye(n) * 1       # Cost for state error along trajectory
            Q[:3] = Q[:3] * 5
            R_cost = np.eye(m) * 1  # Cost for control input
            QN = np.eye(n) * 5     # Cost for final state error
            QN[:3] = QN[:3] * 4
            W = np.eye(m) * 0       # Control continuity cost

            # Create trajectory to track (A* path with 0s at other states)
            basic_state_traj = np.zeros((K+1,n))
            basic_state_traj[:,:3] = path_xyz_smooth

            # Create initial control inputs
            basic_hover_action = np.ones((K,m)) * np.sqrt(dyn.mass*dyn.g/(4*dyn.thrust_coef))

            # Segment trajectory so that each path is ~120 time steps
            segs = round(np.shape(basic_state_traj)[0] / 120.)
            if not segs:
                segs = 1

            # Solve AL-iLQR policy
            policy = PolicyALiLQR(
                dynamics=copy.deepcopy(dyn),
                Q=Q,
                R=R_cost,
                QN=QN,
                W=W,
                x_track=basic_state_traj,
                u_track=basic_hover_action,
                segments=segs,
                eps=1e-1,
                max_iters=max_alilqr_iters,
                verbose=True,
                run_folder=log_folder,
            )

            # Create an agent
            agent = Agent(
                state_initial=state_initial,
                policy=policy,
                state_size=dyn.state_size(),
                action_ranges=dyn.action_ranges(),
                zero_pad_state=None
            ) 

            # Create the environment
            num_steps = int(num_seconds / dyn.dt)
            environment = Environment(
                state_initial=state_initial,
                state_goal=state_goal,
                dynamics=dyn,
                map_=map_,
                episode_length=num_steps,
            )
            use_gpu_if_available = False

            # Create AL-iLQR path
            ilqr_traj = [state_initial[:3]]

            # Run the simulation for some number of steps
            pbar = tqdm(total=num_steps, desc="Running simulation")
            continue_after_done_secs = 0
            continue_after_done_steps = int(continue_after_done_secs / dyn.dt)    
            for i in range(num_steps):
                # Take an action (this is based on previous observations)
                action = agent.act()
                state, done_flag, done_message = environment.step(action)
                ilqr_traj.append(state[:3])
                pbar.update(1)

                # If we're done exit the loop in X timesteps
                if done_flag:
                    pbar.set_description(done_message)
                    continue_after_done_steps -= 1
                    if continue_after_done_secs == 0:
                        break
                if continue_after_done_steps == 0 and continue_after_done_secs != 0:
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

            # Take last action to log in AL-iLQR policy
            action = agent.act()

            # Log everything of interest
            environment.log(log_folder)

            # Log the A* path
            utils.logging.save_to_npz(
                os.path.join(log_folder, "a_star", "start_to_goal.npz"),
                path_xyz,
            )

            # Log the smooth A* path
            utils.logging.save_to_npz(
                os.path.join(log_folder, "a_star", "start_to_goal_smooth.npz"),
                path_xyz_smooth,
            )

            # Log the SDF spheres
            # utils.logging.pickle_to_filepath(
            #     os.path.join(log_folder, "signed_distance_function.pkl"),
            #     sdfs,
            # )

            # Log the iLQR path
            utils.logging.save_to_npz(
                os.path.join(log_folder, "al_ilqr", "al_ilqr.npz"),
                ilqr_traj,
            )

            # Generate AL-iLQR logs
            policy.generate_logs()

            # Render visuals
            visual = Visual(log_folder)
            visual.plot_environment()
            visual.render_video(desired_fps=25)
            run_complete = 1
    except KeyboardInterrupt:
        None