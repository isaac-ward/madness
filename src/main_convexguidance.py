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
    dyn = standard.get_standard_dynamics_jax_quadcopter_3d()

    # Create a map representation
    #map_ = standard.get_standard_map()
    # map_ = standard.get_28x28x28_at_111()

    map_ = standard.get_28x28x28_at_111_with_obstacles()

    # Start and goal states
    # NOTE: The following utility finds two random points - it doesn't check for collisions!
    # If you're using a map with invalid positions then you might need to specify the start and goal states manually
    state_initial = np.zeros(dyn.state_size())
    state_initial[:3] = 5
    state_initial[3] = 1
    state_goal = np.zeros(dyn.state_size())
    state_goal[:3] = 25
    state_goal[3] = 1
    # state_goal[:3] = [5,5,8]

    # # Generate a path from the initial state to the goal state
    xyz_initial = state_initial[0:3]
    xyz_goal = state_goal[0:3]
    path_xyz = np.array([xyz_initial, xyz_goal])
    path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*4) # Ultra safe
    # path_xyz_smooth = path_xyz # TODO
    path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz)
    print(path_xyz_smooth.shape)
    K = path_xyz_smooth.shape[0] - 1


    # Create initial trajectory guess for SCP
    trajInit = Trajectory()

    # Extract dynamics constants and coeffs
    k = dyn.thrust_coef
    m = dyn.mass
    g = dyn.g
    # w_trim = np.sqrt(m*g/(4*k))

    # Initialize position state guess with smooth Astar results
    trajInit.state = np.zeros((K+1, dyn.state_size()))
    trajInit.state[:,:3] = path_xyz_smooth

    # Use finite difference to back out velocities at each step (assume final velocity of zero)
    vel = np.zeros(np.shape(path_xyz_smooth))
    vel[:-1] = (path_xyz_smooth[1:] - path_xyz_smooth[:-1])/dyn.dt
    vel[-1] = vel[-2]

    # Smooth the velocity components using Savitzky-Golay filter
    smoothed_vel_x = savgol_filter(vel[:, 0], window_length=5, polyorder=1)
    smoothed_vel_y = savgol_filter(vel[:, 1], window_length=5, polyorder=1)
    smoothed_vel_z = savgol_filter(vel[:, 2], window_length=5, polyorder=1)

    smoothed_vel = np.stack([smoothed_vel_x, smoothed_vel_y, smoothed_vel_z], axis=-1)
    trajInit.state[:,7:10] = vel
    print("shape of smoothed vel: ", smoothed_vel.shape)

    # Use finite difference to back out accelerations -> actions (acceleration at first step is assumed to be from zero velocity to starting velocity)
    accel = np.zeros((K+1,3))
    accel[1:] = (smoothed_vel[1:] - smoothed_vel[:-1])/dyn.dt
    accel -= np.array([[0,0,g]])
    # print(accel)

    # Specify the window size for smoothing
    window_size = 5  # Adjust as needed

    # Smooth the acceleration components using weighted moving average
    smoothed_accel_x = savgol_filter(accel[:, 0], window_length=5, polyorder=1)
    smoothed_accel_y = savgol_filter(accel[:, 1], window_length=5, polyorder=1)
    smoothed_accel_z = savgol_filter(accel[:, 2], window_length=5, polyorder=1)
    smoothed_accel = np.stack([smoothed_accel_x, smoothed_accel_y, smoothed_accel_z], axis=-1)

    w = np.sqrt( m*np.linalg.norm(smoothed_accel[:-1], axis=-1)/(4*k) )
    w_bounds = dyn.action_ranges()
    w = np.where( w > w_bounds[0,1], w_bounds[0,1], w)
    w = np.where( w < w_bounds[0,0], w_bounds[0,0], w)
    trajInit.action = w[:,np.newaxis]*np.ones((K,4))
    # trajInit.action = w_trim*np.ones((K,4))

    # Use acceleration vector to determine attitude assuming thrust vector corresponds to -z body axis
    
    # thrust direction in global frame
    v1 = -smoothed_accel / np.linalg.norm(smoothed_accel, axis=-1)[:,np.newaxis] 

    # thrust direction in body frame
    v2 = np.zeros((K+1,3))
    v2[:,2] = 1 

    # create quaternion representation of heading by computing axis-angle rotation between the body and global
    q_v = np.cross(v2, v1, axis=-1) / np.sqrt(2 * (1 + np.sum(v1*v2, axis=-1)))[:,np.newaxis]
    q_0 = np.sqrt(2 * (1 + np.sum(v1*v2, axis=-1)))[:,np.newaxis] / 2
    q = np.concat([q_0, q_v],axis=-1)

    # normalize quaternion
    q /= np.linalg.norm(q,axis=-1)[:, np.newaxis]

    trajInit.state[:,3:7] = q
    # trajInit.state[:,3] = 1

    # Compute the angular velocity
    qf = q[1:] # advanced time-step history
    qb = q[:-1] # prior time-step history

    # initialize om
    om = np.zeros((K+1, 3))

    # populate using vectorized quaternion conjugate multiplication
    om[:-1] = 2/dyn.dt * np.stack([
        qb[:,0]*qf[:,1] - qb[:,1]*qf[:,0] - qb[:,2]*qf[:,3] + qb[:,3]*qf[:,2],
        qb[:,0]*qf[:,2] + qb[:,1]*qf[:,3] - qb[:,2]*qf[:,0] - qb[:,3]*qf[:,1],
        qb[:,0]*qf[:,3] - qb[:,1]*qf[:,2] + qb[:,2]*qf[:,1] - qb[:,3]*qf[:,0]
    ], axis=-1)
    om[-1] = om[-2]
    trajInit.state[:,10:] = om


    # for i in range(1,K):
    #     trajInit.state[i,:] = dyn.step(trajInit.state[i-1,:], trajInit.action[i-1,:])

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
    print("Sphere Count: " + str(len(sdfs.sdf_list)))

    # initialize SCP solver object
    scp = SCPSolver(K = K,
                    dynamics=copy.deepcopy(dyn),
                    sdf = sdfs,
                    trajInit=trajInit,
                    maxiter = 15,
                    eps_dyn=1e5,
                    eps_sdf=10.,
                    sig = 30.,
                    rho=2.,
                    pull_from_cache=True)

    # Setup SCP iterations manually until exit condition is implemented
    state_history = state_initial
    optimal_action_history, optimal_state_history = scp.solve(state_goal=state_goal,
                state_history=state_history[np.newaxis,:])
    
    # Extract euclidean coordinates of drone path from state history
    position_history = optimal_state_history[:,:3]

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
    utils.logging.save_to_npz(
        os.path.join(log_folder, "a_star", "start_to_goal.npz"),
        path_xyz,
    )
    utils.logging.save_to_npz(
        os.path.join(log_folder, "a_star", "start_to_goal_smooth.npz"),
        path_xyz_smooth,
    )

    # Log the CVX path
    utils.logging.save_to_npz(
        os.path.join(log_folder, "cvx", "path_xyz_cvx.npz"),
        position_history,
    )

    # Render visuals
    visual = Visual(log_folder)
    #visual.plot_histories()
    #visual.render_video(desired_fps=25)
    visual.plot_environment()