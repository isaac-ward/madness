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
import dynamics
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
    # Seed everything
    utils.general.random_seed(42)

    # Will log everything here
    log_folder = utils.logging.make_log_folder(name="run")
    v = Visual(log_folder)

    # The environment follows some true dynamics, and the agent
    # has an internal model of the environment
    dyn = standard.get_standard_dynamics_jax_quadcopter_3d()
    n = dyn.state_size()
    m = dyn.action_size()

    # Create a map representation
    # map_ = standard.get_standard_map()
    # map_ = standard.get_28x28x28_at_111()
    map_ = standard.get_28x28x28_at_111_with_obstacles()

    # Start and goal states
    state_initial = np.zeros(dyn.state_size())
    state_initial[:3] = 5
    # state_initial[3] = 1
    state_goal = np.zeros(dyn.state_size())
    # state_goal[:3] = np.array([10,5,2])
    state_goal[:3] = 25
    # state_goal[:3] = np.array([25,25,5])
    # state_goal[3] = 1
    # state_goal[:3] = state_initial[:3] + np.array([5,5,20])

    # # Generate a path from the initial state to the goal state
    xyz_initial = state_initial[0:3]
    xyz_goal = state_goal[0:3]
    path_xyz = np.array([xyz_initial, xyz_goal])
    path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*4) # Ultra safe
    try:
        path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz, desired_points_per_meter=10)
    except Exception as e:
        print(e)
        path_xyz_smooth = path_xyz
    K = path_xyz_smooth.shape[0] - 1

    # SDFs --------------------------------------------------------------------------------------------------------------------
    sdfs = Environment_SDF(dyn)
    sdfs.characterize_env_with_spheres_perturbations(
        start_point_meters=xyz_initial,
        end_point_meters=xyz_goal,
        path_xyz=path_xyz_smooth,
        map_env=map_,
        max_spheres=500,
        randomness_deg=45
    )

    # SCP --------------------------------------------------------------------------------------------------------------------
    # Create initial trajectory guess for SCP
    trajInit = Trajectory()

    # Extract dynamics constants and coeffs
    k = dyn.thrust_coef
    m = dyn.mass
    g = dyn.g
    w_trim = np.sqrt(m*g/(4*k))

    dyn.dt = 0.05

    # We need to formulate an initial guess for the trajectory based on the A* path and
    # finite difference methods, using an euler angle angle representation (123 scheme)
    def finite_diff_helper(vector, clamp=True):
        fd = np.zeros(np.shape(vector))
        dt = dyn.dt

        # Keep the first and last points the same
        if clamp:
            fd[0] = vector[0]
            fd[-1] = vector[-1]

            # Apply finite difference for each middle point
            for i in range(1, len(vector) - 1):
                fd[i] = (vector[i + 1] - vector[i - 1]) / (2 * dt)
        else:
            # Apply finite difference for each middle point

            # Forward difference for the first point
            fd[0] = (vector[1] - vector[0]) / dt

            # Central difference for middle points
            for i in range(1, len(vector) - 1):
                fd[i] = (vector[i + 1] - vector[i - 1]) / (2 * dt)

            # Backward difference for the last point
            fd[-1] = (vector[-1] - vector[-2]) / dt
            
        # Assert that the shape is good
        assert fd.shape == vector.shape, f"Shape is {fd.shape} but should be {vector.shape}"

        return fd 
    
    def clamped_smoothness_helper(vector, smoothness_weight=20.0, closeness_weight=1.0):
        n_points, n_dims = vector.shape
        smoothed_vector = np.zeros_like(vector)

        for i in range(n_dims):
            # Define the optimization variable for this column
            x = cvxpy.Variable(n_points)

            # Objective 1: Smoothness - minimize squared differences between consecutive points
            smoothness_objective = cvxpy.sum_squares(x[1:] - x[:-1])

            # Objective 2: Closeness to the original path - minimize deviation from the original vector
            closeness_objective = cvxpy.sum_squares(x - vector[:, i])

            # Combined objective with weights
            objective = cvxpy.Minimize(smoothness_weight * smoothness_objective + closeness_weight * closeness_objective)

            # Constraints to keep the first and last points fixed (to avoid drifting)
            constraints = [x[0] == vector[0, i], x[-1] == vector[-1, i]]

            # Set up and solve the problem
            prob = cvxpy.Problem(objective, constraints)
            prob.solve()

            # Store the optimized column in the smoothed vector
            smoothed_vector[:, i] = x.value
        return smoothed_vector
    
    # Get the linear kinematics
    pos = clamped_smoothness_helper(path_xyz_smooth, smoothness_weight=200)
    # Compute velocities from finite difference with zero padding
    vel = clamped_smoothness_helper(finite_diff_helper(pos, clamp=False))
    acc = clamped_smoothness_helper(finite_diff_helper(vel, clamp=False))
    acc -= np.array([[0,0,g]])

    # Compute the xyz 123 scheme euler angles
    rot = np.zeros(pos.shape)
    # Iterate over each time step to compute the rotation matrix and Euler angles
    for i in range(len(vel)):
        # # Forward axis (x-axis) - normalize velocity vector
        # forward = vel[i] / np.linalg.norm(vel[i])
        
        # # Up axis (z-axis) - gravity-aligned up vector
        # up = np.array([0, 0, 1])  # Gravity points down along z
        
        # # Right axis (y-axis) - perpendicular to forward and up
        # right = np.cross(up, forward)
        # right /= np.linalg.norm(right)  # Normalize
        
        # # Recompute up to ensure orthogonality
        # up = np.cross(forward, right)
        
        # # Construct the rotation matrix
        # R_matrix = np.column_stack((forward, right, up))
        
        # # Convert rotation matrix to Euler angles (XYZ convention)
        # rotation = R.from_matrix(R_matrix)  # Create a Rotation object
        # euler_angles = rotation.as_euler('zyx', degrees=False)  # Get Euler angles in radians
    
        # # Store the Euler angles
        # rot[i] = euler_angles

        # Up axis is normalized accel
        up = - acc[i] / np.linalg.norm(acc[i]) # negative because +z is down in this world
        # Forward axis is normalized velocity
        forward = vel[i] / np.linalg.norm(vel[i])
        # Right axis is cross product of up and forward
        right = np.cross(up, forward)
        # Then forward is cross product of right and up
        forward = np.cross(right, up)

        # Compute the euler angle rotations that would transform a vector in the global frame into the body frame
        rotation_matrix = R.from_matrix(np.column_stack((forward, right, up)))
        euler_angles = rotation_matrix.as_euler('zyx', degrees=False)
        rot[i] = euler_angles

    # Smoothen
    # rot = clamped_smoothness_helper(rot)
    
    # Compute the angular velocities
    #ang_vel = clamped_smoothness_helper(finite_diff_helper(rot))
    ang_vel = finite_diff_helper(rot, clamp=False)

    # Angular velocity order is x y z so swap it around
    ang_vel = ang_vel[:, ::-1]

    ang_vel = clamped_smoothness_helper(ang_vel)

    # Assemble in the order pos, rot, vel, ang_vel
    # x, y, z, φ, θ, ψ, xd, yd, zd, wx, wy, wz
    trajInit.state = np.concatenate([pos, rot, vel, ang_vel], axis=-1)

    # Assert that the shape is correct
    assert trajInit.state.shape == (K+1, dyn.state_size()), f"Shape is {trajInit.state.shape} but should be {(K+1, dyn.state_size())}"
    
    # Create an initial rough guess of the contorl inputs by 
    # lookginat the mag of the acceleration times the mass and dividing 
    # by 4 times the thrust coefficient (one for each rotor), and then 
    # taking the square root. In other words:
    # ft = k * (w1_sq + w2_sq + w3_sq + w4_sq)
    # m * a = k * (w1_sq + w2_sq + w3_sq + w4_sq)
    # w = sqrt(m * a / (4 * k))
    action_guesses = np.sqrt(m*np.linalg.norm(acc[:-1], axis=-1)/(4*k))
    # Clip it into the action bounds
    action_guesses = np.clip(action_guesses, dyn.action_ranges()[0,0], dyn.action_ranges()[0,1])
    # Set the actions
    trajInit.action = action_guesses[:,np.newaxis]*np.ones((K,4))

    # Assert that the shape is correct
    assert trajInit.action.shape == (K, dyn.action_size()), f"Shape is {trajInit.action.shape} but should be {(K, dyn.action_size())}"

    # Extract dynamics constants and coeffs
    n,m = dyn.state_size(),dyn.action_size()

    # dyn.dt = 0.05
    basic_state_traj = np.zeros((K+1,n))#np.zeros_like(optimal_state_history)
    basic_state_traj[:,:3] = path_xyz_smooth#[0]
    # basic_state_traj[:,3] = 1
    basic_hover_action = np.ones((K,m)) * np.sqrt(dyn.mass*dyn.g/(4*dyn.thrust_coef))
    
    x_start = np.zeros(n)
    x_start[:3] = path_xyz_smooth[0]
    x_goal = np.zeros(n)
    x_goal[:3] = path_xyz_smooth[-1]

    scvx = SCvxSolver(
        dynamics=dyn,
        x_traj_init=trajInit.state,#basic_state_traj,
        u_traj_init=trajInit.action,#basic_hover_action,
        x_start=x_start,
        x_goal=x_goal,
        sdf=sdfs,
        verbose=True,
    )

    def plot_progress_helper(
            x,
            u,
            indx,
    ):
        """
        """
        propagated_traj = np.zeros_like(x)
        propagated_traj[0,:] = np.copy(x[0])
        for j in range(1,K+1):
            propagated_traj[j,:] = dyn.step(propagated_traj[j-1,:], u[j-1,:])
        propagated_traj_path = propagated_traj[:,:3]
        v.plot_environment_from_objects(
            map_=map_,
            sdfs=sdfs,
            path_xyz=path_xyz,
            path_xyz_smooth=path_xyz_smooth,
            path_xyz_cvx=x[:,:3],
            path_propagated=propagated_traj_path,
            save_filename=f"environment_{indx}",
        )

    x_scvx,u_scvx,logs_per_iter = scvx.solve(max_iters=30,plot_progress_helper=plot_progress_helper)

    path_scvx = x_scvx[:,:3]

    propagated_traj = np.zeros_like(x_scvx)
    propagated_traj[0,:] = np.copy(basic_state_traj[0])
    for i in range(1,K+1):
        propagated_traj[i,:] = dyn.step(propagated_traj[i-1,:], u_scvx[i-1,:])
    propagated_traj_path = propagated_traj[:,:3]

    # Create the figure and axis
    plt.figure(figsize=(10, 6))

    # Plot the data
    plt.plot([lpi['u_cost'] for lpi in logs_per_iter], label="Control Cost", color="blue", linewidth=2.5, linestyle="-")
    plt.plot([lpi['x_cost'] for lpi in logs_per_iter], label="State-Goal Cost", color="orange", linewidth=2.5, linestyle="-")
    plt.plot([lpi['nu_cost'] for lpi in logs_per_iter], label="Virtual Control Cost", color="red", linewidth=2.5, linestyle="-")
    plt.plot([lpi['sdf_cost'] for lpi in logs_per_iter], label="SDF Cost", color="black", linewidth=2.5, linestyle="-")

    # Beautify the chart
    plt.title("Beautiful Line Chart", fontsize=18, fontweight="bold", color="darkblue")
    plt.xlabel("X-axis (Iteration)", fontsize=14)
    plt.ylabel("Y-axis (Cost)", fontsize=14)
    plt.yscale('log')
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()

    # Save and display the chart
    plt.savefig(f'{log_folder}/beautiful_line_chart.png', dpi=300)
    plt.show()

    # Create the environment
    num_steps = np.shape(path_xyz_smooth)[0]
    num_seconds = dyn.dt * num_steps
    environment = Environment(
        state_initial=state_initial,
        state_goal=state_goal,
        dynamics=dyn,
        map_=map_,
        episode_length=num_steps,
    )
    use_gpu_if_available = False

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
        propagated_traj_path#path_xyz_smooth,
    )

    # Log the CVX path
    utils.logging.save_to_npz(
        os.path.join(log_folder, "cvx", "path_xyz_cvx.npz"),
        path_scvx,
    )

    # Log the iLQR path
    # TODO

    # Render visuals
    #v.plot_histories()
    # v.plot_environment()
    # v.render_video(desired_fps=25)
