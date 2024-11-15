import numpy as np
from scipy.spatial.transform import Rotation as R
import cvxpy
from tqdm import tqdm
import scipy
import os
import pickle
import csv
import time
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    # state_initial[3] = 1
    state_goal = np.zeros(dyn.state_size())
    state_goal[:3] = np.array([25,25,25])
    #state_goal[:3] = state_initial[:3] + np.array([5,0,0])

    # # Generate a path from the initial state to the goal state
    xyz_initial = state_initial[0:3]
    xyz_goal = state_goal[0:3]
    path_xyz = np.array([xyz_initial, xyz_goal])
    path_xyz = map_.plan_path(xyz_initial, xyz_goal, dyn.diameter*4) # Ultra safe
    #path_xyz = upsample(path_xyz, num_points_between=5)
    # path_xyz_smooth = path_xyz # TODO
    try:
        path_xyz_smooth = utils.geometric.smooth_path_same_endpoints(path_xyz, desired_points_per_meter=10)
    except Exception as e:
        print(e)
        path_xyz_smooth = path_xyz
    
    #print(path_xyz_smooth.shape)
    K = int(path_xyz_smooth.shape[0] - 1)

    print(f"Desire {K+1} states")
    print(f"Desire {K} actions")

    # Create initial trajectory guess for SCP
    trajInit = Trajectory()

    # Extract dynamics constants and coeffs
    k = dyn.thrust_coef
    m = dyn.mass
    g = dyn.g
    w_trim = np.sqrt(m*g/(4*k))

    dyn.dt = 0.025

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
    ang_vel = finite_diff_helper(rot)

    # Angular velocity order is x y z so swap it around
    ang_vel = ang_vel[:, ::-1]

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

    def log_states_and_actions(name, states, actions, log_accelerations=False, accels=None):
        # Log the state and action guesses for the initial trajectory for visualization
        # For the states and for the actions we want time series plots as subplots
        # For the states we want position, euler angles, velocities, and angular velocities
        # Make subplots
        num_plots = dyn.state_size() 
        state_labels = dyn.state_labels() 
        if log_accelerations:
            num_plots += 3
            state_labels += ["ax", "ay", "az"]
        fig = plt.figure(figsize=(10, num_plots*2))
        for i in range(num_plots):
            ax = fig.add_subplot(num_plots, 1, i+1)
            # Disable scientific notation on the y-axis (or x-axis if needed)
            ax.ticklabel_format(useOffset=False)
            if i < dyn.state_size():
                ax.plot(states[:,i])
            else:
                ax.plot(accels[:,i-dyn.state_size()])
            ax.set_title(state_labels[i])
        plt.tight_layout()
        plt.savefig(os.path.join(log_folder, f"{name}_states.png"))

        # For the actions we want the rotor speeds
        # Make subplots
        num_plots = dyn.action_size()
        fig = plt.figure(figsize=(10, num_plots*2))
        action_labels = dyn.action_labels()
        for i in range(num_plots):
            ax = fig.add_subplot(num_plots, 1, i+1)
            # Disable scientific notation on the y-axis (or x-axis if needed)
            ax.ticklabel_format(useOffset=False)
            ax.plot(actions[:,i])
            ax.set_title(action_labels[i])
        plt.tight_layout()
        plt.savefig(os.path.join(log_folder, f"{name}_actions.png"))
    log_states_and_actions("initial_guess", trajInit.state, trajInit.action, log_accelerations=True, accels=acc)

    use_legacy = False
    if use_legacy:
        # Initialize position state guess with smooth Astar results
        trajInit.state = np.zeros((K+1, dyn.state_size()))
        trajInit.state[:,:3] = path_xyz_smooth
        # trajInit.action = np.ones((K,4)) * np.sqrt(dyn.mass*dyn.g/(4*dyn.thrust_coef))
        # trajInit.action = np.ones((K,4)) * w_trim

        # Use finite difference to back out velocities at each step (assume final velocity of zero)
        vel = np.zeros(np.shape(path_xyz_smooth))
        vel[:-1] = (path_xyz_smooth[1:] - path_xyz_smooth[:-1])/dyn.dt
        vel[-1] = vel[-2]

        # Smooth the velocity components using Savitzky-Golay filter
        padded_vel = np.pad(vel, ((10,10), (0,0)), mode="symmetric")
        smoothed_vel_x = savgol_filter(padded_vel[:, 0], window_length=2, polyorder=1)[10:-10]
        smoothed_vel_y = savgol_filter(padded_vel[:, 1], window_length=2, polyorder=1)[10:-10]
        smoothed_vel_z = savgol_filter(padded_vel[:, 2], window_length=2, polyorder=1)[10:-10]

        # Define the time vector
        time = np.linspace(0, K * dyn.dt, K+1)

        smoothed_vel = np.stack([smoothed_vel_x, smoothed_vel_y, smoothed_vel_z], axis=-1)
        trajInit.state[:,6:9] = vel
        #print("shape of smoothed vel: ", smoothed_vel.shape)

        # Use finite difference to back out accelerations -> actions (acceleration at first step is assumed to be from zero velocity to starting velocity)
        accel = np.zeros((K+1,3))
        accel[1:] = (smoothed_vel[1:] - smoothed_vel[:-1])/dyn.dt
        #print("Before g: ", accel)
        accel -= np.array([[0,0,g]])
        #print("After g: ", accel)
        # print(accel)

        # Specify the window size for smoothing
        window_size = 5  # Adjust as needed

        # Smooth the acceleration components using weighted moving average
        padded_accel = np.pad(accel, ((10,10), (0,0)), mode="symmetric")
        #print("accel padded: ", padded_accel)
        smoothed_accel_x = savgol_filter(padded_accel[:, 0], window_length=2, polyorder=1)[10:-10]
        smoothed_accel_y = savgol_filter(padded_accel[:, 1], window_length=2, polyorder=1)[10:-10]
        smoothed_accel_z = savgol_filter(padded_accel[:, 2], window_length=2, polyorder=1)[10:-10]
        smoothed_accel = np.stack([smoothed_accel_x, smoothed_accel_y, smoothed_accel_z], axis=-1)
        #print("accel smoothed: ", smoothed_accel)

        fig, ax = plt.subplots(3, 1, figsize=(10, 8))

        # Plot Euler angles (z, y, x)
        ax[0].plot(time, path_xyz_smooth[:,0], label='p_x', color='b')
        ax[0].plot(time, path_xyz_smooth[:,1], label='p_y', color='g')
        ax[0].plot(time, path_xyz_smooth[:,2], label='p_z', color='r')
        ax[0].set_title("Velocities")
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel("vel [m/s]")
        ax[0].legend()
        ax[0].grid(True)

        # Plot Euler angles (z, y, x)
        ax[1].plot(time, smoothed_vel_x[:], label='v_x', color='b')
        ax[1].plot(time, smoothed_vel_y[:], label='v_y', color='g')
        ax[1].plot(time, smoothed_vel_z[:], label='v_z', color='r')
        ax[1].set_title("Velocities")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("vel [m/s]")
        ax[1].legend()
        ax[1].grid(True)

        # Plot Euler angles (z, y, x)
        ax[2].plot(time, smoothed_accel_x[:], label='a_x', color='b')
        ax[2].plot(time, smoothed_accel_y[:], label='a_y', color='g')
        ax[2].plot(time, smoothed_accel_z[:], label='a_z', color='r')
        ax[2].set_title("Accelerations")
        ax[2].set_xlabel("Time [s]")
        ax[2].set_ylabel("accel [m/s^2]")
        ax[2].legend()
        ax[2].grid(True)

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
        q = np.concatenate([q_0, q_v],axis=-1)

        # normalize quaternion
        q /= np.linalg.norm(q,axis=-1)[:, np.newaxis]

        # trajInit.state[:,3:7] = q
        # trajInit.state[:,3] = 1

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot Euler angles (z, y, x)
        ax.plot(time, q[:, 0], label='q0', color='b')
        ax.plot(time, q[:, 1], label='q1', color='g')
        ax.plot(time, q[:, 2], label='q2', color='r')
        ax.plot(time, q[:, 3], label='q3',color='m')
        ax.set_title("Quaternions")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("q")
        ax.legend()
        ax.grid(True)

        # For converting quaternions to Euler angles

        R = utils.geometric.q2R(q.T)
        u = utils.geometric.R2Euler123(R).T
        #print("u: " + str(u))

        padded_u = np.pad(vel, ((10,10), (0,0)), mode="symmetric")
        smoothed_u_x = savgol_filter(padded_u[:, 0], window_length=2, polyorder=1)[10:-10]
        smoothed_u_y = savgol_filter(padded_u[:, 1], window_length=2, polyorder=1)[10:-10]
        smoothed_u_z = savgol_filter(padded_u[:, 2], window_length=2, polyorder=1)[10:-10]
        smoothed_u = np.stack([smoothed_u_x, smoothed_u_y, smoothed_u_z], axis=-1)
        trajInit.state[:,3:6] = np.array([smoothed_u[:,2],smoothed_u[:,1],smoothed_u[:,0]]).T
        #print("Traj: " + str(trajInit.state[:,3:6]))

        

        # Compute the angular velocity
        qf = q[1:] # advanced time-step history
        qb = q[:-1] # prior time-step history

        # initialize om
        om = np.zeros((K+1, 3))

        # # populate using vectorized quaternion conjugate multiplication
        # om[:-1] = 2/dyn.dt * np.stack([
        #     qb[:,0]*qf[:,1] - qb[:,1]*qf[:,0] - qb[:,2]*qf[:,3] + qb[:,3]*qf[:,2],
        #     qb[:,0]*qf[:,2] + qb[:,1]*qf[:,3] - qb[:,2]*qf[:,0] - qb[:,3]*qf[:,1],
        #     qb[:,0]*qf[:,3] - qb[:,1]*qf[:,2] + qb[:,2]*qf[:,1] - qb[:,3]*qf[:,0]
        # ], axis=-1)
        # om[-1] = om[-2]

        # Compute the relative rotation as the quaternion conjugate of the previous state multiplied by the next state (TRIAL)
        delta_q = np.stack([
            qb[:, 0] * qf[:, 0] + qb[:, 1] * qf[:, 1] + qb[:, 2] * qf[:, 2] + qb[:, 3] * qf[:, 3],
            -qb[:, 1] * qf[:, 0] + qb[:, 0] * qf[:, 1] - qb[:, 3] * qf[:, 2] + qb[:, 2] * qf[:, 3],
            -qb[:, 2] * qf[:, 0] + qb[:, 3] * qf[:, 1] + qb[:, 0] * qf[:, 2] - qb[:, 1] * qf[:, 3],
            -qb[:, 3] * qf[:, 0] - qb[:, 2] * qf[:, 1] + qb[:, 1] * qf[:, 2] + qb[:, 0] * qf[:, 3]
        ], axis=-1)

        # Compute the angular velocity in the body frame
        om[:-1] = 2 / dyn.dt * delta_q[:, 1:]  # Only take vector part for angular velocity
        om[-1] = om[-2]

        padded_om = np.pad(om, ((10,10), (0,0)), mode="symmetric")
        smoothed_om_x = savgol_filter(padded_om[:, 0], window_length=2, polyorder=1)[10:-10]
        smoothed_om_y = savgol_filter(padded_om[:, 1], window_length=2, polyorder=1)[10:-10]
        smoothed_om_z = savgol_filter(padded_om[:, 2], window_length=2, polyorder=1)[10:-10]
        smoothed_om = np.stack([smoothed_om_x, smoothed_om_y, smoothed_om_z], axis=-1)

        # trajInit.state[:,10:] = om
        trajInit.state[:,9:] = smoothed_om



        # Extracting the Euler angles and angular velocities
        euler_angles = trajInit.state[:, 3:6]  # Euler angles in order z, y, x
        angular_velocities = trajInit.state[:, 9:]  # Angular velocities in body frame

        # Plot Euler angles
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        # Plot Euler angles (z, y, x)
        ax[0].plot(time, euler_angles[:, 0], label='Euler angle (z)', color='b')
        ax[0].plot(time, euler_angles[:, 1], label='Euler angle (y)', color='g')
        ax[0].plot(time, euler_angles[:, 2], label='Euler angle (x)', color='r')
        ax[0].set_title("Euler Angles (z, y, x)")
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel("Angle [rad]")
        ax[0].legend()
        ax[0].grid(True)

        # Plot angular velocities in the body frame
        ax[1].plot(time, angular_velocities[:, 0], label='Angular velocity ω_x', color='c')
        ax[1].plot(time, angular_velocities[:, 1], label='Angular velocity ω_y', color='m')
        ax[1].plot(time, angular_velocities[:, 2], label='Angular velocity ω_z', color='y')
        ax[1].set_title("Angular Velocities in Body Frame")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Angular Velocity [rad/s]")
        ax[1].legend()
        ax[1].grid(True)

        # Display the plots
        plt.tight_layout()
        plt.savefig(os.path.join(log_folder, "initial_guess.png"))

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
    #print("Sphere Count: " + str(len(sdfs.sdf_list)))

    # initialize SCP solver object
    scp = SCPSolver(K = K,
                    dynamics=copy.deepcopy(dyn),
                    sdf = sdfs,
                    trajInit=trajInit,
                    maxiter = 10,
                    eps_dyn=1e3,
                    eps_sdf=1e-4,
                    eps_quat=10,
                    sig = 30.,
                    rho=2.,
                    pull_from_cache=False)

    # Setup SCP iterations manually until exit condition is implemented
    state_history = state_initial
    optimal_action_history, optimal_state_history, cvx_logs = scp.solve(
        state_goal=state_goal,
        state_history=state_history[np.newaxis,:],
        return_information=True,
        verbose=False,
    )

    # Log it out
    log_states_and_actions("optimal_guess", optimal_state_history, optimal_action_history, log_accelerations=False)
    
    # print( "norm of scp quat: ", np.linalg.norm( optimal_state_history[:,3:7] , axis=-1) )
    # print("Optimal Control: " + str(optimal_action_history))
    # Extract euclidean coordinates of drone path from state history
    position_history = optimal_state_history[:,:3]

    # Given the known initial action state and the optimal action history, we can propagate the state history
    # to get the propagated path
    propagated_traj = np.zeros_like(optimal_state_history)
    propagated_traj[0,:] = state_initial
    for i in range(1,K+1):
        propagated_traj[i,:] = dyn.step(propagated_traj[i-1,:], optimal_action_history[i-1,:])
    propagated_traj_path = propagated_traj[:,:3]

    # # Propagated path real dynamics
    # propagated_traj = np.copy(optimal_state_history)
    # for i in range(1,K+1):
    #     propagated_traj[i,:] = dyn.step(propagated_traj[i-1,:], optimal_action_history[i-1,:])
    # propagated_traj_path = propagated_traj[:,:3]

    # # Propagated path real dynamics
    # for i in range(1,K+1):
    #     trajInit.state[i,:] = dyn.step(trajInit.state[i-1,:], trajInit.action[i-1,:])
    # propagated_trajInit_path = trajInit.state[:,:3]

    # print("trimmed rotor speed: ", w_trim)
    # print("rotor speed history", optimal_action_history)
    # print(propagated_traj_path)
    # print(position_history)

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

    # For the CVX lets log the costs all on a nice tasty little plot
    # These are all timeseries of the same lenght, per iteration
    # log_total_cost, log_terminal_cost, log_action_cost, log_distance_cost
    # Plot each on its own axes arrange vertically with a common x axis
    # Unpack the logs and plot
    log_total_cost, log_terminal_cost, log_action_cost, log_distance_cost, log_slack_bound = cvx_logs
    num_subplots = len(cvx_logs)
    fig, ax = plt.subplots(num_subplots, 1, figsize=(10, num_subplots*2))
    for i, (name, log) in enumerate(
        [
            ("Total Cost", log_total_cost),
            ("Terminal Cost", log_terminal_cost),
            ("Action Cost", log_action_cost),
            ("Distance Cost", log_distance_cost),
            ("Slack Bound", log_slack_bound),
        ]
    ):
        #print(f"Plotting {name}: {log}")
        ax[i].plot(log)
        ax[i].set_title(name)
        ax[i].set_xlabel("Iteration")
        ax[i].set_ylabel("Cost")
        ax[i].grid(True)
    plt.tight_layout()
    # Save it to the log folder
    plt.savefig(os.path.join(log_folder, "costs.png"))

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
    utils.logging.save_to_npz(
        os.path.join(log_folder, "cvx", "propagated.npz"),
        propagated_traj_path #propagated_trajInit_path#path_xyz_smooth,
    )

    # Render visuals
    visual = Visual(log_folder)
    #visual.plot_histories()
    #visual.render_video(desired_fps=25)
    visual.plot_environment()