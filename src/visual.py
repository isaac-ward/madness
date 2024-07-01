import matplotlib as mpl 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.spatial.transform import Rotation as R
import math

import os
from tqdm import tqdm
import pickle

import utils.geometric

# Given a run folder, generate visual assets
class Visual:
    """
    Handles all plotting, 3d visuals, renders, etc. 'Gimme a visual on that!'
    """

    def __init__(
        self,
        run_folder,
    ):
        """
        A visualizer visualizes a single run, so all the data that it
        needs should be in this folder
        """
        self.run_folder = run_folder

        # Create a visuals folder and ensure that it exists
        self.visuals_folder = os.path.join(self.run_folder, "visuals")
        if not os.path.exists(self.visuals_folder):
            os.makedirs(self.visuals_folder)

        # Create a red/green color map
        colormap = mpl.cm.get_cmap('hsv')
        # But I actually only want the last half and in reverse
        # https://matplotlib.org/stable/users/explain/colors/colormaps.html
        color_samples = 255
        colormap = colormap(np.linspace(0.0, 0.33, num=color_samples+1))
        self.colormap_red_green = colormap

    def sample_colormap(self, value_01):
        """
        Given a value between 0 and 1, return the color from the colormap
        """
        return self.colormap_red_green[math.floor(value_01 * (len(self.colormap_red_green) - 1))]

    def plot_histories(
        self,
    ):
        """
        Plot the history of states and actions 
        """
        
        # Load the state and action histories, and the dynamics model
        state_history, action_history = utils.logging.load_state_and_action_trajectories(os.path.join(self.run_folder, "environment"))
        dynamics = utils.logging.unpickle_from_filepath(os.path.join(self.run_folder, "environment", "dynamics.pkl"))

        # Get the state size and labels, and the action size and labels
        state_size = dynamics.state_size()
        state_labels = dynamics.state_labels()
        # Note that the groups tell us how many plots are in each row
        state_plot_groups = dynamics.state_plot_groups()
        action_size = dynamics.action_size()
        action_labels = dynamics.action_labels()
        action_plot_groups = dynamics.action_plot_groups()
        
        # We'll have as many rows as the max number of plots in a group
        # e.g. [3,4,3] means the first row has 3 plots, the second has 4, the third has 3
        plot_groups = state_plot_groups + action_plot_groups
        num_rows = len(plot_groups)
        num_cols = max(plot_groups)

        # Create the figure
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 8), dpi=100)

        def plot_helper(axs, data, labels, groups, color):
            """
            Helper function to plot the data
            """
            for i, group in enumerate(groups):
                for j in range(num_cols):
                    if j >= group:
                        # Don't plot anything if there are no more states in this group,
                        # and disable the axis
                        axs[i, j].axis('off')
                        continue
                    idx = sum(groups[:i]) + j
                    axs[i, j].plot(data[:, idx], color=color)
                    axs[i, j].set_title(labels[idx])
                    # Set the x and y axis directly to data
                    axs[i, j].set_xlim(0, len(data))
                    min_val = min(data[:, idx])
                    max_val = max(data[:, idx])
                    if min_val == max_val:
                        min_val -= 1
                        max_val += 1
                    axs[i, j].set_ylim(min_val, max_val)
        
        # Plot the states and actions
        plot_helper(axs[:len(state_plot_groups)], state_history, state_labels, state_plot_groups, color="blue")
        plot_helper(axs[len(state_plot_groups):], action_history, action_labels, action_plot_groups, color="red")

        # Save the figure
        plt.tight_layout()
        fig.savefig(os.path.join(self.visuals_folder, "history.png"))

        # Close the figure
        plt.close(fig)

    def load_mppi_steps_states_actions_rewards(self):
        mppi_folder = os.path.join(self.run_folder, "policy", "mppi")
        step_folders = [f for f in os.listdir(mppi_folder) if os.path.isdir(os.path.join(mppi_folder, f))]
        # Load the states, actions, and rewards for each step
        S = []
        A = []
        S_opt = []
        A_opt = []
        R = []
        for step_folder in step_folders:
            this_folder = os.path.join(mppi_folder, step_folder)
            # Try loading the state and action trajectories
            state_trajectories, action_trajectories = utils.logging.load_state_and_action_trajectories(this_folder)
            # And the rewards
            rewards = utils.logging.unpickle_from_filepath(os.path.join(this_folder, "rewards.pkl"))
            # And the optimal action plan
            optimal_state_plan, optimal_action_plan = utils.logging.load_state_and_action_trajectories(this_folder, suffix="optimal")
            S.append(state_trajectories)
            A.append(action_trajectories)
            S_opt.append(optimal_state_plan)
            A_opt.append(optimal_action_plan)
            R.append(rewards)
        S, A, S_opt, A_opt, R = np.array(S), np.array(A), np.array(S_opt), np.array(A_opt), np.array(R)
        return S, A, S_opt, A_opt, R

    def render_video(
        self,
    ):
        """
        We'll use matplotlib's funcanimation to render a video of the simulation
        """

        # Load the environment data (they were saved with savez)
        state_history, action_history = utils.logging.load_state_and_action_trajectories(os.path.join(self.run_folder, "environment"))
        dynamics = utils.logging.unpickle_from_filepath(os.path.join(self.run_folder, "environment", "dynamics.pkl"))
        # The state is set up as [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        # The action is set up as [w1, w2, w3, w4] corresponding to the forward, left, backward, right rotor inputs

        # We also want the policy path, if it exists
        path_flag = False
        try:
            path_xyz = utils.logging.unpickle_from_filepath(os.path.join(self.run_folder, "policy", "path_xyz.pkl"))
            path_flag = True
        except:
            pass

        # And if it's an MPPI policy, we want to load the states, actions, and rewards
        mppi_flag = False
        try:
            mppi_states, mppi_actions, mppi_opt_states, mppi_opt_actions, mppi_rewards = self.load_mppi_steps_states_actions_rewards()
            # Over all steps and samples, so that we have a color range
            # that stays static throughout the render
            mppi_reward_min = np.min(mppi_rewards)
            mppi_reward_max = np.max(mppi_rewards)
            mppi_flag = True
        except:
            pass

        # The dynamics model has information about the quadcopter that we need
        quadcopter_diameter = dynamics.diameter

        # To achieve this we'll do the following
        # - create a figure and 3d axis
        # - create a function that plots the 3d quadcopter oriented correctly given the state
        #     - this works by creating a 'diameter' long line in the x axis, and a 'diameter' long line in the y axis
        #     - at the tips of the lines, we'll draw vectors representing the action at that timestep
        #     - then we'll rotate everything according to the quaternion
        # - create an update function that given a frame number will update the plot with the new state
        # - use funcanimation to render the video, using the interval parameter to ensure playback
        #   at 60 fps
        # - save out the file
        
        # Create a figure
        fig = plt.figure(figsize=(12, 8))
        # 1 row, two columns
        gs = gridspec.GridSpec(6, 4, height_ratios=[1, 1, 0.15, 0.15, 0.15, 0.15])
        axs = {
            # main world view render
            "main": fig.add_subplot(gs[0:2, 0:2], projection='3d'),

            # closeup view
            "closeup": fig.add_subplot(gs[0, 2], projection='3d'),

            # orthographic views (from which axis)
            "z": fig.add_subplot(gs[0, 3], projection='3d'),
            "x": fig.add_subplot(gs[1, 2], projection='3d'),
            "y": fig.add_subplot(gs[1, 3], projection='3d'),

            # action inputs plots
            "action0": fig.add_subplot(gs[2, 0:2]),
            "action1": fig.add_subplot(gs[3, 0:2]),
            "action2": fig.add_subplot(gs[4, 0:2]),
            "action3": fig.add_subplot(gs[5, 0:2]),
        }

        # add a plot for the mppi data if we have it
        if mppi_flag:
            axs["mppi"] = fig.add_subplot(gs[2:6, 2:4])

        # Set plot limits based on the trajectory min and max
        def get_min_max_val_in_history(history, idx):
            min_val = min([state[idx] for state in history])
            max_val = max([state[idx] for state in history])
            # If they are less than the diameter, we'll set them to the diameter
            if abs(max_val - min_val) < quadcopter_diameter:
                min_val = -0.5 * quadcopter_diameter
                max_val = +0.5 * quadcopter_diameter
            return min_val, max_val
        extents = [ get_min_max_val_in_history(state_history, i) for i in range(3) ]
        # Which extent is the widest?
        max_extent = max([abs(extent[1] - extent[0]) for extent in extents])
        # Make them all the same, centered around the middle position encountered for
        # that axis
        for i in range(3):
            center = 0.5 * (extents[i][1] + extents[i][0])
            extents[i] = [center - 0.5 * max_extent, center + 0.5 * max_extent]

        def setup_axes_3d(axs, x, y, z):
            # xyz are the locations of the drone this frame
            
            # These axes in the same aspect ratio
            for ax_name in ["main", "x", "y", "z", "closeup"]:
                axs[ax_name].set_box_aspect([1,1,1])

            # These axes need the same extents
            for ax_name in ["main", "x", "y", "z"]:
                axs[ax_name].set_xlim(*extents[0])
                axs[ax_name].set_ylim(*extents[1])
                axs[ax_name].set_zlim(*extents[2])

            # These axes should be centered around the drone
            for ax_name in ["closeup"]:
                sf = 1
                axs[ax_name].set_xlim(x - sf * quadcopter_diameter, x + sf * quadcopter_diameter)
                axs[ax_name].set_ylim(y - sf * quadcopter_diameter, y + sf * quadcopter_diameter)
                axs[ax_name].set_zlim(z - sf * quadcopter_diameter, z + sf * quadcopter_diameter)

            # These axes should have removed ticks
            for ax_name in ["x", "y", "z", "closeup"]:
                # This has the effect of removing the grid too
                axs[ax_name].set_xticks([])
                axs[ax_name].set_yticks([])
                axs[ax_name].set_zticks([])
                
            # These axes should be orthographic
            for ax_name in ["x", "y", "z"]:
                axs[ax_name].set_proj_type('ortho')

            def write_label_to_top_left_of_axis(ax, text):
                ax.text2D(0.5, 0.95, text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='center')

            # Manually set the rotation of the orthographic views
            # x is the behind view (along -x axis)
            write_label_to_top_left_of_axis(axs["x"], "behind")
            axs["x"].view_init(azim=180, elev=0, roll=0)
            # y is the right view (along -y axis)
            write_label_to_top_left_of_axis(axs["y"], "right")
            axs["y"].view_init(azim=-90, elev=0, roll=0)
            # z is the top view (along -z axis)
            write_label_to_top_left_of_axis(axs["z"], "top")
            axs["z"].view_init(azim=-90, elev=90, roll=0)

            # label the closeup
            write_label_to_top_left_of_axis(axs["closeup"], "closeup")

        # We need to know how many frames we have in the simulation
        num_frames_simulation = len(action_history)
        # We want to render at X fps, so how many frames will we have in the video?
        simulation_dt = dynamics.dt
        T = num_frames_simulation * simulation_dt
        playback_speed = 1
        desired_fps = 25
        num_frames_to_render = math.floor((T / playback_speed) * desired_fps)

        # Track progress with a pbar
        pbar = tqdm(total=num_frames_to_render, desc="Rendering video")

        def update(render_frame_index): 

            # The frame in the video is not the frame in the simulation. In reality
            # we have a very small dt, but in the video we want to show the video
            # at about 25 fps
            sim_frame_index = math.floor(num_frames_simulation / num_frames_to_render * render_frame_index)
            frame = sim_frame_index

            state = state_history[frame]
            x, y, z, qx, qy, qz, qw = state[:7]

            # For visualization it helps to render the moving average of the action
            # history to now and then the current action will be the smoothed version
            # Recall that action_history is shaped (N, action_size)
            def moving_average(actions, window_size):
                """Calculate the moving average of the actions with zero padding to ensure the output size is the same as the input size."""
                if len(actions) < window_size:
                    raise ValueError(f"Not enough data points for moving average. Minimum required: {window_size}, available: {len(actions)}")
                
                # array, pad_width (before, after)
                padded_actions = np.pad(actions, ((window_size, 0), (0, 0)), mode='edge')
                cumulative_sum = np.cumsum(padded_actions, axis=0)
                moving_avg = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

                assert len(moving_avg) == len(actions), f"Moving average length {len(moving_avg)} != actions length {len(actions)}"
                
                return moving_avg
            action_history_smoothed = moving_average(action_history, window_size=4)
            action_smoothed = action_history_smoothed[frame]
            action = action_history[frame]

            # Time to render, clear the old axes, but keep the extents
            for ax in axs.values():
                ax.cla()
            setup_axes_3d(axs, x, y, z)

            # We can compute this once per frame

            # Compute, in the body frame, the 
            # locations of the rotors 1 (forward), 2 (left), 3 (backward), 4 (right)
            rotor_locations = np.array([
                [quadcopter_diameter / 2, 0, 0],
                [0, quadcopter_diameter / 2, 0],
                [-quadcopter_diameter / 2, 0, 0],
                [0, -quadcopter_diameter / 2, 0]
            ])

            # Compute, in the body frame, the end points of the thrust vectors
            action_ranges = dynamics.action_ranges()
            # Get our progress through this range with this action input
            action_normalized = np.array([
                (action_smoothed[i] - action_ranges[i][0]) / (action_ranges[i][1] - action_ranges[i][0])
                for i in range(4)
            ])
            # Normalized is between 0 and 1, so we'll translate it to be around zero and then scale it
            # to our pleasure. If the range is:
            # - [-8, 8] then we need to compute 0.5
            # - [-1, 1] then we need to compute 0.5
            # - [-2, 6] then we need to compute 0.25
            # THERE is something wrong with this. It's backwards TODO
            # magnitude_ratio = np.abs(action_ranges[:,1]/ (action_ranges[:,1] - action_ranges[:,0]))
            # action_scaled = (action_normalized - magnitude_ratio) * (quadcopter_diameter * 0.8)
            desired_max_length = quadcopter_diameter * 0.8
            scale_factor = desired_max_length / (action_ranges[:,1] - action_ranges[:,0])
            action_scaled = action_smoothed * scale_factor
            action_vector_startpoints = rotor_locations
            action_vector_endpoints = rotor_locations + np.array([
                [0, 0, action_scaled[0]],
                [0, 0, action_scaled[1]],
                [0, 0, action_scaled[2]],
                [0, 0, action_scaled[3]],
            ])
            
            # Everything will now be rotated and translated into place
            translation = np.array([x, y, z])
            rotation = R.from_quat([qx, qy, qz, qw])

            # Transform from local frame into world frame
            rotor_locations = rotation.apply(rotor_locations) + translation
            action_vector_startpoints = rotation.apply(action_vector_startpoints) + translation
            action_vector_endpoints   = rotation.apply(action_vector_endpoints) + translation

            def render_3d_axes(axes_name):
                ax = axs[axes_name]
                    
                # Draw a black line from rotor 1 to rotor 3 and from rotor 2 to rotor 4
                # to represent the quadcopter body
                ax.plot(
                    [rotor_locations[0, 0], rotor_locations[2, 0]],
                    [rotor_locations[0, 1], rotor_locations[2, 1]],
                    [rotor_locations[0, 2], rotor_locations[2, 2]],
                    'k-',
                    linewidth=2,
                )
                ax.plot(
                    [rotor_locations[1, 0], rotor_locations[3, 0]],
                    [rotor_locations[1, 1], rotor_locations[3, 1]],
                    [rotor_locations[1, 2], rotor_locations[3, 2]],
                    'k-',
                    linewidth=2,
                )
                # Draw from the center to rotor 1 to represent forward
                ax.plot(
                    [translation[0], rotor_locations[0, 0]],
                    [translation[1], rotor_locations[0, 1]],
                    [translation[2], rotor_locations[0, 2]],
                    color='purple',
                    linewidth=2,
                )

                # Draw the action inputs from the rotor locations to the end points
                for i in range(4):
                    ax.plot(
                        [action_vector_startpoints[i, 0], action_vector_endpoints[i, 0]],
                        [action_vector_startpoints[i, 1], action_vector_endpoints[i, 1]],
                        [action_vector_startpoints[i, 2], action_vector_endpoints[i, 2]],
                        'r-',
                    )        

                # In 3D, plot the state history as a dotted line
                if axes_name in ["main", "x", "y", "z"]:
                    states_so_far = state_history[:frame]
                    ax.plot(
                        [state[0] for state in states_so_far],
                        [state[1] for state in states_so_far],
                        [state[2] for state in states_so_far],
                        'b--',
                    )     

                # In 3D, plot the path as a yellow dashed line
                if path_flag and axes_name in ["main", "x", "y", "z"]:
                    ax.plot(
                        path_xyz[:, 0],
                        path_xyz[:, 1],
                        path_xyz[:, 2],
                        'y--',
                    )
                
                # If we have access to MPPI data, then render it to some plots too
                if mppi_flag and axes_name in ["main", "x", "y", "z", "closeup"]:

                    # What was the actual trajectory used? We'll highlight it!
                    a_opt = mppi_opt_actions[frame]
                    s_opt = mppi_opt_states[frame]
                    # Include the current state so that it's not disjoint
                    st_opt = np.vstack([state_history[frame], s_opt])
                    ax.plot(
                        st_opt[:, 0],
                        st_opt[:, 1],
                        st_opt[:, 2],
                        color='orange',
                        alpha=1,
                        linewidth=2,
                        linestyle='-',
                        # Set z order so this is always visible
                        zorder=100,
                    )

                    # Plot every state trajectory from this frame
                    # Only plot N random samples - upping this is a major source of slow down!
                    num_random_samples = 64
                    mppi_states_sampled = mppi_states[frame][np.random.choice(mppi_states.shape[1], num_random_samples, replace=False)]
                    for i, state_trajectory in enumerate(mppi_states_sampled):
                        # Get the reward for this trajectory, scaled to 0,1
                        # and then map it to a color
                        reward_01 = (mppi_rewards[frame][i] - mppi_reward_min) / (mppi_reward_max - mppi_reward_min)

                        # Include the current state so that it's not disjoint
                        st = np.vstack([state_history[frame], state_trajectory])
                        ax.plot(
                            st[:, 0],
                            st[:, 1],
                            st[:, 2],
                            color=self.sample_colormap(reward_01),
                            alpha=0.25,
                            linewidth=0.5,
                            linestyle='-',
                        )    

            # Render all the 3d axes
            for ax_name in ["main", "x", "y", "z", "closeup"]:
                render_3d_axes(ax_name)

            # Render the action inputs plot
            def plot_action_inputs():
                # Plot the action input up to now 
                # (shape of action_history is (N, action_size))
                actions_so_far = action_history[:frame]
                for i in range(4):
                    axs[f"action{i}"].plot(
                        [action[i] for action in actions_so_far],
                        'r-',
                    )
                    # And the smoothed
                    axs[f"action{i}"].plot(
                        [action[i] for action in action_history_smoothed[:frame]],
                        'r--',
                        alpha=0.5,
                    )
                    # Set the x axis to the number of frames
                    axs[f"action{i}"].set_xlim(0, num_frames_simulation)
                    # Set the y axis to the action range
                    axs[f"action{i}"].set_ylim(*dynamics.action_ranges()[i])     
                    # Remove axis ticks
                    axs[f"action{i}"].set_xticks([])
                    axs[f"action{i}"].set_yticks([])
            plot_action_inputs()               
                
            # Render the mppi breakdown plot
            def plot_mppi_distribution():
                # Plot the distribution of rewards for this frame
                # (shape of mppi_states is (FRAMES, K, H, state_size))
                _, K, H, _ = mppi_states.shape
                rewards_this_frame = mppi_rewards[frame]
                num_bins = 50
                # bins have to be constant!
                bins = np.linspace(mppi_reward_min, mppi_reward_max, num=num_bins+1)
                N, _, patches = axs["mppi"].hist(rewards_this_frame, bins=bins, edgecolor='white', linewidth=0)
                colors_per_bin = [self.sample_colormap(x) for x in np.linspace(0, 1, num_bins)]
                for i in range(num_bins):
                    patches[i].set_facecolor(colors_per_bin[i])
                # Plot the average reward as a vertical line, in the right color
                avg_reward = np.mean(rewards_this_frame)
                axs["mppi"].axvline(
                    avg_reward, 
                    color='k',#self.sample_colormap((avg_reward - mppi_reward_min) / (mppi_reward_max - mppi_reward_min)), 
                    linewidth=2
                )
                # Set up the axis limits                
                axs["mppi"].set_xlim(mppi_reward_min, mppi_reward_max)
                axs["mppi"].set_ylim(0, int(0.25 * K)) # Unlikely to be more X% of the samples in one bin
                # Remove axis ticks
                axs["mppi"].set_xticks([])
                axs["mppi"].set_yticks([])
                # Plot as text the number of samples and horizon
                axs["mppi"].text(0.01, 0.98, f"K={K:.0f}", transform=axs["mppi"].transAxes, verticalalignment='top', horizontalalignment='left')          
                axs["mppi"].text(0.01, 0.92, f"H={H:.0f}", transform=axs["mppi"].transAxes, verticalalignment='top', horizontalalignment='left')      
            if mppi_flag:
                plot_mppi_distribution()
            else:
                axs["mppi"].axis('off')

            # To the main view, render the simulation frame number,
            # simulation dt, simulation time (current and total).
            # Like so:
            # f=100/500
            # t=1.0/5.0 (dt=0.01)
            axs["main"].text2D(0, 0.98, f"t={frame*simulation_dt:.2f}/{T:.2f} (dt={simulation_dt:.2f})", transform=axs["main"].transAxes, verticalalignment='top', horizontalalignment='left')
            axs["main"].text2D(0, 0.94, f"f={frame+1}/{num_frames_simulation}", transform=axs["main"].transAxes, verticalalignment='top', horizontalalignment='left')
            
            # Tight layout
            plt.tight_layout()    

            pbar.update(1)

            return axs,
        
        ani = FuncAnimation(fig, update, frames=num_frames_to_render) #, interval=1000/desired_fps, blit=True)
        
        # Save the video
        filepath_output = os.path.join(self.visuals_folder, "render.mp4")
        ani.save(filepath_output, fps=desired_fps, extra_args=['-vcodec', 'libx264'])
        # Set the pbar to 100%
        pbar.update(num_frames_to_render)
        pbar.close()

        print(f"Video available at {filepath_output}")
        
        # Free resources
        plt.close(fig)
