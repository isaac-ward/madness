import matplotlib as mpl 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from itertools import product, combinations
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
        # Reverse because its costs not rewards
        self.colormap_red_green = colormap[::-1]

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
        plot_helper(axs[:len(state_plot_groups)], state_history, state_labels, state_plot_groups, color="royalblue")
        plot_helper(axs[len(state_plot_groups):], action_history, action_labels, action_plot_groups, color="red")

        # Save the figure
        plt.tight_layout()
        fig.savefig(os.path.join(self.visuals_folder, "history.png"))

        # Close the figure
        plt.close(fig)

    def plot_environment(self):

        # For plot environment to work correctly we need the following
        # - signed_distance_function.pkl
        # - map.pkl

        # Create a figure
        fig = plt.figure(figsize=(12, 12))
        # 1 row, two columns
        gs = gridspec.GridSpec(2, 2) #, height_ratios=[1, 1, 0.15, 0.15, 0.15, 0.15])
        axs = {
            # main world view render
            "main": fig.add_subplot(gs[0, 0], projection='3d'),

            # orthographic views (from which axis)
            "z": fig.add_subplot(gs[0, 1], projection='3d'),
            "x": fig.add_subplot(gs[1, 0], projection='3d'),
            "y": fig.add_subplot(gs[1, 1], projection='3d'),
        }

        # Load the map and the signed distance function data
        map_ = utils.logging.unpickle_from_filepath(os.path.join(self.run_folder, "environment", "map.pkl"))
        sdfs = utils.logging.unpickle_from_filepath(os.path.join(self.run_folder, "signed_distance_function.pkl"))

        # shape is (3,2) and is the lower and upper bounds for each axis
        extents = map_.extents_metres_xyz

        # Now get the voxel grid info for rendering
        print("Precomputing voxel information...", end="")

        # Precompute the voxel representation stuff 
        voxel_occupied_centers = np.argwhere(map_.voxel_grid == 1)
        voxel_occupied_centers = [ map_.voxel_coords_to_metres(v) for v in voxel_occupied_centers ]

        print("done")

        def setup_axes_3d(axs, ax_name):
            # xyz are the locations of the drone this frame
            
            # These axes in the same aspect ratio
            for ax_name in ["main", "x", "y", "z"]:
                axs[ax_name].set_box_aspect([1,1,1])

            # These axes need the same extents
            for ax_name in ["main", "x", "y", "z"]:
                axs[ax_name].set_xlim(*extents[0])
                axs[ax_name].set_ylim(*extents[1])
                axs[ax_name].set_zlim(*extents[2])

            # These axes should have removed ticks
            for ax_name in ["x", "y", "z"]:
                # This has the effect of removing the grid too
                axs[ax_name].set_xticks([])
                axs[ax_name].set_yticks([])
                axs[ax_name].set_zticks([])

            # Remove the axis themselves
            for ax_name in ["x", "y", "z"]:
                axs[ax_name].axis('off')
                
            # These axes should be orthographic
            for ax_name in ["x", "y", "z"]:
                axs[ax_name].set_proj_type('ortho')

            def write_label_to_top_left_of_axis(ax, text):
                ax.text2D(0.5, 0.95, text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='center')

            # Manually set the rotation of the orthographic views
            # Useful link: https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.view_init.html
            # x is the behind view (along -x axis)
            write_label_to_top_left_of_axis(axs["x"], "behind (↓z →y)")
            axs["x"].view_init(azim=0, elev=180, roll=0)
            # y is the right view (along -y axis)
            write_label_to_top_left_of_axis(axs["y"], "right (↓z →x)")
            axs["y"].view_init(azim=90, elev=0, roll=180)
            # z is the top view (along -z axis)
            write_label_to_top_left_of_axis(axs["z"], "top (↓y →x)")
            axs["z"].view_init(azim=90, elev=-90, roll=180)

            # Set the rotation of the project views to be the same and
            # to suit a NED frame orientation as in the KTH paper figure 2
            # Default for Axes3D is 
            # azim, elev, roll = -60, 30, 0
            azim, elev, roll = 60, -30, 180
            axs["main"].view_init(azim=azim, elev=elev, roll=roll)

        for axes_name in ["main", "x", "y", "z"]:
            setup_axes_3d(axs, axes_name)

        # In 3D, plot the bounding box
        # In 3D, plot the bounding box
        for axes_name in ["main", "x", "y", "z"]:
            def draw_bounding_box(ax, bounding_box):
                x1, x2 = bounding_box[0]
                y1, y2 = bounding_box[1]
                z1, z2 = bounding_box[2]
                # Draw the cuboid
                col = 'r'
                ls = ':'
                alpha = 0.5
                ax.plot([x1, x2], [y1, y1], [z1, z1], color=col, linestyle=ls, alpha=alpha) # | (up)
                ax.plot([x2, x2], [y1, y2], [z1, z1], color=col, linestyle=ls, alpha=alpha) # -->
                ax.plot([x2, x1], [y2, y2], [z1, z1], color=col, linestyle=ls, alpha=alpha) # | (down)
                ax.plot([x1, x1], [y2, y1], [z1, z1], color=col, linestyle=ls, alpha=alpha) # <--

                ax.plot([x1, x2], [y1, y1], [z2, z2], color=col, linestyle=ls, alpha=alpha) # | (up)
                ax.plot([x2, x2], [y1, y2], [z2, z2], color=col, linestyle=ls, alpha=alpha) # -->
                ax.plot([x2, x1], [y2, y2], [z2, z2], color=col, linestyle=ls, alpha=alpha) # | (down)
                ax.plot([x1, x1], [y2, y1], [z2, z2], color=col, linestyle=ls, alpha=alpha) # <--
                
                ax.plot([x1, x1], [y1, y1], [z1, z2], color=col, linestyle=ls, alpha=alpha) # | (up)
                ax.plot([x2, x2], [y2, y2], [z1, z2], color=col, linestyle=ls, alpha=alpha) # -->
                ax.plot([x1, x1], [y2, y2], [z1, z2], color=col, linestyle=ls, alpha=alpha) # | (down)
                ax.plot([x2, x2], [y1, y1], [z1, z2], color=col, linestyle=ls, alpha=alpha) # <--

            # Shape is (3,2) and is the lower and upper bounds for each axis
            ax = axs[axes_name]
            draw_bounding_box(ax, map_.extents_metres_xyz)
            
            # In 3D, plot the voxel map
            # Plot an X at each filled voxel center
            ax.scatter(
                [v[0] for v in voxel_occupied_centers],
                [v[1] for v in voxel_occupied_centers],
                [v[2] for v in voxel_occupied_centers],
                color='red',
                marker='x',
                alpha=0.1,
            )

            # Now we go through and plot each sphere (orthogonals as a circle,
            # main as a sphere)
            for sdf in sdfs.sdf_list:
                #print(f"{sdf}")

                # Get the center and radius
                center = sdf.center_metres_xyz
                radius = sdf.radius_metres
                
                # We'll make the spheres
                color = 'purple'
                
                # Behavior changes depending on axis
                #if axes_name == "main":

                # Plot using 3d, which we'll do by
                ax.scatter(
                    [v[0] for v in sdf.interior_metre_coords],
                    [v[1] for v in sdf.interior_metre_coords],
                    [v[2] for v in sdf.interior_metre_coords],
                    color=color,
                    marker='x',
                    alpha=0.5,
                )

                # else:
                #     if axes_name == "x":
                #         c1, c2 = 1, 2
                #     elif axes_name == "y":
                #         c1, c2 = 0, 2
                #     elif axes_name == "z":
                #         c1, c2 = 0, 1
                #     # Plot using circle patches
                #     circle = patches.Circle(
                #         (center[c1], center[c2]),
                #         radius,
                #         color=color,
                #         # no fill
                #         fill=False,
                #         linewidth=1,
                #     )
                #     ax.add_patch(circle)

        # Save the figure
        #plt.tight_layout()
        fig.savefig(os.path.join(self.visuals_folder, "environment.png"))               

    def load_mppi_steps_states_actions_costs(self):
        mppi_folder = os.path.join(self.run_folder, "policy", "mppi")
        step_folders = [f for f in os.listdir(mppi_folder) if os.path.isdir(os.path.join(mppi_folder, f))]
        # Sort alphabetically
        step_folders = sorted(step_folders, reverse=False)
        # Load the states, actions, and costs for each step
        S = []
        A = []
        S_opt = []
        A_opt = []
        J = []
        for step_folder in step_folders:
            this_folder = os.path.join(mppi_folder, step_folder)
            # Try loading the state and action trajectories
            state_trajectories, action_trajectories = utils.logging.load_state_and_action_trajectories(this_folder)
            # And the costs
            costs = utils.logging.unpickle_from_filepath(os.path.join(this_folder, "costs.pkl"))
            # And the optimal action plan
            optimal_state_plan, optimal_action_plan = utils.logging.load_state_and_action_trajectories(this_folder, suffix="optimal")
            S.append(state_trajectories)
            A.append(action_trajectories)
            S_opt.append(optimal_state_plan)
            A_opt.append(optimal_action_plan)
            J.append(costs)
        S, A, S_opt, A_opt, J = np.array(S), np.array(A), np.array(S_opt), np.array(A_opt), np.array(J)
        return S, A, S_opt, A_opt, J

    def render_video(
        self,
        desired_fps=25,
    ):
        """
        We'll use matplotlib's funcanimation to render a video of the simulation
        """

        print("Loading required visual data...", end="")
        
        # The state is set up as [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        # The action is set up as [w1, w2, w3, w4] corresponding to the forward, left, backward, right rotor inputs

        # Load the environment data (they were saved with savez)
        state_history, action_history = utils.logging.load_state_and_action_trajectories(os.path.join(self.run_folder, "environment"))
        desired_state_shape = (state_history.shape[0], 12)
        assert state_history.shape == desired_state_shape, f"State history shape {state_history.shape} != {desired_state_shape}"
        desired_action_shape = (action_history.shape[0], 4)
        assert action_history.shape == desired_action_shape, f"Action history shape {action_history.shape} != {desired_action_shape}"
        dynamics = utils.logging.unpickle_from_filepath(os.path.join(self.run_folder, "environment", "dynamics.pkl"))
        map_ = utils.logging.unpickle_from_filepath(os.path.join(self.run_folder, "environment", "map.pkl"))

        print("done")
        print("Precomputing voxel information...", end="")

        # Precompute the voxel representation stuff 
        voxel_occupied_centers = np.argwhere(map_.voxel_grid == 1)
        voxel_occupied_centers = [ map_.voxel_coords_to_metres(v) for v in voxel_occupied_centers ]

        print("done")
        print("Loading optional visual data...", end="")

        # We also want the policy path, if it exists
        path_flag = False
        try:
            path_xyz = utils.logging.unpickle_from_filepath(os.path.join(self.run_folder, "environment", "path_xyz.pkl"))
            path_flag = True
        except:
            pass

        path_smooth_flag = False
        try:
            path_xyz_smooth = utils.logging.unpickle_from_filepath(os.path.join(self.run_folder, "environment", "path_xyz_smooth.pkl"))
            path_smooth_flag = True
        except:
            pass

        # And if it's an MPPI policy, we want to load the states, actions, and costs
        mppi_flag = False
        try:
            mppi_states, mppi_actions, mppi_opt_states, mppi_opt_actions, mppi_costs = self.load_mppi_steps_states_actions_costs()
            # For each step, over all samples, get the mins and maxes
            mppi_cost_mins = [np.min(c) for c in mppi_costs]
            mppi_cost_maxs = [np.max(c) for c in mppi_costs]
            mppi_flag = True
        except:
            pass

        print("done")

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
        def get_min_max_val_in_history_or_map_extents(history, idx):
            # First look at the history
            min_val = min([state[idx] for state in history])
            max_val = max([state[idx] for state in history])
            # Then look at the map (shape is (3,2)), adding a small buffer
            buffer = 0.5 * quadcopter_diameter
            min_val = min(min_val, map_.extents_metres_xyz[idx][0] - buffer)
            max_val = max(max_val, map_.extents_metres_xyz[idx][1] + buffer)
            # If they are less than the diameter, we'll set them to the diameter
            # NOTE, with reasonable map extents, this should never happen
            if abs(max_val - min_val) < quadcopter_diameter:
                min_val -= buffer
                max_val += buffer
            return min_val, max_val
        extents = [ get_min_max_val_in_history_or_map_extents(state_history, i) for i in range(3) ]
        #print(extents)
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
                # Smaller scale factor here means more zoomed in
                sf = 3
                axs[ax_name].set_xlim(x - sf * quadcopter_diameter, x + sf * quadcopter_diameter)
                axs[ax_name].set_ylim(y - sf * quadcopter_diameter, y + sf * quadcopter_diameter)
                axs[ax_name].set_zlim(z - sf * quadcopter_diameter, z + sf * quadcopter_diameter)

            # These axes should have removed ticks
            for ax_name in ["x", "y", "z", "closeup"]:
                # This has the effect of removing the grid too
                axs[ax_name].set_xticks([])
                axs[ax_name].set_yticks([])
                axs[ax_name].set_zticks([])

            # Remove the axis themselves
            for ax_name in ["x", "y", "z"]:
                axs[ax_name].axis('off')
                
            # These axes should be orthographic
            for ax_name in ["x", "y", "z"]:
                axs[ax_name].set_proj_type('ortho')

            def write_label_to_top_left_of_axis(ax, text):
                ax.text2D(0.5, 0.95, text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='center')

            # Manually set the rotation of the orthographic views
            # Useful link: https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.view_init.html
            # x is the behind view (along -x axis)
            write_label_to_top_left_of_axis(axs["x"], "behind (↓z →y)")
            axs["x"].view_init(azim=0, elev=180, roll=0)
            # y is the right view (along -y axis)
            write_label_to_top_left_of_axis(axs["y"], "right (↓z →x)")
            axs["y"].view_init(azim=90, elev=0, roll=180)
            # z is the top view (along -z axis)
            write_label_to_top_left_of_axis(axs["z"], "top (↓y →x)")
            axs["z"].view_init(azim=90, elev=-90, roll=180)

            # label the closeup
            write_label_to_top_left_of_axis(axs["closeup"], "closeup")

            # Set the rotation of the project views to be the same and
            # to suit a NED frame orientation as in the KTH paper figure 2
            # Default for Axes3D is 
            # azim, elev, roll = -60, 30, 0
            azim, elev, roll = 60, -30, 180
            axs["main"].view_init(azim=azim, elev=elev, roll=roll)
            axs["closeup"].view_init(azim=azim, elev=elev, roll=roll)

        # We need to know how many frames we have in the simulation
        num_frames_simulation = len(action_history)
        # We want to render at X fps, so how many frames will we have in the video?
        simulation_dt = dynamics.dt
        T = num_frames_simulation * simulation_dt
        playback_speed = 1.0
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
            x, y, z, rz, ry, rx, vx, vy, vz, wx, wy, wz = state

            # For visualization it helps to render the moving average of the action
            # history to now and then the current action will be the smoothed version
            # Recall that action_history is shaped (N, action_size)
            def moving_average(actions, window_size):
                """Calculate the moving average of the actions with zero padding to ensure the output size is the same as the input size."""
                if len(actions) < window_size:
                    # TODO replace with a warning and return the original
                    #raise ValueError(f"Not enough data points for moving average. Minimum required: {window_size}, available: {len(actions)}")
                    return actions
                
                # array, pad_width (before, after)
                padded_actions = np.pad(actions, ((window_size, 0), (0, 0)), mode='edge')
                cumulative_sum = np.cumsum(padded_actions, axis=0)
                moving_avg = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

                assert len(moving_avg) == len(actions), f"Moving average length {len(moving_avg)} != actions length {len(actions)}"
                
                return moving_avg
            action_history_smoothed = moving_average(action_history, window_size=5)
            action_smoothed = action_history_smoothed[frame]
            action = action_history[frame]

            # Time to render, clear the old axes, but keep the extents
            for ax in axs.values():
                ax.cla()
            setup_axes_3d(axs, x, y, z)

            # We can compute this once per frame

            # Compute, in the body frame, the 
            # locations of the rotors (+x, +y, -x, -y)
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
            # Make them a nice size
            desired_max_length = quadcopter_diameter * 0.8
            scale_factor = desired_max_length / (action_ranges[:,1] - action_ranges[:,0])
            action_scaled = action_smoothed * scale_factor
            action_vector_startpoints = rotor_locations
            action_vector_endpoints   = rotor_locations - np.array([
                # The order of the actions must line up with rotor locations
                # defined above => +x forward, +y right, -x rear, -y left
                [0, 0, action_scaled[4-1]],
                [0, 0, action_scaled[3-1]],
                [0, 0, action_scaled[2-1]],
                [0, 0, action_scaled[1-1]],
            ])
            
            # Everything will now be rotated and translated into place
            translation = np.array([x, y, z])
            # Why -ve x? God only knows
            rotation = R.from_euler('zyx', [rz, ry, -rx], degrees=False)

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
                # Draw from the center down a bit to represent the drone's up vector
                drone_up_in_world_frame = rotation.apply([0, 0, quadcopter_diameter / 4])
                ax.plot(
                    [translation[0], translation[0] - drone_up_in_world_frame[0]],
                    [translation[1], translation[1] - drone_up_in_world_frame[1]],
                    [translation[2], translation[2] - drone_up_in_world_frame[2]],
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
                    
                # In 3D, plot the bounding box
                if axes_name in ["main", "x", "y", "z"]:
                    def draw_bounding_box(ax, bounding_box):
                        x1, x2 = bounding_box[0]
                        y1, y2 = bounding_box[1]
                        z1, z2 = bounding_box[2]
                        # Draw the cuboid
                        col = 'r'
                        ls = ':'
                        alpha = 0.5
                        ax.plot([x1, x2], [y1, y1], [z1, z1], color=col, linestyle=ls, alpha=alpha) # | (up)
                        ax.plot([x2, x2], [y1, y2], [z1, z1], color=col, linestyle=ls, alpha=alpha) # -->
                        ax.plot([x2, x1], [y2, y2], [z1, z1], color=col, linestyle=ls, alpha=alpha) # | (down)
                        ax.plot([x1, x1], [y2, y1], [z1, z1], color=col, linestyle=ls, alpha=alpha) # <--

                        ax.plot([x1, x2], [y1, y1], [z2, z2], color=col, linestyle=ls, alpha=alpha) # | (up)
                        ax.plot([x2, x2], [y1, y2], [z2, z2], color=col, linestyle=ls, alpha=alpha) # -->
                        ax.plot([x2, x1], [y2, y2], [z2, z2], color=col, linestyle=ls, alpha=alpha) # | (down)
                        ax.plot([x1, x1], [y2, y1], [z2, z2], color=col, linestyle=ls, alpha=alpha) # <--
                        
                        ax.plot([x1, x1], [y1, y1], [z1, z2], color=col, linestyle=ls, alpha=alpha) # | (up)
                        ax.plot([x2, x2], [y2, y2], [z1, z2], color=col, linestyle=ls, alpha=alpha) # -->
                        ax.plot([x1, x1], [y2, y2], [z1, z2], color=col, linestyle=ls, alpha=alpha) # | (down)
                        ax.plot([x2, x2], [y1, y1], [z1, z2], color=col, linestyle=ls, alpha=alpha) # <--

                    # Shape is (3,2) and is the lower and upper bounds for each axis
                    draw_bounding_box(ax, map_.extents_metres_xyz)

                # In 3D, plot the state history as a dotted line
                if axes_name in ["main", "x", "y", "z"]:
                    states_so_far = state_history[:frame]
                    ax.plot(
                        [state[0] for state in states_so_far],
                        [state[1] for state in states_so_far],
                        [state[2] for state in states_so_far],
                        color='royalblue',
                        linestyle='--',
                    )     

                # In 3D, plot the path and smooth paths in 
                if path_flag and axes_name in ["main", "x", "y", "z", "closeup"]:
                    ax.plot(
                        path_xyz[:, 0],
                        path_xyz[:, 1],
                        path_xyz[:, 2],
                        color='grey',
                        linestyle=':',
                        alpha=0.8,
                    )
                if path_smooth_flag and axes_name in ["main", "x", "y", "z", "closeup"]:
                    ax.plot(
                        path_xyz_smooth[:, 0],
                        path_xyz_smooth[:, 1],
                        path_xyz_smooth[:, 2],
                        color='orange',
                        linestyle='-',
                        alpha=0.8,
                    )

                # In 3D, plot the voxel map
                if axes_name in ["main", "x", "y", "z"]:
                    # Plot an X at each filled voxel center
                    ax.scatter(
                        [v[0] for v in voxel_occupied_centers],
                        [v[1] for v in voxel_occupied_centers],
                        [v[2] for v in voxel_occupied_centers],
                        color='red',
                        marker='x',
                        alpha=0.1,
                    )
                
                # If we have access to MPPI data, then render it to some plots too
                if mppi_flag and axes_name in ["main", "x", "y", "z", "closeup"]:

                    # What was the min and max cost this frame (true), or throughout the whole run (false)
                    make_colors_local = False
                    if make_colors_local:
                        mppi_cost_min = mppi_cost_mins[frame]
                        mppi_cost_max = mppi_cost_maxs[frame]
                    else:
                        mppi_cost_min = min(mppi_cost_mins)
                        mppi_cost_max = max(mppi_cost_maxs)

                    # What was the actual trajectory used? We'll highlight it!
                    a_opt = mppi_opt_actions[frame]
                    s_opt = mppi_opt_states[frame]
                    # Include the current state so that it's not disjoint
                    st_opt = np.vstack([state_history[frame], s_opt])
                    ax.plot(
                        st_opt[:, 0],
                        st_opt[:, 1],
                        st_opt[:, 2],
                        color='royalblue',
                        alpha=0.8,
                        linewidth=1,
                        # Keep this small so that we can see
                        # the mppi samples
                        linestyle='--', 
                        # Set z order so this is always visible
                        zorder=100,
                    )

                    # Plot every state trajectory from this frame
                    # Only plot N equally spaced samples - upping this is a major source of slow down!
                    num_samples = 64
                    mppi_states_sampled = mppi_states[frame][np.linspace(0, len(mppi_states[frame])-1, num_samples, dtype=int)]
                    for i, state_trajectory in enumerate(mppi_states_sampled):
                        # Get the cost for this trajectory, scaled to 0,1
                        # and then map it to a color
                        cost_01 = (mppi_costs[frame][i] - mppi_cost_min) / (mppi_cost_max - mppi_cost_min)

                        # Include the current state so that it's not disjoint
                        st = np.vstack([state_history[frame], state_trajectory])
                        ax.plot(
                            st[:, 0],
                            st[:, 1],
                            st[:, 2],
                            color=self.sample_colormap(cost_01),
                            alpha=0.33,
                            linewidth=1,
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
                        linestyle='-',
                        color='black',
                        lw=1,
                        zorder=1,
                        alpha=0.5,
                    )
                    # And the smoothed
                    axs[f"action{i}"].plot(
                        [action[i] for action in action_history_smoothed[:frame]],
                        linestyle='-',
                        color='red',
                        lw=1,
                        zorder=100,
                    )
                    # Set the x axis to the number of frames
                    axs[f"action{i}"].set_xlim(0, num_frames_simulation - 1)
                    # Set the y axis to the action range
                    axs[f"action{i}"].set_ylim(*dynamics.action_ranges()[i])     
                    # Remove axis ticks
                    axs[f"action{i}"].set_xticks([])
                    axs[f"action{i}"].set_yticks([])
            plot_action_inputs()               
                
            # Render the mppi breakdown plot
            def plot_mppi_distribution():
                # Plot the distribution of costs for this frame
                # (shape of mppi_states is (FRAMES, K, H, state_size))
                _, K, H, _ = mppi_states.shape
                costs_this_frame = mppi_costs[frame]
                num_bins = 100
                # bins have to be constant!
                mppi_cost_min = mppi_cost_mins[frame]
                mppi_cost_max = mppi_cost_maxs[frame]
                bins = np.linspace(mppi_cost_min, mppi_cost_max, num=num_bins+1)
                N, _, patches = axs["mppi"].hist(costs_this_frame, bins=bins, edgecolor='white', linewidth=0)
                colors_per_bin = [self.sample_colormap(x) for x in np.linspace(0, 1, num_bins)]
                for i in range(num_bins):
                    patches[i].set_facecolor(colors_per_bin[i])
                # Plot the average cost as a vertical line, in the right color
                avg_cost = np.mean(costs_this_frame)
                axs["mppi"].axvline(
                    avg_cost, 
                    color='k',
                    linewidth=2
                )
                # Set up the axis limits                
                axs["mppi"].set_xlim(mppi_cost_min, mppi_cost_max)
                axs["mppi"].set_ylim(0, int(0.125 * K)) # Unlikely to be more X% of the samples in one bin
                # Remove axis ticks
                axs["mppi"].set_xticks([])
                axs["mppi"].set_yticks([])
                # Plot as text the number of samples and horizon
                axs["mppi"].text(0.01, 0.98, f"K={K:.0f}", transform=axs["mppi"].transAxes, verticalalignment='top', horizontalalignment='left')          
                axs["mppi"].text(0.01, 0.92, f"H={H:.0f}", transform=axs["mppi"].transAxes, verticalalignment='top', horizontalalignment='left')      
            if mppi_flag:
                plot_mppi_distribution()
            else:
                #axs["mppi"].axis('off')
                pass

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
        
        # TODO implement blitting and artists list
        ani = FuncAnimation(fig, update, frames=num_frames_to_render)#, blit=True)
        
        # Save the video
        filepath_output = os.path.join(self.visuals_folder, "render.mp4")
        ani.save(filepath_output, fps=desired_fps)#, extra_args=['-vcodec', 'libx264'])
        # Set the pbar to 100%
        pbar.update(num_frames_to_render)
        pbar.close()

        print(f"Video available at {filepath_output}")
        
        # Free resources
        plt.close(fig)

        # Return the filepath
        return filepath_output
