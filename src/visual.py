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

    def load_state_action_histories_and_dynamics(
        self,
    ):
        """
        Load the state and action histories, and the dynamics model
        """
        # Load the environment data (they were saved with savez)
        state_history = np.load(os.path.join(self.run_folder, "environment", "state_history.npz"))["arr_0"]
        action_history = np.load(os.path.join(self.run_folder, "environment", "action_history.npz"))["arr_0"]

        # Load the dynamics model
        with open(os.path.join(self.run_folder, "environment", "dynamics.pkl"), "rb") as f:
            dynamics = pickle.load(f)

        return state_history, action_history, dynamics
        
    def plot_histories(
        self,
    ):
        """
        Plot the history of states and actions 
        """
        
        # Load the state and action histories, and the dynamics model
        state_history, action_history, dynamics = self.load_state_action_histories_and_dynamics()

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
                    axs[i, j].set_ylim(min(data[:, idx]), max(data[:, idx]))
        
        # Plot the states and actions
        plot_helper(axs[:len(state_plot_groups)], state_history, state_labels, state_plot_groups, color="blue")
        plot_helper(axs[len(state_plot_groups):], action_history, action_labels, action_plot_groups, color="red")

        # Save the figure
        plt.tight_layout()
        fig.savefig(os.path.join(self.visuals_folder, "history.png"))

        # Close the figure
        plt.close(fig)
        
    def render_video(
        self,
    ):
        """
        We'll use matplotlib's funcanimation to render a video of the simulation
        """

        # Load the environment data (they were saved with savez)
        state_history, action_history, dynamics = self.load_state_action_histories_and_dynamics()
        # The state is set up as [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        # The action is set up as [w1, w2, w3, w4] corresponding to the forward, left, backward, right rotor inputs

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
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        axs = [
            fig.add_subplot(gs[0], projection='3d'),
            fig.add_subplot(gs[1], projection='3d'),
        ]

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

        def setup_axes(axes, x, y, z):
            ax0, ax1 = axes

            # The first axes is the world
            ax0.set_xlim(*extents[0])
            ax0.set_ylim(*extents[1])
            ax0.set_zlim(*extents[2])

            # The second axes is the local view (zoomed in)
            sf = 1
            ax1.set_xlim(x - sf * quadcopter_diameter, x + sf * quadcopter_diameter)
            ax1.set_ylim(y - sf * quadcopter_diameter, y + sf * quadcopter_diameter)
            ax1.set_zlim(z - sf * quadcopter_diameter, z + sf * quadcopter_diameter)

            # Both need to have the same aspect ratio
            for ax in axes:
                ax.set_box_aspect([1,1,1])

            # Ticks off on the second one cos it's distracting
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])

        # We need to know how many frames we have in the simulation
        num_frames_simulation = len(action_history)
        # We want to render at 60 fps, so how many frames will we have in the video?
        simulation_dt = dynamics.dt
        desired_fps = 25
        T = num_frames_simulation * simulation_dt
        num_frames_to_render = math.floor(T * desired_fps)

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
            action = action_history[frame]

            # Time to render, clear the old axis, but keep the extents
            ax0, ax1 = axs
            ax0.cla()
            ax1.cla()
            setup_axes(axs, x, y, z)

            def render_agent_to_axes(ax, world_view):

                # Compute, in the body frame, the 
                # locations of the rotors 1 (forward), 2 (left), 3 (backward), 4 (right)
                rotor_locations = np.array([
                    [quadcopter_diameter / 2, 0, 0],
                    [0, quadcopter_diameter / 2, 0],
                    [-quadcopter_diameter / 2, 0, 0],
                    [0, -quadcopter_diameter / 2, 0]
                ])

                # Compute, in the body frame, the end points of the thrust vectors
                scale_factor = 0.1 # TODO get from policy the action space limits
                action_vector_startpoints = rotor_locations
                action_vector_endpoints   = rotor_locations + np.array([
                    [0, 0, action[0] * scale_factor],
                    [0, 0, action[1] * scale_factor],
                    [0, 0, action[2] * scale_factor],
                    [0, 0, action[3] * scale_factor],
                ])
                
                # Everything will now be rotated and translated into place
                translation = np.array([x, y, z])
                rotation = R.from_quat([qx, qy, qz, qw])

                # Transform from local frame into world frame
                rotor_locations = rotation.apply(rotor_locations) + translation
                action_vector_startpoints = rotation.apply(action_vector_startpoints) + translation
                action_vector_endpoints   = rotation.apply(action_vector_endpoints) + translation
                    
                # Draw a black line from rotor 1 to rotor 3 and from rotor 2 to rotor 4
                # to represent the quadcopter body
                ax.plot(
                    [rotor_locations[0, 0], rotor_locations[2, 0]],
                    [rotor_locations[0, 1], rotor_locations[2, 1]],
                    [rotor_locations[0, 2], rotor_locations[2, 2]],
                    'k-',
                )
                ax.plot(
                    [rotor_locations[1, 0], rotor_locations[3, 0]],
                    [rotor_locations[1, 1], rotor_locations[3, 1]],
                    [rotor_locations[1, 2], rotor_locations[3, 2]],
                    'k-',
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
                if world_view:
                    states_so_far = state_history[:frame]
                    ax.plot(
                        [state[0] for state in states_so_far],
                        [state[1] for state in states_so_far],
                        [state[2] for state in states_so_far],
                        'b--',
                    )     

            render_agent_to_axes(ax0, world_view=True)
            render_agent_to_axes(ax1, world_view=False)

            pbar.update(1)

            return axs,
        
        ani = FuncAnimation(fig, update, frames=num_frames_to_render) #, interval=1000/desired_fps, blit=True)
        
        # Save the video
        filepath_output = os.path.join(self.visuals_folder, "render.mp4")
        ani.save(filepath_output, fps=desired_fps, extra_args=['-vcodec', 'libx264'])
        
        # Free resources
        plt.close(fig)
        pbar.close()
