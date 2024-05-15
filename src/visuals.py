import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np
import os
import moviepy.editor as mpy
from tqdm import tqdm

import utils

def vis_occupancy_grid(filepath, occupancy_grid, metres_per_pixel, points_metres=[], path_metres=[], plot_coordinates=True):
    """
    Draws the occupancy grid (matrix) with matplotlib, and draws
    in the bottom right corner a scale bar that is one metre long
    and labeled with the number of pixels in that one metre. Then
    draws all the points in the list `points` as red dots.
    """
    
    # Define image size based on occupancy grid dimensions
    height, width = occupancy_grid.shape

    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Plot occupancy grid
    ax.imshow(occupancy_grid, cmap='binary', origin='lower')

    # Plot points
    for point in np.array(points_metres):
        # Convert from metres to pixels
        x = point[0] / metres_per_pixel
        y = point[1] / metres_per_pixel
        ax.scatter(x, y, color='red', marker='x')
        if plot_coordinates:
            # Plot the coordinates as given and in metres
            string = f'  ({point[0]:.2f}, {point[1]:.2f}) metres\n  ({x:.2f}, {y:.2f}) pixels'
            ax.text(x, y, string, color='red', fontsize=6)

    # Plot path
    if len(path_metres) > 0:
        path_pixels = np.array(path_metres) / metres_per_pixel
        ax.plot(path_pixels[:, 0], path_pixels[:, 1], color='blue', linewidth=1, linestyle='--')
        # And plot points with xs
        # for point in path_pixels:
        #     ax.scatter(point[0], point[1], color='blue', marker='x') 

    # Calculate scale bar length dynamically based on 1 meter length
    scale_bar_length = int(1 / metres_per_pixel)  # Length of scale bar in pixels
    scale_bar_text = f'1m = {scale_bar_length} pixels'

    # Plot a bar
    ax.plot([width - 1 - scale_bar_length, width - 1], [1, 1], color='orange', linewidth=1)
    ax.text(width - 1, 120*metres_per_pixel, scale_bar_text, color='orange', ha='right', fontsize=16 * (0.01/metres_per_pixel))

    # Hide axes
    ax.axis('off')

    # Save figure
    plt.savefig(filepath, bbox_inches='tight', dpi=600)

def plot_trajectory(
        filepath, 
        state_trajectory, 
        state_element_labels,
        action_trajectory,
        action_element_labels,
        dt  
    ):
    """
    State trajectory is an iterable where each iterate is 
    a state vector

    State element labels is a list of strings that label the
    elements of the state vector

    Equivalent for action trajectory
    """

    state_trajectory = np.array(state_trajectory)
    action_trajectory = np.array(action_trajectory)

    # Plot as a video, so save the frames to a folder
    # and then use moviepy
    base_folder = os.path.dirname(filepath)
    frames_folder = f"{base_folder}/frames"
    os.makedirs(frames_folder, exist_ok=True)

    # Box to plot in
    x_limit = [-20, 20]
    y_limit = [-20, 20]

    # Expand limits to fit full trajectory
    x_limit = [min(x_limit[0], np.min(state_trajectory[:, 0])), max(x_limit[1], np.max(state_trajectory[:, 0]))]
    y_limit = [min(y_limit[0], np.min(state_trajectory[:, 2])), max(y_limit[1], np.max(state_trajectory[:, 2]))]

    # But the limits must be the same for the aspect ratio to be correct
    x_range = x_limit[1] - x_limit[0]
    y_range = y_limit[1] - y_limit[0]
    if x_range > y_range:
        y_limit[0] -= (x_range - y_range) / 2
        y_limit[1] += (x_range - y_range) / 2
    else:
        x_limit[0] -= (y_range - x_range) / 2
        x_limit[1] += (y_range - x_range) / 2

    def plot_frame(i):
        # We'll plot the state and action at this time step
        # in an easy to view way
        state = state_trajectory[i]
        action = action_trajectory[i]

        # Create a single figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Set axes limits based on full trajectory
        ax.set_xlim(x_limit)
        ax.set_ylim(y_limit)
        
        # Compute the current time step and the total time and plot it
        # in the top right corner
        t = i * dt
        T = len(state_trajectory) * dt
        def plot_text_y_down(y, text):
            ax.text(0.99, y, text, color='black', fontsize=12, ha='right', va='bottom', transform=ax.transAxes)
        plot_text_y_down(0.95, f"t  = {t:.4f} s")
        plot_text_y_down(0.91, f"dt = {dt:.4f} s")
        plot_text_y_down(0.87, f"T  = {T:.4f} s")

        # Plot the drone as a rectangle, centered at the drone's position
        # which is state[0] and state[2], rotated by state[4]
        visualization_scale = 1
        drone_length = visualization_scale*2 # in metres
        drone_width = visualization_scale*1
        angle = state[4]
        center_x, center_y = state[0], state[2]

        # Compute the rectangle corners relative to the center and rotated by the angle
        corners = np.array([
            # The order is bottom left, bottom right, top right, top left
            [-drone_length / 2, -drone_width / 2],
            [drone_length / 2, -drone_width / 2],
            [drone_length / 2, drone_width / 2],
            [-drone_length / 2, drone_width / 2]
        ])
        
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        translation = np.array([center_x, center_y])
        
        corners_world_Frame = (rotation_matrix @ corners.T).T + translation
        # Plot with hatching
        rectangle = plt.Polygon(
            corners_world_Frame, 
            edgecolor='black', 
            # Transparent face
            facecolor='none',
        )
        ax.add_patch(rectangle)

        # Plot the positions thus far
        ax.plot(state_trajectory[:i, 0], state_trajectory[:i, 2], color='blue', linestyle='--')

        # Plot the first action element as a vector coming from the left rotor
        # and the second from the right rotor. The vector should be scaled by
        # the action value and perpendicular to the drone's orientation, and
        # should be a little arrow        
        def plot_vector(drone_frame_offset, rotation, translation, length, color, x_offset, y_offset, ha):
            world_frame_offset = rotation @ drone_frame_offset + translation
            world_frame_vector = rotation @ np.array([0, length])
            ax.arrow(
                world_frame_offset[0], 
                world_frame_offset[1], 
                world_frame_vector[0], 
                world_frame_vector[1], 
                head_width=0.1, 
                head_length=0.1, 
                fc=color,
                ec=color
            )

            # Plot the text of the action value
            ax.text(
                world_frame_offset[0] + x_offset,
                world_frame_offset[1] + y_offset,
                f"{length:.2f}", 
                color=color,
                fontsize=12,
                ha=ha,
                va='center',
            )

        # Plot left then right
        plot_vector(
            corners[-1], 
            rotation_matrix, 
            translation, 
            action[0],
            'green',
            -1,
            0,
            'left',
            
        )
        plot_vector(
            corners[-2],
            rotation_matrix, 
            translation, 
            action[1],
            'red',
            1,
            0,
            'right',
        )

        # Set the dpi to 600
        fig.set_dpi(600)
        # Save the figure to the frames directory
        frame_filepath = f"{frames_folder}/{i}.png"
        fig.savefig(frame_filepath)
        return frame_filepath
    
    # Plot all the frames
    frame_filepaths = []
    for i in tqdm(range(len(action_trajectory)), desc="Plotting frames"):
        frame_filepaths.append(plot_frame(i))

    # Now use moviepy to create a video
    fps = 25
    clip = mpy.ImageSequenceClip(frame_filepaths, fps=fps)
    clip.write_videofile(filepath, fps=fps)

    # Delete the frames using os
    for frame_filepath in frame_filepaths:
        os.remove(frame_filepath)
    # Then the frames folder
    os.rmdir(frames_folder)