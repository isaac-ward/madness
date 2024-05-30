import matplotlib as mpl 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import moviepy.editor as mpy
from tqdm import tqdm
import math
from concurrent.futures import ProcessPoolExecutor

import utils
import globals

def vis_occupancy_grid(filepath, occupancy_grid, metres_per_pixel, points_metres=[], path_metres=[], path2_metres=[], plot_coordinates=True, path_boxes=[]):
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

    # Sometimes we want a second path (e.g. a fit)
    if len(path2_metres) > 0:
        path2_pixels = np.array(path2_metres) / metres_per_pixel
        ax.plot(path2_pixels[:, 0], path2_pixels[:, 1], color='red', linewidth=1, linestyle='--')
    
    # Plot path boxes
    if len(path_boxes) > 0:
        box_pixels = np.array(path_boxes) / metres_per_pixel
        for _i in range(len(path_boxes)):
            width = box_pixels[_i,1] - box_pixels[_i,3]
            height = box_pixels[_i,0] - box_pixels[_i,2]
            rectangle = patches.Rectangle((box_pixels[_i,3], box_pixels[_i,2]), width, height, linewidth=1, edgecolor='orange', facecolor='none')
            ax.add_patch(rectangle)

    # Calculate scale bar length dynamically based on 1 meter length
    scale_bar_length = int(1 / metres_per_pixel)  # Length of scale bar in pixels
    scale_bar_text = f'1m = {scale_bar_length} pixels'

    # Plot a bar
    ax.plot([width - 1 - scale_bar_length, width - 1], [1, 1], color='orange', linewidth=1)
    ax.text(width - 1, 120*metres_per_pixel, scale_bar_text, color='orange', ha='right', fontsize=16 * (0.01/metres_per_pixel))

    # Hide axes
    ax.axis('off')

    # Black background
    fig.patch.set_facecolor('black')

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
        plt.close()
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

def generate_frame(i, num_states, map, start_point, finish_point, paths, simulation_dt,
                   state_trajectory, control_trajectory, scored_rollouts_per_step,
                   min_score, max_score, frames_folder):
    """
    Generate a frame for the given index.
    """
    # Progress is from 0 to 1
    progress = i / num_states
    frame_filepath = f"{frames_folder}/{i}.png"
    plot_experiment(
        frame_filepath,
        map,
        start_point,
        finish_point,
        paths,
        dt=simulation_dt,
        state_trajectory=state_trajectory,
        control_trajectory=control_trajectory,
        scored_rollouts=scored_rollouts_per_step[i],
        score_bounds=(min_score, max_score),
        progress=progress
    )
    return frame_filepath

def plot_experiment_video(
    filepath,
    map,
    start_point,
    finish_point,
    paths,
    simulation_dt,
    state_trajectory=[],
    control_trajectory=[],
    scored_rollouts_per_step=[],
):
    """
    Call plot experiment at some interval to generate frames and then use
    moviepy to generate a video
    """

    # Want this in realtime. If the simulation is 10x faster than realtime
    # then we want to play it back at 0.1x speed
    num_states = len(state_trajectory)
    fps_desired = 25
    
    # Create a folder to store the frames
    base_folder = os.path.dirname(filepath)
    frames_folder = f"{base_folder}/frames"
    os.makedirs(frames_folder, exist_ok=True)

    # What were the highest and lowest non infinity scores encountered 
    # in the scored rollouts? 
    min_score = np.inf
    max_score = -np.inf
    for scored_rollouts in scored_rollouts_per_step:
        scores = scored_rollouts[1]
        for score in scores:
            if score < min_score and score != -np.inf:
                min_score = score
            if score > max_score and score != +np.inf:
                max_score = score
    print(f"Non infinite scores encountered: [{min_score}, {max_score}]")
    #print(f"Have {len(scored_rollouts_per_step)} scored rollouts per step")

    # Generate frames
    frame_filepaths = []
    max_workers = 4 #os.cpu_count() - 4
    print(f"Using {max_workers} workers to generate frames")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_frame, i, num_states, map, start_point, finish_point, paths,
                            simulation_dt, state_trajectory, control_trajectory,
                            scored_rollouts_per_step, min_score, max_score, frames_folder)
            for i in range(num_states - 1)
        ]
        for future in tqdm(futures, desc="Generating frames"):
            frame_filepaths.append(future.result())

    # The filepaths need to be sorted by the frame number
    frame_filepaths = sorted(frame_filepaths, key=lambda x: int(os.path.basename(x).split('.')[0]))

    # Now use moviepy to create a video, and compress it
    clip = mpy.ImageSequenceClip(frame_filepaths, fps=fps_desired)
    clip.write_videofile(filepath, fps=fps_desired, threads=max_workers, bitrate='2M')

    # Delete the frames and folder
    delete_frames = False 
    if delete_frames:
        for frame_filepath in frame_filepaths:
            os.remove(frame_filepath)
        os.rmdir(frames_folder)

def plot_experiment(
    filepath,
    map,
    start_point,
    finish_point,
    paths,
    dt,
    state_trajectory=[],
    control_trajectory=[],
    scored_rollouts=[],
    score_bounds=(0,1),
    progress=1    
):
    """
    The filepath to save this to, the map object, and a list of path objects that we also want to plot
    in the form {
        "path": path,
        "color": color
    }

    If a state trajectory or a control trajectory is given, then we'll plot those as well, up to the progress
    value (0 to 1)
    """
    drone_color = 'purple'

    # The plot is going to look like so, it will be 2 plots side by side, on the left
    # will be the world, with the occupancy grid, the paths, and the drone's position path
    # On the right will be a close up of the drone, with the current control rendered

    # Create a figure
    fig = plt.figure(figsize=(16, 8))
    # 1 row, two columns
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

    # In the top right of the drone view we want to plot the
    # current time and the total time, and the dt
    def plot_text_y_down(ax, y, text):
        ax.text(0.01, y, text, color='black', fontsize=8, ha='left', va='bottom', font='monospace', transform=ax.transAxes)
    T = len(state_trajectory) * dt
    plot_text_y_down(axs[1], 0.95, f"dt = {dt:.4f} s")
    plot_text_y_down(axs[1], 0.91, f"t  = {progress * T:.4f} s")
    plot_text_y_down(axs[1], 0.87, f"T  = {T:.4f} s")

    # Plot the occupancy grid
    axs[0].imshow(map.occupancy_grid, cmap='binary', origin='lower')
    # Set the limits based on the occupancy grid
    axs[0].set_xlim([0, map.occupancy_grid.shape[1]])
    axs[0].set_ylim([0, map.occupancy_grid.shape[0]])
    # Plot an x on every point that is a boundary cell (self.boundary_positions)
    # Need to convert to pixels
    axs[0].scatter(
        map.metres_to_pixels(map.boundary_positions)[:, 0],
        map.metres_to_pixels(map.boundary_positions)[:, 1],
        color='orange', 
        marker=',',
        s=0.03
    )
    # Plot a scale in the bottom left corner of the world view
    annotation = f"1m = {int(1 / map.metres_per_pixel)} pixels"
    axs[0].text(0, 0.01, annotation, color='orange', fontsize=6, ha='left', va='bottom', transform=axs[0].transAxes)
    # And plot a bar in the bottom left corner that shows this scale
    scale_length = 1 / map.metres_per_pixel
    axs[0].plot([0, scale_length], [0, 0], color='orange', lw=4)
    # Don't need axis ticks
    axs[0].axis('off')

    # Now plot the points
    axs[0].scatter(
        map.metres_to_pixels([start_point])[0, 0],
        map.metres_to_pixels([start_point])[0, 1],
        color='grey',
        marker='o'
    )
    axs[0].scatter(
        map.metres_to_pixels([finish_point])[0, 0],
        map.metres_to_pixels([finish_point])[0, 1],
        color='grey',
        marker='o'
    )

    # Now plot the paths 
    for path in paths:
        p = path["path"]
        c = path["color"]
        axs[0].plot(
            map.metres_to_pixels(p.path_metres)[:, 0],
            map.metres_to_pixels(p.path_metres)[:, 1],
            color=c,
            linestyle='--',
            lw=1
        )

    # Now plot the drone's position up to the point in progress
    if len(state_trajectory) > 0:
        state_trajectory_of_interest = state_trajectory[:int(progress * len(state_trajectory))]
        state_trajectory_of_interest = map.metres_to_pixels(state_trajectory_of_interest)
        # 0, and 2 are x and y
        axs[0].plot(
            state_trajectory_of_interest[:, 0],
            state_trajectory_of_interest[:, 2],
            color=drone_color,
            linestyle='-',
            lw=1
        )

    # We have the scored rollouts - these are all the samples that
    # MPPI took, and the score that each one got. We want to plot every single
    # one, the color of the line will be based on the score
    colormap = mpl.cm.get_cmap('hsv')
    # But I actually only want the last half (the green to blue bit),
    # and in reverse
    color_samples = 255
    colormap = colormap(np.linspace(0.0, 0.33, color_samples+1))#[::-1]
    for i, _ in tqdm(enumerate(scored_rollouts[0]), desc="Plotting rollouts", leave=False, disable=True):
        Xs    = scored_rollouts[0][i]
        score = scored_rollouts[1][i]
        # Convert to pixels
        pos_x = map.metres_to_pixels(Xs[:, 0])
        pos_y = map.metres_to_pixels(Xs[:, 2])
        # Normalize the score to be between 0 and 1
        if score == -np.inf:
            score = 0
        else:
            score = (score - score_bounds[0]) / (score_bounds[1] - score_bounds[0])

        color = colormap[int(score * color_samples)]
        # Plot the line
        axs[0].plot(
            pos_x,
            pos_y,
            color=color,
            lw=0.2,
            linestyle='-',
            # Draw the highest scores on top
            zorder=20+score,
            alpha=0.1,
        )
    
    # Now we want to plot the drone centric view on the other axis
    # We'll plot the drone as a rectangle, centered at the drone's position, rotated by the angle
    # and with the control action vectors coming from the rotors. We want to plot the drone on both
    # the drone centric view and the world view, so we'll make a helper function to do so
    def plot_drone(ax, x, y, angle, control, zoom=1):
        length = zoom*globals.DRONE_HALF_LENGTH * 2
        height = zoom*globals.DRONE_HALF_LENGTH / 2

        # Compute the rectangle corners relative to the center and rotated by the angle
        bottom_left  = np.array([-length / 2, -height / 2])
        bottom_right = np.array([length / 2, -height / 2])
        top_right    = np.array([length / 2, height / 2])
        top_left     = np.array([-length / 2, height / 2])
        rectangle = np.array([bottom_left, bottom_right, top_right, top_left])

        # Rotate the rectangle
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        rectangle = (rotation_matrix @ rectangle.T).T

        # Translate the rectangle
        rectangle += np.array([x, y])

        # Draw the rectangle (always on top!)
        patch = plt.Polygon(rectangle, edgecolor=drone_color, facecolor=drone_color)
        patch.set_zorder(10)
        ax.add_patch(patch)

        # Draw a little triangle pointing up so we know the orientation
        triangle = np.array([
            [0, height / 2],
            [length / 8, 0],
            [-length / 8, 0]
        ])
        triangle = (rotation_matrix @ triangle.T).T
        triangle += np.array([x, y])
        patch = plt.Polygon(triangle, edgecolor='black', facecolor='black')
        patch.set_zorder(11)
        ax.add_patch(patch) 

        # Also need to plot the controls, which can go from 0 to globals.MAX_THRUST_PER_PROP
        # and are perpendicular to the drone's orientation
        left_control = control[0]
        right_control = control[1]
        control_length = 0.25*zoom
        left_control_vector = np.array([0, (left_control/globals.MAX_THRUST_PER_PROP) * control_length])
        right_control_vector = np.array([0, (right_control/globals.MAX_THRUST_PER_PROP) * control_length])
        left_control_vector = (rotation_matrix @ left_control_vector.T).T
        right_control_vector = (rotation_matrix @ right_control_vector.T).T

        # Plot as arrows from the rotated top left and top right corners
        ax.arrow(
            rectangle[3, 0], 
            rectangle[3, 1], 
            left_control_vector[0], 
            left_control_vector[1], 
            head_width=0.1, 
            head_length=0.1, 
            fc=drone_color,
            ec=drone_color,
            zorder=11
        )
        ax.arrow(
            rectangle[2, 0], 
            rectangle[2, 1], 
            right_control_vector[0], 
            right_control_vector[1], 
            head_width=0.1, 
            head_length=0.1, 
            fc=drone_color,
            ec=drone_color,
            zorder=11
        )
    
    # Can't plot the drone without trajectories
    if len(state_trajectory) > 0 and len(control_trajectory) > 0:

        # Where is the drone now
        current_index = math.floor(progress * len(state_trajectory))
        current_x = state_trajectory[current_index, 0]
        current_y = state_trajectory[current_index, 2]
        current_angle = state_trajectory[current_index, 4]
        current_control = control_trajectory[current_index]

        # Plot the drone on the world view
        plot_drone(
            axs[0], 
            map.metres_to_pixels([current_x])[0],
            map.metres_to_pixels([current_y])[0],
            current_angle, 
            current_control, 
            zoom=1/map.metres_per_pixel
        )

        # Plot the drone on the drone centric view
        plot_drone(axs[1], 0, 0, current_angle, current_control)

    # The drone centric view is centered around zero
    axs[1].set_xlim(-globals.DRONE_HALF_LENGTH * 2, globals.DRONE_HALF_LENGTH * 2)
    axs[1].set_ylim(-globals.DRONE_HALF_LENGTH * 2, globals.DRONE_HALF_LENGTH * 2)
    # And doesn't need ticks
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    # And should be exactly square
    axs[1].set_aspect('equal')

    # Save the figure
    plt.savefig(filepath, bbox_inches='tight', dpi=600)
    # Close the figure
    plt.close()