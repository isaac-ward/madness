import matplotlib as mpl 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import os
import moviepy.editor as mpy
from tqdm import tqdm
import math
from concurrent.futures import ProcessPoolExecutor

import utils
import globals

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
    fps=25
):
    # We want to render at like 25 fps, but the simulation_dt
    # could be much smaller. What we'll do is iterate through the
    # number of render frames and render the closest simulation
    # frame to that time

    # So how many frames are there to render?
    num_frames_simulation = len(state_trajectory)
    T = num_frames_simulation * simulation_dt
    num_frames_to_render = int(T * fps)

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
    print(f"Non-infinite scores encountered are in range: [{min_score}, {max_score}]")
    #print(f"Have {len(scored_rollouts_per_step)} scored rollouts per step")

    # The plot is going to look like so, it will be 2 plots side by side, on the left
    # will be the world, with the occupancy grid, the paths, and the drone's position path
    # On the right will be a close up of the drone, with the current control rendered

    # Create a figure
    fig = plt.figure(figsize=(20, 8))
    # 1 row, two columns
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

    # Monitor progress with tqdm
    pbar = tqdm(total=num_frames_to_render, desc="Rendering frames")
    
    def animate(render_frame_index):
        # What simulation step are we closest to?
        sim_frame_index = math.floor(num_frames_simulation / num_frames_to_render * render_frame_index)

        # Clear the axes
        for ax in axs:
            ax.cla()

        # Plot the experiment
        plot_experiment(
            axs,
            map,
            start_point,
            finish_point,
            paths,
            simulation_dt,
            state_trajectory=state_trajectory,
            control_trajectory=control_trajectory,
            scored_rollouts=scored_rollouts_per_step[sim_frame_index],
            score_bounds=(min_score, max_score),
            progress=sim_frame_index / num_frames_simulation
        )
        pbar.update(1)
        plt.tight_layout()
        return axs
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, animate, frames=num_frames_to_render-1
    )
    #plt.tight_layout()
    anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])

def plot_experiment(
    axs,
    map,
    start_point,
    finish_point,
    paths,
    simulation_dt,
    state_trajectory=[],
    control_trajectory=[],
    scored_rollouts=[],
    score_bounds=(0,1),
    progress=1    
):
    """
    The map object, and a list of path objects that we also want to plot
    in the form {
        "path": path,
        "color": color
    }

    If a state trajectory or a control trajectory is given, then we'll plot those as well, up to the progress
    value (0 to 1)
    """
    drone_color = 'purple'

    # In the top right of the drone view we want to plot the
    # current time and the total time, and the dt
    def plot_text_y_down(ax, y, text):
        ax.text(0.01, y, text, color='black', fontsize=8, ha='left', va='bottom', font='monospace', transform=ax.transAxes)
    T = len(state_trajectory) * simulation_dt
    plot_text_y_down(axs[1], 0.95, f"dt = {simulation_dt:.4f} s")
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
        s=0.3
    )
    # Plot a scale in the bottom left corner of the world view
    annotation = f"1m = {int(1 / map.metres_per_pixel)} pixels"
    axs[0].text(0, 0.01, annotation, color='orange', fontsize=6, ha='left', va='bottom', transform=axs[0].transAxes)
    # And plot a bar in the bottom left corner that shows this scale
    scale_length = 1 / map.metres_per_pixel
    axs[0].plot([0, scale_length], [0, 0], color='orange', lw=4)
    # Don't need axis ticks
    # axs[0].axis('off')

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

    # Plot a grid on ax[0] at each metre
    axs[0].set_xticks(np.arange(0, map.occupancy_grid.shape[1], 1 / map.metres_per_pixel))
    axs[0].set_yticks(np.arange(0, map.occupancy_grid.shape[0], 1 / map.metres_per_pixel))
    # But I don't want to see the numbers
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[0].grid(True, which='both', color='grey', linestyle='--', linewidth=0.5)

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
        elif score == np.inf:
            score = 1
        else:
            score = (score - score_bounds[0]) / (score_bounds[1] - score_bounds[0])

        #print(score)
            
        color = colormap[int(score * color_samples)]
        # Plot the line
        axs[0].plot(
            pos_x,
            pos_y,
            color=color,
            lw=0.5,
            linestyle='-',
            # Draw the highest scores on top
            zorder=20+int(score*0.01),
            alpha=0.2,
        )

    # Plot the best rollout in a different color
    best_rollout_index = np.argmax(scored_rollouts[1])
    best_rollout = scored_rollouts[0][best_rollout_index]
    best_rollout = map.metres_to_pixels(best_rollout)
    axs[0].plot(
        best_rollout[:, 0],
        best_rollout[:, 2],
        color='pink',
        lw=1,
        linestyle='--',
        zorder=1000,
        alpha=0.5
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

        # Plot as arrows from the rotated top left and top right corners, if the control  
        # is not zero or very close to zero
        draw_threshold = 0.02
        if abs(left_control/globals.MAX_THRUST_PER_PROP) > draw_threshold:
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
        if abs(right_control/globals.MAX_THRUST_PER_PROP) > draw_threshold:
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