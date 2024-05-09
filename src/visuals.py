import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np

import utils

def vis_occupancy_grid(occupancy_grid, metres_per_pixel, points_metres=[], path_metres=[], plot_coordinates=True):
    """
    Draws the occupancy grid (matrix) with matplotlib, and draws
    in the bottom right corner a scale bar that is one metre long
    and labeled with the number of pixels in that one metre. Then
    draws all the points in the list `points` as red dots.
    """
    
    filepath = f"{utils.get_logs_dir()}/map_{utils.get_timestamp()}.png"
    
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