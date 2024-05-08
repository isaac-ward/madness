import math
import numpy as np 
import scipy
from PIL import Image

def load_map_file_as_occupancy_grid(filepath, metres_per_pixel=0.01):
    """
    A map file is a png with only black and white pixels. Black represents
    obstacles and white represents free space. This function reads the map
    file and returns the occupancy grid (i.e., a matrix). Default scale is
    1 pixel = 1 cm.
    

    We have the following maps designed for 1 pixel = 1 cm:
    - 3x7.png: A 3 metre tall straight hallway that is 7 metres long
    - 3x28.png: A 3 metre tall straight hallway that is 28 metres long
    - downup.png: A 3 metre tall hallway that goes down 6 metres and then up 6 metres over 28 metres horizontally.
    - downup-obstacles.png: downup.png with obstacles in the lower region
    """
    
    # Open the image file
    img = Image.open(filepath)
    
    # Convert the image to grayscale
    img = img.convert("L")
    
    # Convert the image to a numpy array
    occupancy_grid = np.array(img)
    
    # Threshold the image to get binary values (0 for obstacles, 1 for free space)
    occupancy_grid = (occupancy_grid < 128).astype(int)
    
    # Convert from pixels to meters based on the given scale
    occupancy_grid = occupancy_grid * metres_per_pixel

    # Horizontal will be correct, but will be upside down, so flip
    occupancy_grid = np.flipud(occupancy_grid)
    
    return occupancy_grid

def reduce_occupancy_grid_resolution(occupancy_grid, scale_factor):
    """
    If the input occupancy grid is n x m, this function returns an occupancy grid
    that is n*scale_factor x m*scale_factor. This is useful for speeding up path
    planning algorithms. Note that this does change the scale of the occupancy grid
    """

    # Get dimensions of the input occupancy grid
    height, width = occupancy_grid.shape

    # Calculate new dimensions based on the scale factor
    new_height = math.ceil(height * scale_factor)
    new_width = math.ceil(width * scale_factor)

    # Reshape the occupancy grid to a larger size
    resized_grid = np.zeros((new_height, new_width))

    # Fill in the values in the resized grid using interpolation
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the corresponding indices in the original grid
            orig_i = int(i / scale_factor)
            orig_j = int(j / scale_factor)
            # Assign the value from the original grid to the corresponding position in the resized grid
            resized_grid[i, j] = occupancy_grid[orig_i, orig_j]

    return resized_grid