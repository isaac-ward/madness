import numpy as np 
import scipy

def compute_path_over_occupancy_grid(
        occupancy_grid, 
        start_metres, 
        finish_metres, 
        agent_radius_metres
    ):
    """
    Uses A* or RRT* or similar to compute a path from the start coordinate
    to the finish coordinate, noting that the agent has a certain radius defining
    a circle that cannot intersect with obstacles.
    """
    pass