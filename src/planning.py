import numpy as np
import scipy.ndimage
import networkx as nx
import heapq
from tqdm import tqdm

def dilate_grid(occupancy_grid, metres_per_pixel, agent_radius_metres):
    """
    Dilates the occupancy grid to include the agent's radius.

    Parameters:
    - occupancy_grid: 2D numpy array representing the occupancy grid
    - metres_per_pixel: Scale of the occupancy grid in meters per pixel
    - agent_radius_metres: Radius of the agent in meters

    Returns:
    - dilated_grid: Dilated occupancy grid
    """

    if agent_radius_metres <= 0:
        return occupancy_grid

    # Calculate the required dilation radius in pixels
    dilation_radius_pixels = int(agent_radius_metres / metres_per_pixel)

    # Create a circular structuring element
    structuring_element = np.zeros((dilation_radius_pixels*2+1, dilation_radius_pixels*2+1), dtype=bool)
    y, x = np.ogrid[:structuring_element.shape[0], :structuring_element.shape[1]]
    center = (dilation_radius_pixels, dilation_radius_pixels)
    mask = (x - center[0])**2 + (y - center[1])**2 <= dilation_radius_pixels**2
    structuring_element[mask] = 1

    # Dilate the occupancy grid
    dilated_grid = scipy.ndimage.binary_dilation(occupancy_grid, structure=structuring_element)

    return dilated_grid

def compute_path_over_occupancy_grid(
        occupancy_grid, 
        metres_per_pixel, 
        start_coord_metres, 
        finish_coord_metres,
        agent_radius_metres
    ):
    """
    Uses A* or RRT* or similar to compute a path from the start coordinate
    to the finish coordinate, noting that the agent has a certain radius defining
    a circle that cannot intersect with obstacles.

    Occupancy grid is matrix where 0 is free space and 1 is an obstacle,
    and the scale is given by metres_per_pixel.
    Start and finish are coordinates in metres.

    Returns a list of coordinates in metres that define the points along the path
    """

    print(f"Computing path from {start_coord_metres} m to {finish_coord_metres} m")

    # Dilate the occupancy grid to include the agent's radius if necessary
    occupancy_grid = dilate_grid(occupancy_grid, metres_per_pixel, agent_radius_metres)

    # Use A* algorithm for pathfinding
    print(f"Making graph with shape {occupancy_grid.shape}")
    graph = nx.grid_graph(dim=occupancy_grid.shape, periodic=False)
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Define a custom heuristic function for A* (Euclidean distance)
    def heuristic(u, v):
        (x1, y1) = u
        (x2, y2) = v
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Define a custom weight function for A* (considering obstacle dilation)
    def weight(u, v, data):
        return data.get("weight", 1)

    # Set weights for edges based on the dilated grid
    edges = list(graph.edges(data=True))
    with tqdm(total=len(edges), desc="Setting graph weights to encode obstacles", disable=True) as pbar:
        for (u, v, data) in edges:
            # u and v are both nodes, that are tuples of (x, y) coordinates, but they
            # need to be transposed 
            u = (u[1], u[0])
            v = (v[1], v[0])
            # TODO: > 0.5?
            if occupancy_grid[u] == 1 or occupancy_grid[v] == 1:
                data["weight"] = float('inf')
            pbar.update(1)

    # Find the closest nodes to the start and finish coordinates
    start_node = tuple(np.round(np.array(start_coord_metres) / metres_per_pixel).astype(int))
    finish_node = tuple(np.round(np.array(finish_coord_metres) / metres_per_pixel).astype(int))

    # Compute the path using A* algorithm
    try:
        path = nx.astar_path(graph, start_node, finish_node, heuristic=heuristic, weight=weight)
        # Convert path nodes back to coordinates in metres
        path_metres = [(coord[1] * metres_per_pixel, coord[0] * metres_per_pixel) for coord in path]
    
        # Need to transpose
        path_metres = np.array(path_metres)
        path_metres = np.flip(path_metres, axis=1)
        
        # Subsample so that we definitely have the start and final point, but otherwise
        # only a point every X metres
        # TODO

        return path_metres
    except nx.NetworkXNoPath:
        raise ValueError(f"No path found between start ({start_coord_metres} m) and finish ({finish_coord_metres} m) in the occupancy grid (shape: {occupancy_grid.shape})")

