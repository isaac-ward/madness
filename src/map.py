import math
import numpy as np 
from PIL import Image
import scipy
import scipy.ndimage
import scipy.spatial    
import networkx as nx
from tqdm import tqdm
import copy

def test_points():
    # It's gonna be N points from 0,0,0 to 10,0,0
    N = 100
    points = np.zeros((N,3))
    points[:,0] = np.linspace(0,10,N)
    return points

class Map:
    def __init__(
        self,
        map_filepath,
        voxel_per_x_metres,
        extents_metres
    ):
        """
        Input filepath is where we can find the map file. Maps are represented with
        point clouds, where each point is a (x,y,z) in metres

        We will create a voxel grid representation of the map, where each voxel is
        voxel_per_x_metres on a side, and a cube
        """
        
        # Load the occupancy grid 
        self.map_filepath = map_filepath
        self.voxel_per_x_metres = voxel_per_x_metres
        self.extents_metres = extents_metres
        
        # Load the map file as an occupancy grid
        self.points = test_points()

        # What are the extents of the point cloud?
        def min_max_in_axis(points, axis):
            # Return the mid max, ensure they are separated by at least 
            # voxel_per_x_metres
            min_val = np.min(points[:,axis])
            max_val = np.max(points[:,axis])
            difference = max_val - min_val
            if difference < self.voxel_per_x_metres:
                min_val -= (self.voxel_per_x_metres - difference) / 2
                max_val += (self.voxel_per_x_metres - difference) / 2
            return min_val, max_val
        extents_x = min_max_in_axis(self.points, 0)
        extents_y = min_max_in_axis(self.points, 1)
        extents_z = min_max_in_axis(self.points, 2)

        # Create a voxel grid representation of the map
        num_voxels_per_axis = self.metres_to_voxel_coords([
            extents_x[1] - extents_x[0],
            extents_y[1] - extents_y[0],
            extents_z[1] - extents_z[0],
        ])
        self.voxel_grid = np.zeros(num_voxels_per_axis)

        # Fill in the voxel grid
        for point in self.points:
            x, y, z = point
            i, j, k = self.metres_to_voxel_coords([x, y, z])
            # If it's outside the grid, skip
            if not self.voxel_coord_in_bounds([i, j, k]):
                continue
            else:
                self.voxel_grid[i, j, k] = 1

        print("Map loaded")
        print(f"\t-map_filepath: {self.map_filepath}")
        print(f"\t-voxel_per_x_metres: {self.voxel_per_x_metres}")
        print(f"\t-num_points: {len(self.points)}")
        print(f"\t-extents: x={extents_x}, y={extents_y}, z={extents_z}")
        print(f"\t-voxel_grid (shape): {self.voxel_grid.shape}")

    def voxel_coords_to_metres(
        self,
        voxel_coords,
    ):
        return np.array(voxel_coords) * self.voxel_per_x_metres
    def metres_to_voxel_coords(
        self,
        metres_coords,
    ):
        _ = (np.array(metres_coords) / self.voxel_per_x_metres)
        _ = np.floor(_).astype(int)
        return _
    def voxel_coord_in_bounds(
        self,
        voxel_coords,
    ):
        return all([0 <= x < self.voxel_grid.shape[i] for i, x in enumerate(voxel_coords)])

    def plan_path(
        self,
        a_coord_metres,
        b_coord_metres,
        avoid_radius,
    ):
        """
        Uses A* or RRT* or similar to compute a path from the start coordinate
        to the finish coordinate, noting that the agent has a certain radius defining
        a circle that cannot intersect with obstacles

        Occupancy grid is matrix where 0 is free space and 1 is an obstacle,
        and the scale is given by self.voxel_per_x_metres

        Returns a list of coordinates in metres that define the points along the path
        """

        a_voxel_coord = (np.array(a_coord_metres) / self.voxel_per_x_metres).astype(int)
        b_voxel_coord = (np.array(b_coord_metres) / self.voxel_per_x_metres).astype(int)

        # TODO dilate grid to allow for avoidance

        # Use A* algorithm for pathfinding
        print(f"Making graph with shape {self.voxel_grid.shape}")
        graph = nx.grid_graph(dim=self.voxel_grid.shape, periodic=False)
        print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

        # Define a custom heuristic function for A* (Euclidean distance)
        def heuristic_euclidean(u, v):
            (x1, y1, z1) = u
            (x2, y2, z2) = v
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        # Define a custom weight function for A* (considering obstacle dilation)
        def weight(u, v, data):
            return data.get("weight", 1)

        # Set weights for edges based on the dilated grid
        edges = list(graph.edges(data=True))
        with tqdm(total=len(edges), desc="Setting graph weights to encode obstacles", disable=True) as pbar:
            for (u, v, data) in edges:
                # u and v are both nodes, that are tuples of (x, y) coordinates, but they
                # need to be transposed 
                # u = (u[1], u[0])
                # v = (v[1], v[0])
                u = (u[2], u[1], u[0])
                v = (v[2], v[1], v[0])
                # TODO: > 0.5?
                if self.voxel_grid[u] == 1 or self.voxel_grid[v] == 1:
                    data["weight"] = float('inf')
                pbar.update(1)

        # Compute the path using A* algorithm
        try:
            path_coords = nx.astar_path(graph, a_voxel_coord, b_voxel_coord, heuristic=heuristic_euclidean, weight=weight)
            # Convert path nodes back to coordinates in metres
            path_metres = np.array(path_coords) * self.voxel_per_x_metres
            
            # Subsample so that we definitely have the start and final point, but otherwise
            # only a point every X metres
            # TODO

            return path_metres
        except nx.NetworkXNoPath:
            raise ValueError(f"No path found between start ({a_coord_metres} m) and finish ({b_coord_metres} m) in the occupancy grid (shape: {self.voxel_grid.shape})")