import math
import numpy as np 
from PIL import Image
import scipy
import scipy.ndimage
import scipy.spatial    
import networkx as nx
from tqdm import tqdm
import copy
from utils.general import Cacher
import utils.logging

def test_nothing():
    return np.zeros((1,3))

def test_points():
    # It's gonna be N points from 0,0,0 to 10,0,0
    N = 100
    points = np.zeros((N,3))
    points[:,0] = np.linspace(0,10,N)
    return points

def test_column():
    # Put a big column in the way at x=20, y=0, at each z from 0 to 20
    N = 1600
    points = np.zeros((N,3))
    num_slices = 20 * 8
    points_per_slice = N // num_slices
    count = 0
    for i in range(num_slices):
        # Go around the circle at each z
        r = 0.5
        xs = [20 + r*np.cos(2*np.pi*i/points_per_slice) for i in range(points_per_slice)]
        ys = [2*np.sin(r*np.pi*i/points_per_slice) for i in range(points_per_slice)]
        z = 20 * (i / num_slices)
        for i in range(points_per_slice):
            points[count] = [xs[i], ys[i], z]
            count += 1
    return points

def test_columns():
    def generate_positions(size_x, size_y, spacing):
        # Compute the number of columns and rows needed
        num_x_columns = int(np.ceil(size_x / spacing)) + 1
        num_y_columns = int(np.ceil(size_y / spacing)) + 1

        # Create arrays to hold the x and y positions
        x_positions = np.arange(num_x_columns) * spacing
        y_positions = np.arange(num_y_columns) * spacing

        # Initialize an empty list to hold the positions
        positions = []

        # Generate the positions
        for i, x in enumerate(x_positions):
            for y in y_positions:
                if i % 2 == 0:
                    positions.append((x, y))
                else:
                    positions.append((x, y + spacing / 2))

        return positions, num_x_columns, num_y_columns
        
    # Columns from z=0 to z=30 at the following x,y positions
    xy_positions,_,_ = generate_positions(30, 30, 3)

    # Create a column at each of these positions
    points = []
    radius = 0.025
    points_around_circle = 10
    for x, y in xy_positions:
        for z in np.linspace(0, 30, 120):
            for i in range(points_around_circle):
                theta = 2 * np.pi * i / 10
                points.append([x + radius * np.cos(theta), y + radius * np.sin(theta), z])
            
    return np.array(points)

# ----------------------------------------------------------------

class Map:
    def __init__(
        self,
        map_filepath,
        voxel_per_x_metres,
        extents_metres_xyz,
        verbose=False,
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
        self.extents_metres_xyz = extents_metres_xyz
        
        # Load the map file as an occupancy grid
        #self.points = test_columns()
        self.points = test_nothing()

        # Create a voxel grid representation of the map
        num_voxels_per_axis = [
            int((extents_metres_xyz[i][1] - extents_metres_xyz[i][0]) / voxel_per_x_metres)
            for i in range(3)
        ]
        self.voxel_grid = np.zeros(num_voxels_per_axis)

        # Fill in the voxel grid
        for point in tqdm(self.points, desc="Loading point cloud into voxel grid"):
            x, y, z = point
            i, j, k = self.metres_to_voxel_coords([x, y, z])
            # If it's outside the grid, skip
            if not self.voxel_coord_in_bounds([i, j, k]):
                if verbose:
                    print(f"Point {point} or voxel {i, j, k} is out of bounds")
                continue
            else:
                self.voxel_grid[i, j, k] = 1

        # Compute a kd tree for fast collision checking
        self.kd_tree = scipy.spatial.cKDTree(self.points)

        print("Map loaded")
        print(f"\t-map_filepath: {self.map_filepath}")
        print(f"\t-voxel_per_x_metres: {self.voxel_per_x_metres}")
        print(f"\t-num_points: {len(self.points)}")
        print(f"\t-extents: x={self.extents_metres_xyz[0]}, y={self.extents_metres_xyz[1]}, z={self.extents_metres_xyz[2]}")
        print(f"\t-voxel_grid (shape): {self.voxel_grid.shape}")
        print(f"\t-voxel_grid (total): {np.prod(self.voxel_grid.shape):.0f}")
        print(f"\t-voxel_grid (occupied): {np.sum(self.voxel_grid):.0f}")
        print(f"\t-voxel_grid (occupied %): {np.sum(self.voxel_grid) / np.prod(self.voxel_grid.shape) * 100:.6f} %")
    
    # ----------------------------------------------------------------
        
    def metres_to_voxel_coords(
        self,
        metres_coords,
    ):
        """
        Say we have the point [0, 0, 10] and the extents are
        [[-10, 20], [-10, 10], [0, 20]], with a voxel_per_x_metres of 0.5

        Then the voxel grid is 60x40x40, and the point [0, 0, 10] should be
        at [20, 20, 20] in the voxel grid. Return as an integer and always
        round down
        """

        # Look to see our progression through the extents in metres
        progression = [
            (metres_coords[i] - self.extents_metres_xyz[i][0]) / (self.extents_metres_xyz[i][1] - self.extents_metres_xyz[i][0])
            for i in range(3)
        ]

        # Multiply through by the extents in voxels
        num_voxels_per_axis = self.voxel_grid.shape
        voxel_coords = [
            int(progression[i] * num_voxels_per_axis[i])
            for i in range(3)
        ]

        # Return as an integer and round down
        voxel_coords = np.floor(voxel_coords).astype(int)
        return voxel_coords

    def voxel_coords_to_metres(
        self,
        voxel_coords,
    ):
        # Inverse of the above
        num_voxels_per_axis = self.voxel_grid.shape
        progression = [
            voxel_coords[i] / num_voxels_per_axis[i]
            for i in range(3)
        ]
        metres_coords = [
            progression[i] * (self.extents_metres_xyz[i][1] - self.extents_metres_xyz[i][0]) + self.extents_metres_xyz[i][0]
            for i in range(3)
        ]
        return metres_coords
    
    # ----------------------------------------------------------------
    
    def voxel_coord_in_bounds(
        self,
        voxel_coords,
    ):
        return all([0 <= x < self.voxel_grid.shape[i] for i, x in enumerate(voxel_coords)])
    
    def is_voxel_occupied(
        self,
        voxel_coords,
        voxel_grid=None,
    ):
        # Use this map's grid if none is given
        # (we may sometimes use a processed version of the grid)
        if voxel_grid is None:
            voxel_grid = self.voxel_grid
        return voxel_grid[tuple(voxel_coords)] == 1
    
    # ----------------------------------------------------------------
    
    def batch_is_collision(
        self,
        batch_metres_xyzs,
        collision_radius,
    ):
        """
        Given a batch of points in metres, check if they are in collision with the map
        """
            
        # Query kdtree for closest occupied point
        distances, indices = self.kd_tree.query(batch_metres_xyzs)
        return distances < collision_radius
    
    def batch_is_out_of_bounds(
        self,
        batch_metres_xyzs,
    ):
        """
        Given a batch of points in metres (N,3), check if they are out of bounds
        and return a boolean array of shape (N,)
        """
        
        # All the information we need is in here: extents_metres_xyz
        # whose is shaped like [[-10, 50], [-10, 10], [0, 20]]
        batch_x_in_bounds = np.logical_and(
            self.extents_metres_xyz[0][0] <= batch_metres_xyzs[:,0],
            batch_metres_xyzs[:,0] <= self.extents_metres_xyz[0][1],
        )
        batch_y_in_bounds = np.logical_and(
            self.extents_metres_xyz[1][0] <= batch_metres_xyzs[:,1],
            batch_metres_xyzs[:,1] <= self.extents_metres_xyz[1][1],
        )
        batch_z_in_bounds = np.logical_and(
            self.extents_metres_xyz[2][0] <= batch_metres_xyzs[:,2],
            batch_metres_xyzs[:,2] <= self.extents_metres_xyz[2][1],
        )
        is_in_bounds = np.logical_and(
            np.logical_and(batch_x_in_bounds, batch_y_in_bounds),
            batch_z_in_bounds,
        )     
        is_out_of_bounds = np.logical_not(is_in_bounds)
        return is_out_of_bounds

    def batch_is_not_valid(
        self,
        batch_metres_xyzs,
        collision_radius,
    ):
        """
        Given a batch of points in metres, check if they are valid (not colliding and not out of bounds)
        """
        return np.logical_or(
            self.batch_is_collision(batch_metres_xyzs, collision_radius),
            self.batch_is_out_of_bounds(batch_metres_xyzs),
        )
    
    def is_not_valid(
        self,
        metres_xyz,
        collision_radius,
    ):
        return self.batch_is_not_valid(np.array([metres_xyz]), collision_radius)[0]

    # ----------------------------------------------------------------

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

        a_coord_metres = np.array(a_coord_metres)
        b_coord_metres = np.array(b_coord_metres)

        a_voxel_coord = self.metres_to_voxel_coords(a_coord_metres)
        b_voxel_coord = self.metres_to_voxel_coords(b_coord_metres)

        print(f"Planning path from:")
        print(f"\t-m: {a_coord_metres} -> {b_coord_metres}")
        print(f"\t-v: {a_voxel_coord} -> {b_voxel_coord}")
        
        # This can be a computationally expensive operation, so we'll cache the results
        # and only recompute if the inputs change
        computation_inputs = (
            self.voxel_grid,
            a_coord_metres,
            b_coord_metres,
            avoid_radius,
        )
        cacher = Cacher(computation_inputs)
        if cacher.exists():
            outputs = cacher.load()
        else:

            # Define a custom heuristic function for A* (Euclidean distance)
            def euclidean_distance(u, v):
                return np.linalg.norm(np.array(u) - np.array(v))

            # We will need to account for the avoidance radius. If we have a voxel map,
            # we want to use morphological dilation to effectively expand the obstacles
            # by the radius of the agent. This will allow us to plan a path that is
            # feasible for the agent to follow
            if avoid_radius > 0:
                print("Expanding obstacles / creating a buffer using morphological dilation with a spherical structuring element...", end="")
                # Create a copy of the voxel grid
                voxel_grid_expanded = np.copy(self.voxel_grid)
                # Create a spherical structuring element
                radius_voxels = int(avoid_radius / self.voxel_per_x_metres)
                structuring_element = np.zeros((2*radius_voxels+1, 2*radius_voxels+1, 2*radius_voxels+1))
                for i in range(2*radius_voxels+1):
                    for j in range(2*radius_voxels+1):
                        for k in range(2*radius_voxels+1):
                            if np.linalg.norm([i-radius_voxels, j-radius_voxels, k-radius_voxels]) <= radius_voxels:
                                structuring_element[i, j, k] = 1
                # Dilate the voxel grid
                voxel_grid_expanded = scipy.ndimage.binary_dilation(voxel_grid_expanded, structure=structuring_element)
                print("done")

            # Use A* algorithm for pathfinding
            print(f"Making graph with shape {self.voxel_grid.shape}")
            graph = nx.Graph()
            voxel_grid_shape = self.voxel_grid.shape
            # Nodes
            for i in tqdm(range(voxel_grid_shape[0]), desc="Making graph nodes"):
                for j in range(voxel_grid_shape[1]):
                    for k in range(voxel_grid_shape[2]):
                        graph.add_node((i, j, k))
            # Edges
            edges_to_add = []
            for node in tqdm(graph.nodes, desc="Making graph edges"):
                i, j, k = node
                # 6 connectivity
                neighbours = [
                    (i+1, j, k), (i-1, j, k),
                    (i, j+1, k), (i, j-1, k),
                    (i, j, k+1), (i, j, k-1),
                ]
                # If either of the nodes is occupied, don't add the edge
                for neighbour in neighbours:
                    # Check if the neighbour is out of bounds
                    if not self.voxel_coord_in_bounds(neighbour):
                        continue
                    if self.is_voxel_occupied(node, voxel_grid=voxel_grid_expanded) or self.is_voxel_occupied(neighbour, voxel_grid=voxel_grid_expanded):
                        continue
                    else:
                        edges_to_add.append((node, neighbour))
            print("Graph constructing...", end="")
            graph.add_edges_from(edges_to_add, weight=1)
            print("done")
            print(graph)

            # Compute the path using A* algorithm
            try:
                print("Solving for path...", end="")
                path_coords = nx.astar_path(
                    graph, 
                    # Needs to be tuples because the node representation
                    # used earlier was tuples
                    tuple(a_voxel_coord), 
                    tuple(b_voxel_coord), 
                    heuristic=euclidean_distance
                )
                print(f"done")
                # Convert path nodes back to coordinates in metres
                path_metres = [self.voxel_coords_to_metres(np.array([x, y, z])) for x, y, z in path_coords]

            except nx.NetworkXNoPath:
                raise ValueError(f"No path found between start ({a_coord_metres} m) and finish ({b_coord_metres} m) in the occupancy grid (shape: {self.voxel_grid.shape})")
            
            outputs = np.array(path_metres)
            cacher.save(outputs)

        # Always report
        path_length = np.sum(np.linalg.norm(np.diff(outputs, axis=0), axis=1))
        print(f"Path found with {len(outputs)} points and length {path_length:.2f} m")
        return outputs