import math
import numpy as np 
from PIL import Image
import scipy
import scipy.ndimage
import scipy.spatial    
from scipy.ndimage import distance_transform_edt
import networkx as nx
from tqdm import tqdm
import copy
import trimesh
from utils.general import Cacher
import utils.logging
import warnings
import os
import matplotlib.pyplot as plt
import cvxpy


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
        self.map_name = os.path.basename(map_filepath)
        self.voxel_per_x_metres = voxel_per_x_metres
        self.extents_metres_xyz = extents_metres_xyz

        def _load_helper(filepath):
            # Use trimesh to load the obj file
            mesh = trimesh.load(filepath)
            self.points = np.array(mesh.vertices)
            print(f"Loaded map file at: {filepath}, found {len(self.points)} points")
        
        # Load the map file as an occupancy grid, if the file exists
        try:
            _load_helper(map_filepath)
            
        except Exception as e:
            # If we can't find the ob file try unzipping the .zip file of it
            folder_path = os.path.dirname(map_filepath)
            filepath_no_extension = f"{folder_path}/{self.map_name.split('.')[0]}"
            filepath_zip = filepath_no_extension + ".zip"
            if os.path.exists(filepath_zip):
                print(f"Found a .zip file instead of .obj file at: {filepath_zip}, unzipping (this only needs to be done the first time)")
                # Unzip the file to that location
                import zipfile
                with zipfile.ZipFile(filepath_zip, 'r') as zip_ref:
                    zip_ref.extractall(folder_path)
                # Try loading the obj file again
                _load_helper(map_filepath)
            else:
                warnings.warn(f"Error loading map file at: {map_filepath} ({e}), using 'nothing' test map")
                #self.points = test_columns()
                self.points = test_nothing()

        # Create a voxel grid representation of the map
        num_voxels_per_axis = [
            int((extents_metres_xyz[i][1] - extents_metres_xyz[i][0]) / voxel_per_x_metres)
            for i in range(3)
        ]
        self.voxel_grid = np.zeros(num_voxels_per_axis)

        # Fill in the voxel grid
        points_used = 0
        for point in tqdm(self.points, desc=f"Loading point cloud into voxel grid ({len(self.points)} points)"):
            x, y, z = point
            i, j, k = self.metres_to_voxel_coords([x, y, z])
            # If it's outside the grid, skip
            if not self.voxel_coord_in_bounds([i, j, k]):
                if verbose:
                    print(f"Point {point} or voxel {i, j, k} is out of bounds")
                continue
            else:
                # If it's inside the grid, set the voxel containing this
                # point to 1 (occupied)
                points_used += 1
                self.voxel_grid[i, j, k] = 1
        print(f"Loaded {points_used} points into voxel grid (remaining were out of provided bounds {self.extents_metres_xyz})")

        # NOTE we're not actually using the KD tree for collisions currently
        # We want a kd tree for fast collision checking, but we only need 
        # enough points so that the points/m^3 is never too high
        def compute_points_per_m3(points, extents_metres_xyz):
            # Compute the volume of the bounding box
            volume = np.prod([extents_metres_xyz[i][1] - extents_metres_xyz[i][0] for i in range(3)])
            return len(points) / volume
        print(f"Points per m^3: {compute_points_per_m3(self.points, self.extents_metres_xyz):.4f}")
        desired_points_per_m3 = 30
        density_downsample_ratio = desired_points_per_m3 / compute_points_per_m3(self.points, self.extents_metres_xyz)

        self.kd_tree = scipy.spatial.cKDTree(self.points)

        # Load the corresponding .in file (has the same name as the .obj file)
        # and extract the 'inside free space' point if it exists
        filepath_without_obj = os.path.splitext(map_filepath)[0]
        filepath_inside_freespace = filepath_without_obj + ".in"
        print(f"Looking for 'inside free space' file at: {filepath_inside_freespace}, which determines which connected space in the map is the navigable space")
        try:
            with open(filepath_inside_freespace, "r") as f:
                lines = f.readlines()
                # The file has one line in it with a 3d point, i.e.
                # "5, 5, 5"
                point = [float(x) for x in lines[0].strip().split(",")]
                print(f"Found 'inside free space' point: {point}")
                inside_freespace_point = point
        except:
            print(f"No 'inside free space' file found at: {filepath_inside_freespace}, all unoccupied voxels are considered free space")
        
        if inside_freespace_point is not None:
            self._mark_only_navigable_space_as_unoccupied(self.metres_to_voxel_coords(inside_freespace_point))

        print("Map loaded")
        print(f"\t-map_filepath: {self.map_filepath}")
        print(f"\t-voxel_per_x_metres: {self.voxel_per_x_metres}")
        print(f"\t-num_points: {len(self.points)}")
        print(f"\t-extents: x={self.extents_metres_xyz[0]}, y={self.extents_metres_xyz[1]}, z={self.extents_metres_xyz[2]}")
        print(f"\t-voxel_grid (shape): {self.voxel_grid.shape}")
        print(f"\t-voxel_grid (total): {np.prod(self.voxel_grid.shape):.0f}")
        print(f"\t-voxel_grid (occupied): {np.sum(self.voxel_grid):.0f}")
        print(f"\t-voxel_grid (occupied %): {np.sum(self.voxel_grid) / np.prod(self.voxel_grid.shape) * 100:.6f} %")

    def _mark_only_navigable_space_as_unoccupied(self, start_voxel, verbose=True):
        """
        Marks connected free voxels in the voxel grid starting from the given voxel coordinates
        using a numpy-based approach

        Args:
            start_voxel (list or tuple): Starting voxel coordinates as [i, j, k].
        """
        if not self.voxel_coord_in_bounds(start_voxel):
            raise ValueError(f"Start voxel {start_voxel} is out of bounds.")

        # Check if the starting voxel is free space
        i, j, k = start_voxel
        if self.voxel_grid[i, j, k] != 0:
            raise ValueError(f"Start voxel {start_voxel} is not in free space (value={self.voxel_grid[i, j, k]}).")
        else:
            print(f"Marking only navigable space as unoccupied starting from voxel {start_voxel}...")

        # Define the structure for 6-connectivity
        structure = np.array([[[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]],
                              [[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]],
                              [[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]]], dtype=int)

        # Identify all connected components of free space (value=0)
        # The input argument: 
        # "An array-like object to be labeled. Any non-zero values in input are
        # counted as features and zero values are considered the background"
        # Our voxel grid is zero where it is unoccupied and one where it is occupied,
        # so we need to invert it
        labeled_array, num_features = scipy.ndimage.label(self.voxel_grid == 0, structure=structure)

        # # For debugging
        # halfway_index = labeled_array.shape[0] // 2
        # slice_ = labeled_array[halfway_index, :, :]
        # plt.figure(figsize=(10, 10))
        # plt.imshow(slice_)
        # plt.title("Slice of labeled array")
        # plt.colorbar()
        # # Save the plot
        # plt.savefig("slice_of_labeled_array.png")

        # This essentially does segmentation, and gives each connected space
        # a label
        if verbose:
            print(f"Distinct regions/segment labels in voxel map: {np.unique(labeled_array)}")

        # How many of each label are found?
        if verbose:
            for label in np.unique(labeled_array):
                print(f"\t- Label {label} has {np.sum(labeled_array == label)} voxels")

        # Find the label of the connected component containing the start voxel
        start_label = labeled_array[i, j, k]
        print(f"Start voxel label: {start_label}")

        # Count how much space was marked as navigable
        total_voxels = np.prod(self.voxel_grid.shape)

        # Print some statistics
        connected_to_start_voxels = np.sum(labeled_array == start_label)
        print(f"{connected_to_start_voxels} / {total_voxels} voxels found connected to the provided start voxel ({100*connected_to_start_voxels / total_voxels:.4f} %)")
        unoccupied_voxels = np.sum(self.voxel_grid == 0)
        print(f"{unoccupied_voxels} / {total_voxels} voxels were unoccupied (previously=0) ({100*unoccupied_voxels / total_voxels:.4f} %)")
        
        # Mark all voxels NOT in the same connected component as the start
        # label / 'inside free space voxel' as occupied
        self.voxel_grid[labeled_array != start_label] = 1
        # Mark the start voxel region as unoccupied
        self.voxel_grid[labeled_array == start_label] = 0

        navigable_voxels = np.sum(self.voxel_grid == 0)
        print(f"{navigable_voxels} / {total_voxels} voxels are navigable (now=0) ({100*navigable_voxels / total_voxels:.4f} %)")

    
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
    
    def batch_voxel_coords_to_metres(
        self,
        batch_voxel_coords,
    ):
        return np.array([ self.voxel_coords_to_metres(x) for x in batch_voxel_coords ])
    
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
        # If its out of bounds we consider it occupied
        if not self.voxel_coord_in_bounds(voxel_coords):
            return True
        return voxel_grid[tuple(voxel_coords)] == 1
    
    # ----------------------------------------------------------------
    
    def batch_is_collision_metres_xyz(
        self,
        batch_metres_xyzs,
        collision_radius,
    ):
        """
        Given a batch of points in metres, check if they are in collision with the map
        """
            
        # Query kdtree for closest occupied point
        # distances, indices = self.kd_tree.query(batch_metres_xyzs)
        # return distances < collision_radius
        # Convert to voxels and check if they are occupied
        batch_voxel_coords = [self.metres_to_voxel_coords(metres_xyz) for metres_xyz in batch_metres_xyzs]
        return self.batch_is_collision_voxel_coords(batch_voxel_coords, collision_radius)
    
    def batch_is_collision_voxel_coords(
        self,
        batch_voxel_coords,
        collision_radius_voxels,
    ):
        # Given a batch of voxel coordinates, check if they are in collision with the map
        # by checking if the voxel at that coordinate is occupied
        return np.array([
            self.is_voxel_occupied(voxel_coords)
            for voxel_coords in batch_voxel_coords
        ])
        
    def batch_is_out_of_bounds_metres_xyz(
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
            self.batch_is_collision_metres_xyz(batch_metres_xyzs, collision_radius),
            self.batch_is_out_of_bounds_metres_xyz(batch_metres_xyzs),
        )
    
    def is_not_valid(
        self,
        metres_xyz,
        collision_radius,
    ):
        # Helper for a single item which just calls the batch version
        return self.batch_is_not_valid(np.array([metres_xyz]), collision_radius)[0]

    # ----------------------------------------------------------------
    
    def get_voxels_within_radius(
        self,
        center_voxel_coords,
        radius_voxels,
    ):
        # Given a center voxel coordinate and a radius in voxels, return all the voxels
        # within that radius (sphere)

        # Voxels are given as integer values
        center_voxel_coords_ints = [int(center_voxel_coords[0]),int(center_voxel_coords[1]),int(center_voxel_coords[2])]
        radius_voxels = int(radius_voxels)

        return np.array([
            [x, y, z]
            for x in range(center_voxel_coords_ints[0] - radius_voxels, center_voxel_coords_ints[0] + radius_voxels + 1,1)
            for y in range(center_voxel_coords_ints[1] - radius_voxels, center_voxel_coords_ints[1] + radius_voxels + 1,1)
            for z in range(center_voxel_coords_ints[2] - radius_voxels, center_voxel_coords_ints[2] + radius_voxels + 1,1)
            if np.linalg.norm(np.array([x, y, z]) - np.array(center_voxel_coords_ints)) < radius_voxels
        ])

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
            self.map_filepath,
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

                    # If the node and the neighbor are both unoccupied, add the edge
                    if  not self.is_voxel_occupied(node,      voxel_grid=voxel_grid_expanded) and \
                        not self.is_voxel_occupied(neighbour, voxel_grid=voxel_grid_expanded):
                        # This means we only end up with edges between unoccupied voxels, so 
                        # A* can only be solved through unoccupied space
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

                # But the first and last points will be munged into integers because they went
                # metres (unrounded) -> voxels -> metres (rounded)
                # So we need to reset those to exact
                path_metres[0] = a_coord_metres
                path_metres[-1] = b_coord_metres

            except nx.NetworkXNoPath:
                
                # # Visualize the voxel map and the start and goal points
                # # as slices from z=0 to z=max
                # for z in range(self.voxel_grid.shape[2]):
                #     slice_ = self.voxel_grid[:,:,z]
                #     if z == a_voxel_coord[2]:
                #         slice_[a_voxel_coord[0], a_voxel_coord[1]] = 5
                #     if z == b_voxel_coord[2]:
                #         slice_[b_voxel_coord[0], b_voxel_coord[1]] = 10
                #     plt.figure(figsize=(10, 10))
                #     plt.imshow(slice_)
                #     plt.title(f"Slice at z={z}")
                #     # colorbar range 0-10
                #     plt.colorbar()
                #     # Save the plot
                #     plt.savefig(f"slice_at_z_{z}.png")
                #     plt.close()

                raise ValueError(f"No path found between start ({a_coord_metres} m) and finish ({b_coord_metres} m) in the occupancy grid (shape: {self.voxel_grid.shape}). As voxels: {a_voxel_coord} -> {b_voxel_coord}. Avoidance radius: {avoid_radius} m")
            
            outputs = np.array(path_metres)
            cacher.save(outputs)

        # Always report
        path_length = np.sum(np.linalg.norm(np.diff(outputs, axis=0), axis=1))
        print(f"Path found with {len(outputs)} points and length {path_length:.2f} m")
        return outputs

    # ----------------------------------------------------------------

    def compute_spheres_along_path(self, path, avoid_radius):
        """
        In this function we're given a path in 3d. We want to return a list of spheres, the
        first of which is centered at the first point in the path, and the last of which 
        contains the last point in the path. All intermediate spheres are centered at points
        along the path, and have the largest radius possible without colliding with the map 
        (including the avoidance radius). Adjacent spheres should overlap
        """

        class Sphere:
            def __init__(self, center, radius):
                self.center = center # 3-tuple, metres
                self.radius = radius # float, metres

            @staticmethod   
            def get_largest_non_colliding_sphere_at_xyz(xyz, map_, avoid_radius):
                # Start with the largest radius possible
                largest_map_extent = np.max(map_.extents_metres_xyz)
                largest_radius_possible = largest_map_extent / 4
                # We'll decrement x times to zero
                decrement = largest_radius_possible / 100
                pbar = tqdm(
                    np.arange(largest_radius_possible, 0, -decrement), 
                    desc=f"Finding non-colliding sphere: center={[float(f'{f:.2f}') for f in xyz]}, radius_max={largest_radius_possible:.2f} m, decrement={decrement:.2f} m", 
                    leave=False
                )
                for radius in pbar:
                    pbar.set_postfix({"radius": f"{radius:.2f} m"})
                    attempt = Sphere(xyz, radius)
                    if attempt.are_all_internal_points_valid(map_=map_, step_size_m=1):
                        # Return a slightly smaller one to really avoid hitting stuff:
                        return attempt.shrink(0.75)
                # If we can't find a non-colliding sphere, throw an error
                raise ValueError(f"Couldn't find a non-colliding sphere at {xyz} m, trying to decrement from {largest_radius_possible} m in decrements of {decrement} m")
            
            def is_point_inside(self, point):
                return np.linalg.norm(np.array(point) - np.array(self.center)) < self.radius
            
            def are_all_internal_points_valid(self, map_, step_size_m):
                step = step_size_m
                points = []
                for x in np.arange(self.center[0] - self.radius, self.center[0] + self.radius, step):
                    for y in np.arange(self.center[1] - self.radius, self.center[1] + self.radius, step):
                        for z in np.arange(self.center[2] - self.radius, self.center[2] + self.radius, step):
                            if self.is_point_inside([x, y, z]):
                                points.append([x, y, z])
                #print(f"Checking {len(points)} points inside sphere")
                return not map_.batch_is_not_valid(np.array(points), avoid_radius).any()

            def find_furthest_point_along_path_in_sphere(self, path):
                # Go along the path from the start and the first point in the sphere
                for i, point in enumerate(path):
                    if self.is_point_inside(point):
                        first_point_idx = i
                        break

                # Now we go along the path from the first point until we leave the sphere
                for i, point in enumerate(path[first_point_idx:]):
                    if not self.is_point_inside(point):
                        # We found a point outside the sphere, so return the previous
                        # one with respect to the path's total indexing
                        return path[first_point_idx + i - 1]
                    
                # If we never leave the sphere, return the last point in the path
                return path[-1]
                    
            def shrink(self, shrinkage_factor):
                return Sphere(self.center, self.radius * shrinkage_factor)
    
            def sdf_value(self, xyz):
                # Inside is positive, outside negative
                # 1 is right in the middle,
                # 0 is the edge (at the radius)
                return 1 - np.linalg.norm(np.array(xyz) - np.array(self.center)) / self.radius
            
            def sdf_value_cvx(self, xyz):
                return 1 - cvxpy.norm2(xyz - np.array(self.center)) / self.radius
            
            def __str__(self):
                return f"Sphere(center={self.center}, radius={self.radius})"
                    
        # Now we take the path, and create the largest non-colliding sphere at
        # the starting point
        print(f"Computing navigable spheres along path with {len(path)} points")
        spheres = [Sphere.get_largest_non_colliding_sphere_at_xyz(path[0], self, avoid_radius)]
        # Now we enter a loop where we find the furthest point along the path in the last
        # sphere, and then create a new sphere at that point, and then repeat until the
        # last sphere contains the last point in the path
        max_iterations = 256
        pbar = tqdm(range(max_iterations), desc="Computing spheres along path")
        for i in pbar:
            furthest_point = spheres[-1].find_furthest_point_along_path_in_sphere(path)
            # If the furthest point is the last point in the path, we're done
            if np.allclose(furthest_point, path[-1]):
                #print (f"Found the last point in the path in the last sphere after {i} iterations")
                break
            # Otherwise, create a new sphere at the furthest point
            spheres.append(Sphere.get_largest_non_colliding_sphere_at_xyz(furthest_point, self, avoid_radius))
            # Report in the pbar the distance to goal point
            pbar.set_postfix({"distance_to_goal": np.linalg.norm(furthest_point - path[-1])})
        else:
            raise ValueError(f"Couldn't find a non-colliding sphere for the last point in the path after {max_iterations} iterations")
        
        # We did it! Report in ascii the number of spheres (start)->o->O->o->(end) where we have
        # as many os as there are spheres found. Randomly capitalize for fun
        start_rounded = [round(float(x), 1) for x in path[0]]
        end_rounded = [round(float(x), 1) for x in path[-1]]
        def random_oO():
            return 'o' if np.random.rand() < 0.5 else 'O'
        print(f"Solution contains {len(spheres)} spheres! ({start_rounded})->{'->'.join([random_oO() for i in range(len(spheres)-2)])}->({end_rounded})")
        print(f"\t-only navigable space included ... ✓")
        print(f"\t-includes start and end points ... ✓")
        print(f"\t-spheres overlap ... ✓")

        return spheres
            

                