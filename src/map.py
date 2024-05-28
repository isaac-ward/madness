import math
import numpy as np 
from PIL import Image
import scipy
import scipy.ndimage
import scipy.spatial    
import networkx as nx
from tqdm import tqdm

import globals
from path import Path

class Map:
    def __init__(
        self,
        map_filepath,
        metres_per_pixel,
        scale_factor=0.3
    ):
        """
        Input filepath is where we can find the map file. The map file is a png
        with only black and white pixels. Black represents obstacles and white
        represents free space

        Metres per pixel is the scale of the map in meters per pixel

        Scale factor is the factor by which to scale the map by when loading it (0.5
        means that the map will be half the size in each dimension, so it will be
        a quarter of the size in total, but coarser). This is useful for speeding up 
        path planning algorithms
        """
        
        # Load the occupancy grid 
        self.map_filepath = map_filepath
        self.metres_per_pixel = metres_per_pixel
        self.occupancy_grid = self.load_map_file_as_occupancy_grid(map_filepath)

        # Downscale to allow for easier path planning computation - note that this
        # changes the matres_per_pixel
        self.occupancy_grid = self.reduce_occupancy_grid_resolution(
            self.occupancy_grid,
            scale_factor=scale_factor,
        )
        self.metres_per_pixel /= scale_factor
        
        # The map is not going to change (TODO: is it?) so we can compute the 
        # boundary positions now
        self.boundary_positions = self.find_boundary_positions(
            self.occupancy_grid,
            self.metres_per_pixel
        ) 

        # Let's also compute a kd tree representation of the boundary positions
        # for fast collision checking
        self.boundary_positions_kd_tree = scipy.spatial.cKDTree(self.boundary_positions)

        print(f"Loaded map from {map_filepath}, scale divided by {scale_factor:.4f} to reduce from {metres_per_pixel:.4f}->{self.metres_per_pixel:.4f} metres per pixel")

    def pixels_to_metres(self, pixels):
        return np.array(pixels) * self.metres_per_pixel
    
    def metres_to_pixels(self, metres):
        return np.array(metres) / self.metres_per_pixel

    @staticmethod
    def load_map_file_as_occupancy_grid(filepath):
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

        # Horizontal will be correct, but will be upside down, so flip
        occupancy_grid = np.flipud(occupancy_grid)
        
        return occupancy_grid

    @staticmethod
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
                # TODO should be the max value in the region
                resized_grid[i, j] = occupancy_grid[orig_i, orig_j]

        return resized_grid
    
    @staticmethod
    def dilate_grid(
        occupancy_grid, 
        metres_per_pixel, 
        agent_radius_metres
    ):
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

    def path_a_to_b_metres(
        self,
        a_coord_metres, 
        b_coord_metres,
        fudge_factor=1.0
    ):
        """
        a and b are in metres, each a tuple x,y, the fudge factor is a multiplier
        on the size of the agent radius to be more conservative. 1 is the true
        size, 2 is twice as big, etc.

        Note that generally return a path that is subsampled
        very very densely - recommend using the path's 
        downsampling functions
        """
        path_metres = Map.compute_path_over_occupancy_grid(
            self.occupancy_grid,
            self.metres_per_pixel,
            a_coord_metres,
            b_coord_metres,
            agent_radius_metres=globals.DRONE_HALF_LENGTH*fudge_factor
        )
        return Path(path_metres)

    @staticmethod
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
        occupancy_grid = Map.dilate_grid(occupancy_grid, metres_per_pixel, agent_radius_metres)

        # Use A* algorithm for pathfinding
        print(f"Making graph with shape {occupancy_grid.shape}")
        graph = nx.grid_graph(dim=occupancy_grid.shape, periodic=False)
        print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

        # Define a custom heuristic function for A* (Euclidean distance)
        def heuristic_euclidean(u, v):
            (x1, y1) = u
            (x2, y2) = v
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        def heuristic_weighted_euclidean(u, v):
            (x1, y1) = u
            (x2, y2) = v
            weight = 4
            return weight * np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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

        # Find the closest nodes to the start and finish coordinates, in pixel space
        start_node = tuple(np.round(np.array(start_coord_metres) / metres_per_pixel).astype(int))
        finish_node = tuple(np.round(np.array(finish_coord_metres) / metres_per_pixel).astype(int))

        # Compute the path using A* algorithm
        try:
            path = nx.astar_path(graph, start_node, finish_node, heuristic=heuristic_euclidean, weight=weight)
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

    @staticmethod
    def find_boundary_positions(
        occupancy_grid,
        metres_per_pixel,
    ):

        """
        Finds the locations of boundaries (parts of the occupancy grid) 
        that touch both an occupied cell and an unoccupied cell, and returns
        a list of their positions in metres (not pixels)
        """
        boundary_cells = []
        rows = len(occupancy_grid)
        cols = len(occupancy_grid[0])

        # Define the neighbor offsets (8-connected grid)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (1, 1), (-1, 1), (1, -1)]
        
        for r in range(rows):
            for c in range(cols):
                if occupancy_grid[r][c] == 1:  # occupied cell
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if occupancy_grid[nr][nc] == 0:  # neighboring free cell
                                # Convert pixel coordinates to meters
                                boundary_x = c * metres_per_pixel
                                boundary_y = r * metres_per_pixel
                                boundary_cells.append((boundary_x, boundary_y))
                                break  # No need to check other neighbors for this cell

        return np.array(boundary_cells)
    
    def does_path_hit_boundary(self, path):
        """
        Returns True if the path intersects with any boundary positions
        """
        
        # Go through all points in the path - if any of them are within a certain
        # distance of a boundary position, return True
        for xy in path.path_metres:
            if self.does_point_hit_boundary(xy[0], xy[1]):
                return True
        return False
    
    def does_point_hit_boundary(self, x, y):
        """
        Returns True if the point intersects with any boundary positions
        """
        
        # Query the kd tree for the nearest boundary position
        distance, _ = self.boundary_positions_kd_tree.query(np.array([x, y]).T)
        # TODO variable threshold, in case we are not testing collision
        # with the drone
        return distance < globals.DRONE_HALF_LENGTH
    
    def out_of_bounds(self, x, y):
        """
        Returns True if the point is out of bounds
        """
        return x < 0 or y < 0 or x > self.occupancy_grid.shape[1] * self.metres_per_pixel or y > self.occupancy_grid.shape[0] * self.metres_per_pixel