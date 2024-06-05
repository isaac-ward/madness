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
    
    def does_path_hit_boundary(self, path, return_earliest_hit_index=False):
        """
        Returns True if the path intersects with any boundary positions
        """

        # Subsample to make sure we're not missing anything
        # Want a point each centimeter
        points_per_metre = 5 # TODO this is an expensive parameter to increase
        orig_num_points = len(path.path_metres)
        new_num_points  = int(path.length_along_path() * points_per_metre)
        # Might already be densely sampled enough
        if new_num_points > orig_num_points:
            path = path.upsample(new_num_points)
        new_num_points = len(path.path_metres)

        # TODO would prefer to move to some sort of check between points on a line,
        # i.e. is the boundary point near enough to the line between the two points?
        # Easy enough to do this, but how to do it with a kdtree?
        
        # Go through all points in the path - if any of them are within a certain
        # distance of a boundary position, return True
        for i, xy in enumerate(path.path_metres):
            if self.does_point_hit_boundary(xy[0], xy[1]):
                if return_earliest_hit_index:
                    # Need to return index in the original path
                    return True, i * (orig_num_points / new_num_points)
                else:
                    return True
        if return_earliest_hit_index:
            return False, -1
        else:
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
    
    def path_box(self,path,percent_overlap=60):
        """
        Return boxes bounding the operational space of the trajectory
        TODO: Generalize method to 3D
        """

        # Create array of box corners starting at path
        expansion_direct = 4
        path_num,path_dim = np.shape(path.path_metres)
        boxes = -1*np.ones((path_num,expansion_direct))
        max_x = max(self.boundary_positions[:,0])
        min_x = min(self.boundary_positions[:,0])
        max_y = max(self.boundary_positions[:,1])
        min_y = min(self.boundary_positions[:,1])

        # Iterate through path points and define boxes
        for _i in range(path_num):
            expansion_flag = np.ones(expansion_direct)
            current_box = np.array([path.path_metres[_i,1], # Up
                                    path.path_metres[_i,0], # Right
                                    path.path_metres[_i,1], # Down
                                    path.path_metres[_i,0]]) # Left

            while (np.sum(expansion_flag)):
                #print(expansion_flag)
                # Expand box up
                if expansion_flag[0]:
                    current_box[0] += self.metres_per_pixel
                    x_min = (int)(current_box[3]/self.metres_per_pixel)
                    x_max = (int)(current_box[1]/self.metres_per_pixel)
                    for x in range(x_min,x_max):
                        if (self.does_point_hit_boundary(x*self.metres_per_pixel, current_box[0])):
                            expansion_flag[0] = 0
                            current_box[0] -= self.metres_per_pixel
                            break

                # Expand box right
                if expansion_flag[1]:
                    current_box[1] += self.metres_per_pixel
                    y_min = (int)(current_box[2]/self.metres_per_pixel)
                    y_max = (int)(current_box[0]/self.metres_per_pixel)
                    for y in range(y_min,y_max):
                        if (self.does_point_hit_boundary(current_box[1],y*self.metres_per_pixel)):
                            expansion_flag[1] = 0
                            current_box[1] -= self.metres_per_pixel
                            break
                
                # Expand box down
                if expansion_flag[2]:
                    current_box[2] -= self.metres_per_pixel
                    x_min = (int)(current_box[3]/self.metres_per_pixel)
                    x_max = (int)(current_box[1]/self.metres_per_pixel)
                    for x in range(x_min,x_max):
                        if (self.does_point_hit_boundary(x*self.metres_per_pixel, current_box[2])):
                            expansion_flag[2] = 0
                            current_box[2] += self.metres_per_pixel
                            break
                
                # Expand box left
                if expansion_flag[3]:
                    current_box[3] -= self.metres_per_pixel
                    y_min = (int)(current_box[2]/self.metres_per_pixel)
                    y_max = (int)(current_box[0]/self.metres_per_pixel)
                    for y in range(y_min,y_max):
                        if (self.does_point_hit_boundary(current_box[3],y*self.metres_per_pixel)):
                            expansion_flag[3] = 0
                            current_box[3] += self.metres_per_pixel
                            break
                
                # Stop box from blowing up by offseting stuck location
                if (current_box[0] > max_y) or (current_box[1] > max_x) or (current_box[2] < min_y) or (current_box[3] < min_x):
                    current_box = np.array([path.path_metres[_i,1], # Up
                                            path.path_metres[_i,0], # Right
                                            path.path_metres[_i,1], # Down
                                            path.path_metres[_i,0]]) # Left
                    current_box += expansion_flag * self.metres_per_pixel
                    expansion_flag = np.ones(expansion_direct)
            
            # Store box info
            boxes[_i] = np.copy(current_box)
        
        # Box overlap analysis functions
        def rect_area(box):
            return abs(box[0]-box[2])*abs(box[1]-box[3])
        
        def overlap_percent(box1,box2):
            # Get overlap borders
            min_up = min(box1[0],box2[0])
            max_down = max(box1[2],box2[2])
            min_right = min(box1[1],box2[1])
            max_left = max(box1[3],box2[3])

            # Check if no overlap
            if min_up <= max_down or min_right <= max_left:
                return 0,0
            
            # Get rectangle areas
            overlap_box = np.array([min_up,
                                    min_right,
                                    max_down,
                                    max_left])
            overlap_area = rect_area(overlap_box)
            box1_area = rect_area(box1)
            box2_area = rect_area(box2)

            # Calc percent overlaps
            box1_percent = overlap_area/box1_area * 100
            box2_percent = overlap_area/box2_area * 100

            return box1_percent,box2_percent
        
        def point_in_box(point,box):
            x = point[0]
            y = point[1]
            if (y <= box[0]) and (y >= box[2]) and (x <= box[1]) and (x >= box[3]):
                return True
            return False
        
        # Get rid of unnecessary boxes with rounding, uniqueness, and percentage overlap methods
        boxes = np.round(boxes,5)
        boxes = np.unique(boxes,axis=0)
        boxes_temp = np.array([boxes[0]])
        for _i in range(1,np.shape(boxes)[0]):
            box_flag = 1
            for _j in range(np.shape(boxes_temp)[0]):
                percent1,percent2 = overlap_percent(boxes_temp[_j,:],boxes[_i,:])
                if (percent1 > percent2) and (percent1 >= percent_overlap):
                    boxes_temp[_j,:] = np.copy(boxes[_i,:])
                    box_flag = 0
                elif (percent2 >= percent1) and (percent2 >= percent_overlap):
                    box_flag = 0
            # If insufficient overlap, add box
            if box_flag:
                boxes_temp = np.vstack([boxes_temp,boxes[_i]])
        
        boxes_temp = np.unique(boxes_temp,axis=0)
        
        # If no overlap whatsoever, add another box to connect new box with other boxes
        for _i in range(np.shape(boxes_temp)[0]):
            no_overlap = 1
            for _j in range(np.shape(boxes_temp)[0]):
                percent1,percent2 = overlap_percent(boxes_temp[_j,:],boxes_temp[_i,:])
                if percent1 and (_i != _j):
                    no_overlap = 0
            # If there is a box without overlap, find a removed box that connects it to the most other boxes
            boxes_overlapped = np.zeros(np.shape(boxes)[0])
            if no_overlap:
                for _j in range(np.shape(boxes)[0]):
                    percent1,percent2 = overlap_percent(boxes[_j,:],boxes_temp[_i,:])
                    if percent1 and (percent1 < 100):
                        for _k in range(np.shape(boxes_temp)[0]):
                            percent1,percent2 = overlap_percent(boxes[_j,:],boxes_temp[_k,:])
                            if percent1 and (_k != _i):
                                boxes_overlapped[_j] += 1
                boxes_temp = np.vstack([boxes_temp,boxes[np.argmax(boxes_overlapped)]])
        boxes_temp = np.unique(boxes_temp,axis=0)

        # Sort boxes in order
        sorting_flag = 1
        while sorting_flag:
            sorting_flag = 0
            boxes = -1*np.ones(4)
            for _i in range(path_num):
                for _j in range(np.shape(boxes_temp)[0]):
                    expanded_box = boxes_temp[_j,:] + np.array([self.metres_per_pixel,self.metres_per_pixel,-self.metres_per_pixel,-self.metres_per_pixel])
                    if (_i != 0) and (point_in_box(path.path_metres[_i],expanded_box)):
                        percent1,percent2 = overlap_percent(boxes_temp[_j,:],boxes[-1,:])
                        if (boxes_temp[_j,:].tolist() not in boxes.tolist()) and (percent1):
                            boxes = np.vstack([boxes,boxes_temp[_j,:]])
                    elif point_in_box(path.path_metres[_i],boxes_temp[_j,:]):
                        if boxes_temp[_j,:].tolist() not in boxes.tolist():
                            boxes = np.vstack([boxes,boxes_temp[_j,:]])
            boxes = np.copy(boxes[1:])
            # Make sure boxes path didn't get stuck, and if it did remove problem box
            if (np.shape(boxes)[0] != np.shape(boxes_temp)[0]) and (not point_in_box(path.path_metres[-1],boxes[-1,:])):
                mask = ~(boxes_temp == boxes[-1,:]).all(axis=1)
                boxes_temp = boxes_temp[mask]
                sorting_flag = 1

        return boxes