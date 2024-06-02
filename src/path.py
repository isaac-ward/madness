import numpy as np
import scipy.ndimage
import networkx as nx
import heapq
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from fastdtw import fastdtw

class Path:
    """
    Represents a path and provides utilities for working with it
    """

    def __init__(self, path_metres):
        """
        path_metres: a list of (x, y) coordinates in metres
        """
        self.path_metres = np.array(path_metres, dtype=np.float32)

    def downsample_every_n(self, every_n_points=1):
        """
        Downsamples the path by a factor and returns a new path
        """
        return Path(self.path_metres[::every_n_points])
    
    def downsample_to_average_adjacent_distance_metres(self, desired_average_adjacent_distance_metres):
        """
        Downsamples the path to have an average distance between adjacent points (in metres)
        as desired. e.g. 'I want to have a point every 0.5 metres on average'

        Returns a new path
        """
        # Calculate the average distance between adjacent points
        distances = np.linalg.norm(np.diff(self.path_metres, axis=0), axis=1)
        average_distance = np.mean(distances)
        # Calculate the factor by which to downsample
        factor = int(average_distance / desired_average_adjacent_distance_metres)
        # If the factor is 0, we're done
        if factor == 0:
            return self
        # Downsample
        return self.downsample_every_n(factor)

    def upsample(self, num_desired_points):
        """
        Returns a new path that is directly along the current path, but with
        a different number of points. This is done by interpolating between
        each pair of points
        """

        # Not allowed to ask for fewer points than we already have
        assert num_desired_points >= self.num_points_along_path(), f"Cannot upsample to fewer points {num_desired_points} than we already have {self.num_points_along_path()}"

        # If it's the same number of points we're done
        if num_desired_points == self.num_points_along_path():
            return self
        
        # Extract x and y coordinates from the path
        x, y = self.path_metres[:,0], self.path_metres[:,1]
        
        # Calculate the cumulative distance along the path
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        
        # Create the interpolation function
        interpolator_x = interp1d(cumulative_distances, x, kind='linear', fill_value='extrapolate')
        interpolator_y = interp1d(cumulative_distances, y, kind='linear', fill_value='extrapolate')
        
        # Generate new equally spaced distances
        new_distances = np.linspace(0, cumulative_distances[-1], num_desired_points)
        
        # Interpolate new x and y coordinates
        new_x = interpolator_x(new_distances)
        new_y = interpolator_y(new_distances)
        
        # Create the new path
        new_path = list(zip(new_x, new_y))
        
        return Path(new_path)

    def smooth_5th_order_spline(self):
        pass

    def length_along_path(self):
        """
        Returns the length of the path in metres
        """
        return np.sum(np.linalg.norm(np.diff(self.path_metres, axis=0), axis=1))
    
    def length_start_to_finish(self):
        """
        Returns the length from the start to the finish of the path
        """
        return np.linalg.norm(self.path_metres[-1] - self.path_metres[0])
    
    def num_points_along_path(self):
        return len(self.path_metres)

    def other_path_in_same_direction(self, path_b):
        """
        Returns true if the other path is in the same direction
        as this path
        """

        # Need to come up with a measure of forwardness or backwardness. One
        # simple approach is to look at the order of the closest points, if
        # they're generally ascending, then we're going in the right direction,
        # if they're descending, then we're going in the wrong direction
        distances_matrix = np.linalg.norm(self.path_metres[:, np.newaxis, :] - path_b.path_metres[np.newaxis, :, :], axis=2)
        closest_indices = np.argmin(distances_matrix, axis=0)
        differences = np.diff(closest_indices)
        forwardness_measure = np.sum(differences > 0) - np.sum(differences < 0)
        
        return forwardness_measure > 0

    def deviation_from_path(self, path_b, verbose=False):
        """
        Calculate the deviation from the nominal path
        by looking at each point in the actual path
        and finding the closest point in the nominal path
        and taking the distance between them
        """
        # Calculate the deviation in a vectorized way
        distances_matrix = np.linalg.norm(self.path_metres[:, np.newaxis, :] - path_b.path_metres[np.newaxis, :, :], axis=2)
        closest_distances = np.min(distances_matrix, axis=0)
        deviation = np.sum(closest_distances)

        return deviation

    def dtw_distance(self, path_b, verbose=False):
        """
        Uses fast (approximate) Dynamic Time Warping to calculate the deviation
        between this path and the other path. This takes on a value of 0 when the
        paths are identical, or if one path is a perfect subset of the other.

        Small devations have a value close to 0, while large deviations have a
        large positive value
        """
        path_a = self.path_metres
        path_b = path_b.path_metres

        n = len(path_a)
        m = len(path_b)
        
        # Create a cost matrix
        cost = np.zeros((n, m))
        
        # Fill the first row and column
        for i in range(n):
            for j in range(m):
                cost[i, j] = np.linalg.norm(path_a[i] - path_b[j])
        
        # Initialize the accumulated cost matrix
        accumulated_cost = np.zeros((n, m))
        accumulated_cost[0, 0] = cost[0, 0]
        
        # Initialize the first row and column of the accumulated cost matrix
        for i in range(1, n):
            accumulated_cost[i, 0] = accumulated_cost[i-1, 0] + cost[i, 0]
        for j in range(1, m):
            accumulated_cost[0, j] = accumulated_cost[0, j-1] + cost[0, j]
        
        # Fill in the rest of the accumulated cost matrix
        for i in range(1, n):
            for j in range(1, m):
                accumulated_cost[i, j] = cost[i, j] + min(accumulated_cost[i-1, j],    # Insertion
                                                        accumulated_cost[i, j-1],    # Deletion
                                                        accumulated_cost[i-1, j-1])  # Match
                
        return accumulated_cost[-1, -1]