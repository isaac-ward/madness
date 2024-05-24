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
    
    def num_points_along_path(self):
        return len(self.path_metres)

    def deviation_from_other_path(self, path_b, verbose=False):
        """
        Uses fast (approximate) Dynamic Time Warping to calculate the deviation
        between this path and the other path. This takes on a value of 0 when the
        paths are identical, or if one path is a perfect subset of the other.

        Small devations have a value close to 0, while large deviations have a
        large positive value
        """

        # The problem here is that one path might be longer than the other path (both
        # in terms of number of points, and in terms of length). We need them to 
        # be the same length first (shorter), then we can subsample them to have the 
        # same number of points, then we can use fastdtw to calculate the deviation
        path_a = self 
        path_a_length = path_a.length_along_path()
        path_b_length = path_b.length_along_path()
        
        # Which path is longer?
        longer_path  = path_a if path_a_length > path_b_length else path_b
        shorter_path = path_b if path_a_length > path_b_length else path_a 

        # The last point in the longer path needs to be made the same distance
        # from it's start as the shorter path's last point is from it's start
        shorter_path_length = shorter_path.length_along_path()
        longer_path_length  = longer_path.length_along_path()
        desired_length = shorter_path_length

        if verbose:
            print(f"Shorter path length: {shorter_path_length}")
            print(f"Longer path length: {longer_path_length}")

        # If the distance to remove is 0, we're done
        if desired_length < longer_path_length:

            # We need to find the point on the longer path that is distance_from_start
            # from the start
            distances = np.linalg.norm(np.diff(longer_path.path_metres, axis=0), axis=1)
            total_distance = 0
            for i, distance in enumerate(distances):
                total_distance += distance
                if total_distance > desired_length:
                    # How much over the distance are we?
                    over_distance = total_distance - desired_length
                    # We need to interpolate between the previous point and this point
                    # to find the point that is distance_from_start from the start
                    # i's are + 1 because we're iterating over the distance between points
                    previous_point = longer_path.path_metres[i]
                    this_point = longer_path.path_metres[i+1]
                    new_point = previous_point + (this_point - previous_point) * (distance - over_distance) / distance
                    # Now we have the new point, which is going to be the last point
                    # on the longer path
                    longer_path.path_metres = np.concatenate([
                        longer_path.path_metres[:i+1],
                        np.array([new_point])
                    ])
                    break

            # The paths are now the same length, we can resample them to have the same
            # number of points along this length
            new_num_points = max(
                shorter_path.num_points_along_path(),
                longer_path.num_points_along_path()
            )
            shorter_path = shorter_path.upsample(new_num_points)
            longer_path  = longer_path.upsample(new_num_points)

        print("Paths to be compared")
        print(shorter_path.path_metres)
        print(longer_path.path_metres)

        # Can now calculate the deviation
        distance, path = fastdtw(
            shorter_path.path_metres,
            longer_path.path_metres,
            dist=euclidean
        )
        return distance
