import numpy as np
from scipy.spatial.transform import Rotation as R

from numba import njit

def euler_angles_rad_to_quaternion(phi, theta, psi):
    """
    Converts euler angles to a quaternion works for
    single inputs or batch inputs
    """
    euler_angles = np.column_stack((phi, theta, psi))
    r = R.from_euler('xyz', euler_angles, degrees=False)
    r = r.as_quat()
    
    # If it's just one value (not a batch) then return just the quaternion
    if len(r) == 1:
        return r[0]
    return r

def quaternion_to_euler_angles_rad(x, y, z, w):
    """
    Converts a quaternion to euler angles works for
    single inputs or batch inputs
    """
    quaternions = np.column_stack((x, y, z, w))
    r = R.from_quat(quaternions)
    r = r.as_euler('xyz', degrees=False)

    # If it's just one value (not a batch) then return just the euler angles
    if len(r) == 1:
        return r[0]
    return r

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions, again works for batches of quaternions
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r3 = r1 * r2
    r3 = r3.as_quat()
    return r3

def shortest_distance_between_path_and_point(path, point):
    """
    Given a path and a point, return the shortest distance between the path and the point
    """
    distance_to_all_points_in_path = np.linalg.norm(np.array(path) - point, axis=1)
    return np.min(distance_to_all_points_in_path)

def forwardness_of_path_a_wrt_path_b(path_a, path_b):
    """
    Returns > 0 if as you traverse path_a, you are going along path_b in the forward direction
    or < 0 if as you traverse path_a, you are going along path_b in the backward direction
    """
    # Convert paths to numpy arrays for easier manipulation
    path_a = np.array(path_a)
    path_b = np.array(path_b)
    
    # Calculate distances matrix between each point in path_a and path_b
    distances_matrix = np.linalg.norm(path_a[:, np.newaxis, :] - path_b[np.newaxis, :, :], axis=2)
    
    # Find the closest point in path_b for each point in path_a
    closest_indices = np.argmin(distances_matrix, axis=1)
    
    # Calculate differences between consecutive closest indices
    differences = np.diff(closest_indices)
    
    # Measure of forwardness: mean of the differences
    forwardness_measure = np.mean(differences)
    
    return forwardness_measure