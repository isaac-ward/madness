import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_angles_rad_to_quaternion(phi, theta, psi):
    """
    Converts euler angles to a quaternion
    """
    r = R.from_euler('xyz', [psi, theta, phi], degrees=False)
    return r.as_quat()

def quaternion_to_euler_angles_rad(x, y, z, w):
    """
    Converts a quaternion to euler angles
    """
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz', degrees=False)

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r3 = r1 * r2
    return r3.as_quat()
