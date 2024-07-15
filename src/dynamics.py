import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import os
from numba import njit, prange

import utils.general as general
import utils.geometric as geometric

class DynamicsQuadcopter3D:
    """
    This class computes the dynamics for a 3D quadcopter

    Implements the state space model from page 18, section 2.6
    https://www.kth.se/polopoly_fs/1.588039.1600688317!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf

    Initialize with the constants, and then call the step function 
    with the current state and action to get the next state
    
    We'll use the following state representation:
    state = [x, y, z, rx, ry, rz, vx, vy, vz, p, q, r]
    Some important stuff:
    - +x is forward, +y is left, +z is up
    - '+' quadcopter configuration
    - w1, w2, w3, w4 are the angular velocities of the rotors, looking down from above:
    -- w1 is along the +y axis (left), CW
    -- w2 is along the +x axis (forward), CCW
    -- w3 is along the -y axis (right), CW
    -- w4 is along the -x axis (backward), CCW
    """
    def __init__(
        self,
        diameter,        # in metres
        mass,            # in kg
        Ix,              # moment of inertia about x axis
        Iy,              # moment of inertia about y axis
        Iz,              # moment of inertia about z axis
        g,               # acceleration due to gravity (negative if down)
        thrust_coef,     # thrust coefficient (relates rotor angular velocity to yaw torque)
        drag_coef,       # drag coefficient (relates velocity to drag force)
        dt,              # time step
    ):
        self.diameter = diameter
        self.mass = mass
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.g = g
        self.thrust_coef = thrust_coef
        self.drag_coef   = drag_coef
        self.dt = dt

    def state_size(self):
        return 12
    def action_size(self):
        return 4
    def state_plot_groups(self):
        return [3, 3, 3, 3]
    def action_plot_groups(self):
        return [4]
    def state_labels(self):
        return ["x", "y", "z", "rx", "ry", "rz", "vx", "vy", "vz", "p", "q", "r"]
    def action_labels(self):
        return ["w1 (left, CW)", "w2 (forward, CCW)", "w3 (right, CW)", "w4 (rear, CCW)"]
    def action_ranges(self):
        magnitude_lo = 0
        magnitude_hi = 6
        return np.array([
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
        ])
        
    def step(self, state, action):
        """
        This function works for when we get a single state shaped (12,) and a single action shaped (4,)
        """
        # Unwrap the state and action
        x, y, z, rx, ry, rz, vx, vy, vz, p, q, r = state
        w1, w2, w3, w4 = action

        # Compute sin, cos, and tans (xyz order is φ θ ψ)
        s_rx, c_rx, t_rx = math.sin(rx), math.cos(rx), math.tan(rx)
        s_ry, c_ry, t_ry = math.sin(ry), math.cos(ry), math.tan(ry)
        s_rz, c_rz, t_rz = math.sin(rz), math.cos(rz), math.tan(rz)

        # And now compute the control vector (control force, control torques)
        ft = self.thrust_coef * (w1**2 + w2**2 + w3**2 + w4**2)
        tx = self.thrust_coef * (self.diameter / 2) * (w3**2 - w1**2)
        ty = self.thrust_coef * (self.diameter / 2) * (w4**2 - w2**2)
        tz = self.drag_coef * ((w2**2 + w4**2) - (w1**2 + w3**2))

        # Can now compute the second derivatives (2.22)
        sdd = np.array([
            - ft / self.mass * (s_rx * s_rz  +  c_rx * c_rz * s_ry),
            - ft / self.mass * (c_rx * s_rz * s_ry  -  c_rz * s_rx),
            self.g - ft / self.mass * (c_rx * c_ry),
            (self.Iy - self.Iz) / self.Ix * q * r + tx / self.Ix,
            (self.Iz - self.Ix) / self.Iy * p * r + ty / self.Iy,
            (self.Ix - self.Iy) / self.Iz * p * q + tz / self.Iz,
        ])

        # Can now use the euler method to compute the first derivatives
        sd = np.array([vx, vy, vz, p, q, r]) + sdd * self.dt
        s  = np.array([x, y, z, rx, ry, rz]) + sd  * self.dt

        # Assemble the new state
        state_new = np.concatenate([s, sd])
        return state_new



