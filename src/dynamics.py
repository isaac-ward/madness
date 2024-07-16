import math
import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation as R
import pickle
import os
from numba import njit, prange, cuda, float32

import utils.general as general
import utils.geometric as geometric

class DynamicsQuadcopter3D:
    """
    This class computes the dynamics for a 3D quadcopter

    Implements the model described in:
    https://andrew.gibiansky.com/downloads/pdf/Quadcopter%20Dynamics,%20Simulation,%20and%20Control.pdf

    It's also worth looking at the state space model from page 18, section 2.6
    https://www.kth.se/polopoly_fs/1.588039.1600688317!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf

    Initialize with the constants, and then call the step function 
    with the current state and action to get the next state
    
    We'll use the following state representation:
    state = [x, y, z, rx, ry, rz, vx, vy, vz, p, q, r]
    Some important stuff:
    - +x is forward, +y is right, +z is down (NED)
    - tx is large when w3 is large (left)
    - ty is large when w4 is large (forward)
    - see action labels for correct labeling
    """
    def __init__(
        self,
        diameter,        # in metres
        mass,            # in kg
        Ix,              # moment of inertia about x axis
        Iy,              # moment of inertia about y axis
        Iz,              # moment of inertia about z axis
        g,               # acceleration due to gravity (negative if down)
        thrust_coef,     # thrust coefficient 
        drag_yaw_coef,   # drag coefficient controlling drag due to yaw
        drag_force_coef, # drag coefficient controlling drag force due to velocity
        dt,              # time step
    ):
        self.diameter = diameter
        self.mass = mass
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.g = g
        self.thrust_coef = thrust_coef
        self.drag_yaw_coef = drag_yaw_coef
        self.drag_force_coef = drag_force_coef
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
        # x, y, z, φ, θ, ψ, xd, yd, zd, wx, wy, wz
        return ["x", "y", "z", "rz", "ry", "rx", "xd", "yd", "zd", "wx", "wy", "wz"]
    def action_labels(self):
        return ["w1 (left, CW)", "w4 (forward, CCW)", "w3 (right, CW)", "w2 (rear, CCW)"]
    def action_ranges(self):
        magnitude_lo = 0
        magnitude_hi = 0.5
        return np.array([
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
        ]) 
    
    def step(self, state, action):
        """
        This function works for both single state shaped (12,) and action shaped (4,)
        and batched states shaped (B, 12) and batched actions shaped (B, 4)
        """

        state = np.asarray(state)
        action = np.asarray(action)
        
        # If not batched then wrap in a batch
        is_batch = len(state.shape) == 2
        if not is_batch:
            state = state[None, :]
            action = action[None, :]

        # If GPUs are available use them
        if cp.cuda.is_available():
            state = cp.asarray(state)
            action = cp.asarray(action)

        # Call step
        new_states = step_batch_gpu(
            state, action,
            self.diameter, self.mass, self.Ix, self.Iy, self.Iz, self.g, 
            self.thrust_coef, self.drag_yaw_coef, self.drag_force_coef, self.dt
        )

        # Unbatch if needed
        if not is_batch:
            new_states = new_states[0]

        # If we used GPU compute, then move back to CPU
        if cp.cuda.is_available():
            new_states = new_states.get()
        return new_states
        
# --------------------------------------------------------- 

def step_batch_gpu(states, actions, diameter, mass, Ix, Iy, Iz, g, thrust_coef, drag_yaw_coef, drag_force_coef, dt):
    """
    This function works for batched states shaped (B, 12) and batched actions shaped (B, 4).
    """

    # Do we have GPU access?
    xp = cp.get_array_module(states)
    
    # For convenience
    k = thrust_coef
    b = drag_yaw_coef
    kd = drag_force_coef

    # Unwrap the state and action 
    # position, euler angles (xyz=>φθψ), velocity, body rates (eq2.23)
    x, y, z        = states[:, 0],  states[:, 1],  states[:, 2]
    rz, ry, rx     = states[:, 3],  states[:, 4],  states[:, 5]
    xd, yd, zd     = states[:, 6],  states[:, 7],  states[:, 8]
    p, q, r        = states[:, 9],  states[:, 10], states[:, 11]
    w1, w2, w3, w4 = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3]
    # For convenience and to match with the KTH paper
    ψ, θ, φ = rz, ry, rx

    # Compute sin, cos, and tan (xyz order is φ θ ψ)
    s_ψ, c_ψ      = xp.sin(ψ), xp.cos(ψ)
    s_θ, c_θ, t_θ = xp.sin(θ), xp.cos(θ), xp.tan(θ)
    s_φ, c_φ      = xp.sin(φ), xp.cos(φ)

    # Compute the control vector (control force, control torques), eq2.16
    w1_sq = w1 ** 2
    w2_sq = w2 ** 2
    w3_sq = w3 ** 2
    w4_sq = w4 ** 2
    r = diameter / 2
    # This is labeled as u1, u2, u3, u4 in the paper
    ft = k * (w1_sq + w2_sq + w3_sq + w4_sq)
    tx = k * r * (w3_sq - w1_sq)
    ty = k * r * (w4_sq - w2_sq)
    tz = b * ((w2_sq + w4_sq) - (w1_sq + w3_sq))

    # Compute the change in state (eq 2.23, 2.24, 2.25)
    state_delta = xp.zeros_like(states)

    # Positions change according to velocity
    state_delta[:, 0] = xd
    state_delta[:, 1] = yd
    state_delta[:, 2] = zd

    # Euler angles change according to body rates
    state_delta[:, 3] = q * s_φ / c_θ + r * c_φ / c_θ
    state_delta[:, 4] = q * c_φ - r * s_φ
    state_delta[:, 5] = p + q * s_φ * t_θ + r * c_φ * t_θ

    # Velocities change according to forces and moments
    state_delta[:, 6] =   - (ft / mass) * (s_ψ * s_φ  +  c_ψ * s_θ * c_φ)
    state_delta[:, 7] =   - (ft / mass) * (c_ψ * s_φ  -  s_ψ * s_θ * c_φ)
    state_delta[:, 8] = g - (ft / mass) * (c_θ * c_φ)

    # Body rates change according to moments of inertia and torques
    state_delta[:, 9]  = ((Iy - Iz) * q * r + tx) / Ix
    state_delta[:, 10] = ((Iz - Ix) * p * r + ty) / Iy
    state_delta[:, 11] = ((Ix - Iy) * p * q + tz) / Iz
    
    # Use the Euler method to compute the new state
    states_new = states + dt * state_delta
    return states_new