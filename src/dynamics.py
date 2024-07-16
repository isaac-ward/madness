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
    - +x is forward, +y is right, +z is down (NED)
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
        thrust_coef,     # thrust coefficient 
        drag_coef,       # drag coefficient 
        dt,              # time step
    ):
        self.diameter = diameter
        self.mass = mass
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.g = g
        self.thrust_coef = thrust_coef
        self.drag_coef = drag_coef
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
        magnitude_hi = 1
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
        
        # Check if we have a batch
        batched = len(state.shape) == 2
        if batched:
            return step_batched(state, action, self.diameter, self.mass, self.Ix, self.Iy, self.Iz, self.g, self.thrust_coef, self.drag_coef, self.dt)
        else:
            return step_single(state, action, self.diameter, self.mass, self.Ix, self.Iy, self.Iz, self.g, self.thrust_coef, self.drag_coef, self.dt)
        
# ---------------------------------------------------------
        
@njit
def step_single(state, action, diameter, mass, Ix, Iy, Iz, g, thrust_coef, drag_coef, dt):
    """
    This function works for when we get a single state shaped (12,) and a single action shaped (4,).
    """
    # Unwrap the state and action 
    # position, euler angles, velocity, body rates (eq2.23)
    x, y, z, ψ, θ, φ, xd, yd, zd, p, q, r = state
    w1, w2, w3, w4 = action

    # Compute sin, cos, and tan (xyz order is φ θ ψ)
    s_ψ, c_ψ, t_ψ = math.sin(ψ), math.cos(ψ), math.tan(ψ)
    s_θ, c_θ, t_θ = math.sin(θ), math.cos(θ), math.tan(θ)
    s_φ, c_φ, t_φ = math.sin(φ), math.cos(φ), math.tan(φ)

    # Compute the control vector (control force, control torques), eq2.16
    w1_sq = w1 ** 2
    w2_sq = w2 ** 2
    w3_sq = w3 ** 2
    w4_sq = w4 ** 2
    r = diameter / 2
    # This is labeled as u1, u2, u3, u4 in the paper
    ft = thrust_coef * (w1_sq + w2_sq + w3_sq + w4_sq)
    tx = thrust_coef * r * (w3_sq - w1_sq)
    ty = thrust_coef * r * (w4_sq - w2_sq)
    tz = drag_coef * ((w2_sq + w4_sq) - (w1_sq + w3_sq))

    # Compute the change in state (eq 2.23, 2.24, 2.25)
    state_delta = np.zeros(12)

    # Positions change according to velocity
    state_delta[0] = xd
    state_delta[1] = yd
    state_delta[2] = zd

    # # Euler angles change according to body rates
    state_delta[3] = q * s_φ / c_θ + r * c_φ / c_θ
    state_delta[4] = q * c_φ - r * s_φ
    state_delta[5] = p + q * s_φ * t_θ + r * c_φ * t_θ

    # Velocities change according to forces and moments
    state_delta[6] =   - (ft / mass) * (s_ψ * s_φ  +  c_ψ * s_θ * c_φ)
    state_delta[7] =   - (ft / mass) * (c_ψ * s_φ  -  s_ψ * s_θ * c_φ)
    state_delta[8] = g - (ft / mass) * (c_θ * c_φ)

    # Body rates change according to moments of inertia and torques
    state_delta[9]  = ((Iy - Iz) * q * r + tx) / Ix
    state_delta[10] = ((Iz - Ix) * p * r + ty) / Iy
    state_delta[11] = ((Ix - Iy) * p * q + tz) / Iz

    # Check for nans
    if np.any(np.isnan(state_delta)):
        print(f"NaNs in state_delta: {state_delta}, state: {state}, action: {action}")
        raise ValueError(f"NaNs in state_delta: {state_delta}, state: {state}, action: {action}")

    # Use the Euler method to compute the new state
    state_new = state + dt * state_delta
    return state_new    

    # # Compute the second derivatives (2.22)
    # sdd = np.zeros(6)
    # sdd[0] =   - ft / mass * (+ c_rx * s_ry * c_rz + s_rx * s_rz)
    # sdd[1] =   - ft / mass * (+ c_rx * s_ry * s_rz - s_rx * c_rz)
    # sdd[2] = g - ft / mass * (c_rx * c_ry)
    # sdd[3] = ((Iy - Iz) * q * r + tx) / Ix
    # sdd[4] = ((Iz - Ix) * p * r + ty) / Iy
    # sdd[5] = ((Ix - Iy) * p * q + tz) / Iz

    # # Use the Euler method to compute the first derivatives
    # sd = np.array([vx, vy, vz, p, q, r]) + dt * sdd
    # s  = np.array([x, y, z, rx, ry, rz]) + dt * sd

    # # Assemble the new state
    # state_new = np.zeros(12)
    # state_new[:6] = s
    # state_new[6:] = sd
    # return state_new

@njit(parallel=True)
def step_batched(states, actions, diameter, mass, Ix, Iy, Iz, g, thrust_coef, drag_coef, dt):
    B = states.shape[0]
    new_states = np.zeros((B, 12))
    for i in prange(B):
        new_states[i] = step_single(
            states[i], actions[i], diameter, mass, Ix, Iy, Iz, g, thrust_coef, drag_coef, dt
        )
    return new_states