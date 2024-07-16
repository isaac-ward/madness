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
    - w1, w2, w3, w4 are the angular velocities of the rotors
    - the rotors are labeled clockwise from the left: 1,2,3,4
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
        return ["x", "y", "z", "rx", "ry", "rz", "vx", "vy", "vz", "p", "q", "r"]
    def action_labels(self):
        return ["w1 (left, CW)", "w2 (forward, CCW)", "w3 (right, CW)", "w4 (rear, CCW)"]
    def action_ranges(self):
        magnitude_lo = 0
        magnitude_hi = 2
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

        args = (state, action, self.diameter, self.mass, self.Ix, self.Iy, self.Iz, self.g, self.thrust_coef, self.drag_yaw_coef, self.drag_force_coef, self.dt)
        
        # Check if we have a batch
        batched = len(state.shape) == 2
        if batched:
            return step_batched(*args)
        else:
            return step_single(*args)
        
# ---------------------------------------------------------
        
@njit
def step_single(state, action, diameter, mass, Ix, Iy, Iz, g, thrust_coef, drag_yaw_coef, drag_force_coef, dt):
    """
    This function works for when we get a single state shaped (12,) and a single action shaped (4,).
    """
    # For convenience
    k = thrust_coef
    b = drag_yaw_coef
    kd = drag_force_coef

    # Unwrap the state and action 
    # position, euler angles (roll, pitch, yaw), velocity, angular velocity (eq2.23)
    x, y, z, φ, θ, ψ, xd, yd, zd, wx, wy, wz = state
    w1, w2, w3, w4 = action

    # Compute sin, cos, and tan (xyz order is φ θ ψ)
    s_φ, c_φ, t_φ = math.sin(φ), math.cos(φ), math.tan(φ)
    s_θ, c_θ, t_θ = math.sin(θ), math.cos(θ), math.tan(θ)
    s_ψ, c_ψ, t_ψ = math.sin(ψ), math.cos(ψ), math.tan(ψ)

    # The time derivatives of the angles are NOT the angular velocities
    # Angular velocity => vector pointing along axis of rotation, not the time derivative
    # angular_velocity = C * angle_time_derivative
    rate2w = np.array([
        [1, 0,   -s_θ],
        [0, +c_φ, s_φ * c_θ],
        [0, -s_φ, c_φ * c_θ],
    ])
    w2rate = np.linalg.inv(rate2w)

    # We need a matrix where a body frame vector v is converted
    # to the inertial frame via R * v
    rotation_matrix = np.array([
        [c_φ * c_ψ - c_θ * s_φ * s_ψ,       -c_ψ * s_φ - c_θ * c_φ * s_ψ,   +s_θ * s_ψ],
        [c_θ * s_φ * c_ψ + c_φ * s_ψ,       +c_φ * c_θ * c_ψ - s_φ * s_ψ,   -c_ψ * s_θ],
        [s_φ * s_θ,                         +c_φ * s_θ,                     +c_θ]
    ])

    # Compute the total thrust of the quadcopter 
    w1_sq = w1 ** 2
    w2_sq = w2 ** 2
    w3_sq = w3 ** 2
    w4_sq = w4 ** 2
    thrust_vector = np.array([
        0,
        0,
        k * (w1_sq + w2_sq + w3_sq + w4_sq)
    ])

    # Compute the drag forces
    drag_vector = np.array([
        - kd * xd,
        - kd * yd,
        - kd * zd,
    ])

    # Compute the gravity force
    gravity_vector = np.array([
        0,
        0,
        -g,
    ])

    # Compute the torques about each axis
    # φ, θ, ψ
    r = diameter / 2
    torque_vector = np.array([
        r * k * (w1_sq - w3_sq),
        r * k * (w4_sq - w2_sq),
        b * ((w1_sq + w3_sq) - (w2_sq + w4_sq)),
    ])
    
    # We can now compute the updated state
    # x, y, z, φ, θ, ψ, xd, yd, zd, wx, wy, wz
    # delta_xyz = np.array([xd, yd, zd])
    # delta_φθψ = gravity_vector + (1 / mass) * (rotation_matrix @ thrust_vector + drag_vector) 
    # delta_v = w2rate @ np.array([wx, wy, wz])
    # delta_w = torque_vector / np.array([Ix, Iy, Iz]) - np.array([
    #         (Iy - Iz) * wy * wz / Ix,
    #         (Iz - Ix) * wx * wz / Iy,
    #         (Ix - Iy) * wx * wy / Iz,
    # ])

    state_delta = np.array([
        xd,
        yd,
        zd,
        w2rate[0,0] * wx + w2rate[0,1] * wy + w2rate[0,2] * wz,
        w2rate[1,0] * wx + w2rate[1,1] * wy + w2rate[1,2] * wz,
        w2rate[2,0] * wx + w2rate[2,1] * wy + w2rate[2,2] * wz,
        0 + (1 / mass) * (rotation_matrix[0,0] * thrust_vector[0] + rotation_matrix[0,1] * thrust_vector[1] + rotation_matrix[0,2] * thrust_vector[2] + drag_vector[0]),
        0 + (1 / mass) * (rotation_matrix[1,0] * thrust_vector[0] + rotation_matrix[1,1] * thrust_vector[1] + rotation_matrix[1,2] * thrust_vector[2] + drag_vector[1]),
        0 + (1 / mass) * (rotation_matrix[2,0] * thrust_vector[0] + rotation_matrix[2,1] * thrust_vector[1] + rotation_matrix[2,2] * thrust_vector[2] + drag_vector[2]),
        torque_vector[0] / Ix - ((Iy - Iz) / Ix) * wy * wz,
        torque_vector[1] / Iy - ((Iz - Ix) / Iy) * wx * wz,
        torque_vector[2] / Iz - ((Ix - Iy) / Iz) * wx * wy
    ])

    # Use the Euler method to compute the new state
    state_new = state + dt * state_delta
    return state_new    

@njit(parallel=True)
def step_batched(states, actions, diameter, mass, Ix, Iy, Iz, g, thrust_coef, drag_yaw_coef, drag_force_coef, dt):
    B = states.shape[0]
    new_states = np.zeros((B, 12))
    for i in prange(B):
        new_states[i] = step_single(
            states[i], actions[i], diameter, mass, Ix, Iy, Iz, g, thrust_coef, drag_yaw_coef, drag_force_coef, dt
        )
    return new_states