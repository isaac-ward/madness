import math
from scipy.spatial.transform import Rotation as R
import pickle
import os
from numba import njit, prange, cuda, float32
import jax.numpy as jnp
import jax

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

        # Define continuous dynamics describing the state derivative
        self.continuous_dynamics = jax.jit(self.state_delta)

        # Define discrete dynamics describing next_state = dynamics(state, action)
        # Use Euler method for modeling
        self.discrete_dynamics = jax.jit(
            lambda state, action, dt=self.dt: state + dt * self.continuous_dynamics(state, action))

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
        # If you're finding that state space isn't adequately explored,
        # consider increasing the size of the action space
        magnitude_lo = 0
        magnitude_hi = 4
        return jnp.array([
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
        ]) 
    
    def state_delta(self, state, action):
        """
        Function to calculate the continuous nonlinear state derivative given a particular
        state and action. Usable with jax

        Parameters
        ----------
        state: numpy.ndarray
            State vector shaped (12,)
        action: numpy.ndarray
            Control vector shaped (4,)
        
        Returns
        -------
        state_delta: jax.numpy.ndarray
            Continuous state derivative vector at given state and action. Shaped (12,)
        """
        # For convenience
        k = self.thrust_coef
        b = self.drag_yaw_coef
        kd = self.drag_force_coef # TODO Why is this here? - Mark

        # Unwrap the state and action 
        # position, euler angles (xyz=>φθψ), velocity, body rates (eq2.23)
        x, y, z        = state[0],  state[1],  state[2]
        rz, ry, rx     = state[3],  state[4],  state[5]
        xd, yd, zd     = state[6],  state[7],  state[8]
        p, q, r        = state[9],  state[10], state[11]
        w1, w2, w3, w4 = action[0], action[1], action[2], action[3]
        # For convenience and to match with the KTH paper
        ψ, θ, φ = rz, ry, rx

        # Compute sin, cos, and tan (xyz order is φ θ ψ)
        s_ψ, c_ψ      = jnp.sin(ψ), jnp.cos(ψ)
        s_θ, c_θ, t_θ = jnp.sin(θ), jnp.cos(θ), jnp.tan(θ)
        s_φ, c_φ      = jnp.sin(φ), jnp.cos(φ)

        # Compute the control vector (control force, control torques), eq2.16
        w1_sq = w1 ** 2
        w2_sq = w2 ** 2
        w3_sq = w3 ** 2
        w4_sq = w4 ** 2
        r = self.diameter / 2
        # This is labeled as u1, u2, u3, u4 in the paper
        ft = k * (w1_sq + w2_sq + w3_sq + w4_sq)
        tx = k * r * (w3_sq - w1_sq)
        ty = k * r * (w4_sq - w2_sq)
        tz = b * ((w2_sq + w4_sq) - (w1_sq + w3_sq))

        # Compute the change in state (eq 2.23, 2.24, 2.25)
        state_delta = jnp.zeros_like(state)

        # Positions change according to velocity
        state_delta[0] = xd
        state_delta[1] = yd
        state_delta[2] = zd

        # Euler angles change according to body rates
        state_delta[3] = q * s_φ / c_θ + r * c_φ / c_θ
        state_delta[4] = q * c_φ - r * s_φ
        state_delta[5] = p + q * s_φ * t_θ + r * c_φ * t_θ

        # Velocities change according to forces and moments
        state_delta[6] =   - (ft / self.mass) * (s_ψ * s_φ  +  c_ψ * s_θ * c_φ)
        state_delta[7] =   - (ft / self.mass) * (c_ψ * s_φ  -  s_ψ * s_θ * c_φ)
        state_delta[8] = self.g - (ft / self.mass) * (c_θ * c_φ)

        # Body rates change according to moments of inertia and torques
        state_delta[9]  = ((self.Iy - self.Iz) * q * r + tx) / self.Ix
        state_delta[10] = ((self.Iz - self.Ix) * p * r + ty) / self.Iy
        state_delta[11] = ((self.Ix - self.Iy) * p * q + tz) / self.Iz
        
        return state_delta
    
    def linearize(self, state, action):
        """
        Linearize the system dynamics (self.discrete_dynamics) about the given state and action.
        System dynamics must be written in jax.

        Parameters
        ----------
        state: numpy.ndarray
            State vector to linearize about
        action: numpy.ndarray
            Control vector to linearize about

        Returns
        -------
        A: jax.numpy.ndarray
            Jacobian of dynamics function at provided (state, action) with respect to state
        B: jax.numpy.ndarray
            Jacobian of dynamics function at provided (state, action) with respect to action
        """
        # Calculate A and B 
        A, B = jax.jacfwd(self.discrete_dynamics, (0, 1))(state, action)

        return A, B
    
    def affinize(self, state, action):
        """
        Affinize the system dynamics (self.discrete_dynamics) about the given state and action.
        System dynamics must be written in jax.

        Parameters
        ----------
        state: numpy.ndarray
            State vector to affinize about
        action: numpy.ndarray
            Control vector to affinize about

        Returns
        -------
        A: jax.numpy.ndarray
            Jacobian of dynamics function at provided (state, action) with respect to state
        B: jax.numpy.ndarray
            Jacobian of dynamics function at provided (state, action) with respect to action
        C: jax.numpy.ndarry
            The offset term in the first-order Taylor expansion of dynamics function at 
            provided (state, action)
        """
        # Calculate A, B, and C
        A, B = jax.jacfwd(self.discrete_dynamics, (0, 1))(state, action)
        C = self.discrete_dynamics(state, action) - A@state - B@action

        return A, B, C