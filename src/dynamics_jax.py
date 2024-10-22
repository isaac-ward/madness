import math
from scipy.spatial.transform import Rotation as R
import pickle
import os
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
    
    def __getstate__(self):
        """
        Method when pickling. Exclude jit class variables which aren't picklable
        """
        # Get the object's __dict__ and make a copy
        state = self.__dict__.copy()
        
        # Remove the attribute you don't want to pickle
        if 'continuous_dynamics' in state:
            del state['continuous_dynamics']
        if 'discrete_dynamics' in state:
            del state['discrete_dynamics']

        return state

    def __setstate__(self, state):
        """
        Method when unpickling. Remake jit class variables which aren't picklable
        """
        # Restore instance attributes
        self.__dict__.update(state)

        # Reinitialize the excluded variables
        # Define continuous dynamics describing the state derivative
        self.continuous_dynamics = jax.jit(self.state_delta)

        # Define discrete dynamics describing next_state = dynamics(state, action)
        # Use Euler method for modeling
        self.discrete_dynamics = jax.jit(
            lambda state, action, dt=self.dt: state + dt * self.continuous_dynamics(state, action))

    def state_size(self):
        return 13
    def action_size(self):
        return 4
    def state_plot_groups(self):
        return [3, 4, 3, 3]
    def action_plot_groups(self):
        return [4]
    def state_labels(self):
        # x, y, z, q0, q1, q2, q3, xd, yd, zd, wx, wy, wz
        return ["x", "y", "z", "q0", "q1", "q2", "q3" "xd", "yd", "zd", "wx", "wy", "wz"]
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
    
    def step(self, state, action):
        return state + self.state_delta(state, action)
    
    def state_delta(self, state, action):
        """
        Function to calculate the continuous nonlinear state derivative given a particular
        state and action. Usable with jax

        Parameters
        ----------
        state: numpy.ndarray
            State vector shaped (13,)
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
        # position, quaternions, velocity, body rates (eq2.23)
        x, y, z        = state[0],  state[1],  state[2]
        q0, q1, q2, q3 = state[3],  state[4],  state[5], state[6]
        xd, yd, zd     = state[7], state[8], state[9]
        # p, q, r        = state[9],  state[10], state[11]
        omx,omy,omz    = state[10],  state[11], state[12]
        w1, w2, w3, w4 = action[0], action[1], action[2], action[3]
        # Normalize the quaternion
        qmag = jnp.linalg.norm(jnp.array([q0, q1, q2, q3]))
        q0 /= qmag
        q1 /= qmag
        q2 /= qmag
        q3 /= qmag

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
        state_delta = state_delta.at[0].set(xd)
        state_delta = state_delta.at[1].set(yd)
        state_delta = state_delta.at[2].set(zd)

        # Quaternion change according to body rates
        qdot = 0.5 * geometric.q_mul(jnp.array([0, omx, omy, omz]), jnp.array([q0, q1, q2, q3]))
        state_delta = state_delta.at[3].set(qdot[0])
        state_delta = state_delta.at[4].set(qdot[1])
        state_delta = state_delta.at[5].set(qdot[2])
        state_delta = state_delta.at[6].set(qdot[3])

        # Velocities change according to forces and moments
        state_delta = state_delta.at[7].set(-(ft / self.mass) * (2*(q1*q3 + q0*q2)))
        state_delta = state_delta.at[8].set(-(ft / self.mass) * (2*(q2*q3 - q0*q1)))
        state_delta = state_delta.at[9].set(self.g - (ft / self.mass) * (q0**2 - q1**2 - q2**2 + q3**2))

        # Body rates change according to moments of inertia and torques
        state_delta = state_delta.at[10].set(((self.Iy - self.Iz) * omy * omz + tx) / self.Ix)
        state_delta = state_delta.at[11].set(((self.Iz - self.Ix) * omx * omz + ty) / self.Iy)
        state_delta = state_delta.at[12].set(((self.Ix - self.Iy) * omx * omy + tz) / self.Iz)
        
        return state_delta
    
    def linearize(self, states, actions):
        """
        Linearize the system dynamics (self.discrete_dynamics) about the given state and action.
        System dynamics must be written in jax. Works with batch inputs

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
        def linearize_single(state, action):
            # Calculate A and B 
            A, B = jax.jacfwd(self.discrete_dynamics, (0, 1))(state, action)
            return A, B
        
        # Linearize the batch of states and actions
        linearize_batch = jax.vmap(linearize_single, in_axes=(0, 0))
        A, B = linearize_batch(states, actions)
        
        return A, B
    
    def affinize(self, states, actions):
        """
        Affinize the system dynamics (self.discrete_dynamics) about the given state and action.
        System dynamics must be written in jax. Works with batch inputs

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
        C: jax.numpy.ndarray
            The offset term in the first-order Taylor expansion of dynamics function at 
            provided (state, action)
        """
        def affinize_single(state, action):
            # Calculate A, B, and C
            A, B = jax.jacfwd(self.discrete_dynamics, (0, 1))(state, action)
            C = self.discrete_dynamics(state, action) - A@state - B@action
            return A, B, C
        
        # Affinize the batch of states and actions
        affinize_batch = jax.vmap(affinize_single, in_axes=(0, 0))
        A, B, C = affinize_batch(states, actions)
        
        return A, B, C