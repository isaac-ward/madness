import math
from scipy.spatial.transform import Rotation as R
import pickle
import os
import jax.numpy as jnp
import jax
import scipy as sp
from math import sqrt

import utils.general as general
import utils.geometric as geometric

class DynamicsLinear3D:
    """
    This class computes the dynamics for a 3D point for linear testing

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
        self._reload_dynamics()
    
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
    
    def _reload_dynamics(self):

        # Reinitialize the excluded variables
        # Define continuous dynamics describing the state derivative
        self.continuous_dynamics = jax.jit(self.state_delta)
        #self.discrete_dynamics = jax.jit(self._discrete_dynamics)
        self.discrete_dynamics = self._discrete_dynamics

    def __setstate__(self, state):
        """
        Method when unpickling. Remake jit class variables which aren't picklable
        """
        # Restore instance attributes
        self.__dict__.update(state)
        self._reload_dynamics()

    def _discrete_dynamics(self, state, action, Q = None, seed = 228):

        # If the state and action is batched then we need to handle the delta
        # computation is a batch
        if state.ndim == 2 and action.ndim == 2:
            print(action)
            state_delta = jax.vmap(self.state_delta, in_axes=(0, 0))(state, action)
        elif state.ndim == 1 and action.ndim == 1:
            state_delta = self.state_delta(state, action)
        else:
            raise ValueError(f"State and action must have the same number of dimensions. Got state dimension {state.ndim} and action dimension {action.ndim}")

        if not(Q is None):
            key = jax.random.key(seed)
            noise = sp.linalg.sqrtm(Q) @ jax.random.normal(key, state_delta.shape).T
            state_delta += noise.T

        change_in_state = state_delta * self.dt

        def _print_helper(label, tracer):
            if False:
                tracer = jax.block_until_ready(tracer)
                jax.debug.print(f"{label}: {tracer}")

        _print_helper("change_in_state", change_in_state)
        
        # Now we can assemble
        new_state = state + change_in_state

        _print_helper("new_state", new_state)

        return new_state
    
    def step(self, state, action, Q = None, seed = 228):
        return self.discrete_dynamics(state, action, Q, seed)

    def state_size(self):
        return 12
    
    def state_randomization_template(self):
        return ["X", "Y", "Z", 0, 0, 0,   0, 0, 0,   0, 0, 0]
    
    def action_size(self):
        return 4
    
    def state_plot_groups(self):
        return [3, 3, 3, 3]
    
    def action_plot_groups(self):
        return [4]
    
    def state_labels(self):
        # x, y, z, φ, θ, ψ, xd, yd, zd, wx, wy, wz
        return ["x", "y", "z", "rz", "ry", "rx", "xd", "yd", "zd", "wx", "wy", "wz"]
    
    def zero_state(self):
        return jnp.zeros(self.state_size())
    
    def action_labels(self):
        return ["w1 (left, CW)", "w4 (forward, CCW)", "w3 (right, CW)", "w2 (rear, CCW)"]
    
    def action_ranges(self):
        # If you're finding that state space isn't adequately explored,
        # consider increasing the size of the action space
        magnitude_lo = -0.001
        magnitude_hi = +0.001
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

        # Assert that the shapes of the states and actions are correct
        assert state.shape == (12,), f"State shape is {state.shape}, should be (12,)"
        assert action.shape == (4,), f"Action shape is {action.shape}, should be (4,)"
        
        # the first three actions control the position
        x, y, z        = state[0],  state[1],  state[2]
        w1, w2, w3     = action[0], action[1], action[2]
        
        # Compute the change in state
        state_delta = jnp.zeros_like(state)
        # Positions change according to velocity
        # Can move 0.004 per dt (recall we test at dt = 0.025 or 40 Hz)
        state_delta = state_delta.at[0].set(x + w1) 
        state_delta = state_delta.at[1].set(y + w2)
        state_delta = state_delta.at[2].set(z + w3)
        # Euler angles change according to body rates
        state_delta = state_delta.at[3].set(0)
        state_delta = state_delta.at[4].set(0)
        state_delta = state_delta.at[5].set(0)
        state_delta = state_delta.at[6].set(0)
        state_delta = state_delta.at[7].set(0)
        state_delta = state_delta.at[8].set(0)
        state_delta = state_delta.at[9].set(0)
        state_delta = state_delta.at[10].set(0)
        state_delta = state_delta.at[11].set(0)
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
        if states.ndim == 1:
            A,B = linearize_single(states,actions)
        else:
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
        if states.ndim == 1:
            A, B, C = affinize_single(states,actions)
        else:
            A, B, C = affinize_batch(states, actions)
        
        return A, B, C