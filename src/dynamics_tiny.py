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

class DynamicsTiny:
    """
    Dynamics for a tiny gridworld
    """
    def __init__(
        self,
        dt,           
    ):
        self.dt = dt
    
    def step(self, state, action):
        return state + self.state_delta(state, action)

    def state_size(self):
        return 2
    
    def action_size(self):
        return 2
    
    def state_labels(self):
        # x, y
        return ["x", "y"]
    
    def zero_state(self):
        return jnp.zeros(self.state_size())
    
    def action_labels(self):
        return ["x jump", "y jump"]
    
    def action_ranges(self):
        return jnp.array([
            [-0.25, 0.25],
            [-0.25, 0.25],
        ]) 
    
    def state_delta(self, state, action):
        return action
    
    def linearize(self, states, actions):
        """
        Linearize the system dynamics (self.step) about the given state and action.
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
            A, B = jax.jacfwd(self.step, (0, 1))(state, action)
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
        Affinize the system dynamics (self.step) about the given state and action.
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
            A, B = jax.jacfwd(self.step, (0, 1))(state, action)
            C = self.step(state, action) - A@state - B@action
            return A, B, C
        
        # Affinize the batch of states and actions
        affinize_batch = jax.vmap(affinize_single, in_axes=(0, 0))
        if states.ndim == 1:
            A, B, C = affinize_single(states, actions)
        else:
            A, B, C = affinize_batch(states, actions)
        
        return A, B, C