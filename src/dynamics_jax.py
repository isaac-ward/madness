import math
from scipy.spatial.transform import Rotation as R
import pickle
import os
import jax.numpy as jnp
import jax
from math import sqrt
from tqdm import tqdm

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
        self.continuous_dynamics = jax.jit(self.state_delta_batched)
        #self.discrete_dynamics = jax.jit(self._discrete_dynamics)
        self.discrete_dynamics = self._discrete_dynamics

    def __setstate__(self, state):
        """
        Method when unpickling. Remake jit class variables which aren't picklable
        """
        # Restore instance attributes
        self.__dict__.update(state)
        self._reload_dynamics()

    def _discrete_dynamics(self, state, action):
        return state + self.state_delta_batched(state, action) * self.dt
    
    def step(self, state, action):
        # Check to ensure that the number of states and actions are the same and that their 
        # dimensions are the same
        assert state.ndim == action.ndim, f"Expected states and actions to be same dimension, got {state.ndim} states and {action.ndim} actions"
        assert state.shape[-1] == self.state_size(), f"Expected state shape to be {self.state_size()}, got {state.shape[-1]}"
        assert action.shape[-1] == self.action_size(), f"Expected action shape to be {self.action_size()}, got {action.shape[-1]}"
        
        # Is this a batched input
        is_batched = state.ndim > 1
        if is_batched:
            assert state.shape[0] == action.shape[0], f"Expected same number of states and actions, got {state.shape[0]} states and {action.shape[0]} actions"

        # If the input is not batched then batch it
        if not is_batched:
            state = jnp.expand_dims(state, axis=0)
            action = jnp.expand_dims(action, axis=0)

        # Compute the change in state
        state_delta = self.state_delta_batched(state, action)
        # Compute the new state
        new_state = state + state_delta * self.dt

        # If the input was not batched then unbatch the output
        if not is_batched:
            new_state = new_state[0]
            # Check if quaternion is valid
            quaternion = new_state[3:7]
            print(f"quaternion: {quaternion}")

        return new_state


    def state_size(self):
        return 13
    def action_size(self):
        return 4
    def state_plot_groups(self):
        return [3, 4, 3, 3]
    def action_plot_groups(self):
        return [4]
    def state_labels(self):
        return ["x", "y", "z", "q0", "q1", "q2", "q3", "xd", "yd", "zd", "wx", "wy", "wz"]
    def state_randomization_template(self):
        return ["X","Y","Z",  1,0,0,0,  0,0,0,  0,0,0]
    def state_zero_with_quaternion_set_to_identity(self):
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    def action_labels(self):
        return ["w1 (left, CW)", "w4 (forward, CCW)", "w3 (right, CW)", "w2 (rear, CCW)"]
    def action_ranges(self):
        # If you're finding that state space isn't adequately explored,
        # consider increasing the size of the action space

        k = self.thrust_coef
        m = self.mass
        g = self.g
        w_trim = sqrt(m*g/(4*k))
        magnitude_lo = 0
        # magnitude_lo = -w_trim*0.95
        magnitude_hi = 4
        # magnitude_hi = w_trim*1.05
        return jnp.array([
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
        ]) 
    
    def state_delta_batched(self, batched_state, batched_action):
        """
        Batched version of the state_delta function to calculate the continuous nonlinear state
        derivative given a batch of states and actions.

        Parameters
        ----------
        batched_state: numpy.ndarray
            Batched state vector shaped (N, 13)
        batched_action: numpy.ndarray
            Batched control vector shaped (N, 4)
        
        Returns
        -------
        batched_state_delta: jax.numpy.ndarray
            Continuous state derivative vector at given state and action, shaped (N, 12)
        """
    
        def quaternion_multiply(q, r):
            """Multiplies two quaternions q and r."""
            q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            r0, r1, r2, r3 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
            
            # Quaternion multiplication formula
            t0 = q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3
            t1 = q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2
            t2 = q0 * r2 - q1 * r3 + q2 * r0 + q3 * r1
            t3 = q0 * r3 + q1 * r2 - q2 * r1 + q3 * r0
            return jnp.stack([t0, t1, t2, t3], axis=-1)
        
        def quaternion_shape_check(q0, q1, q2, q3):
            # Check if all components are the same shape
            assert q0.shape == q1.shape == q2.shape == q3.shape, f"Expected q0, q1, q2, q3 to be shaped ({num_items_in_batch}), got {q0.shape}, {q1.shape}, {q2.shape}, {q3.shape}"
            # Check if the first one is the right shape
            assert q0.shape == (num_items_in_batch,), f"Expected q0 to be shaped ({num_items_in_batch},), got {q0.shape}"
        def quaternion_zero_magnitude_check(q0, q1, q2, q3):
            qmag = jnp.linalg.norm(jnp.stack([q0, q1, q2, q3], axis=-1), axis=-1, keepdims=True).reshape(-1)
            # Quaternions cannot be zero magnitude
            zero_mag_quaternion_exists = jnp.any(jnp.isclose(qmag, 0))
            if zero_mag_quaternion_exists:
                for i in range(num_items_in_batch):
                    if jnp.isclose(qmag[i], 0):
                        print(f"i={i}, (q0,q1,q2,q3)=({q0[i]}, {q1[i]}, {q2[i]}, {q3[i]}) with magnitude {qmag[i]}")
            assert not zero_mag_quaternion_exists, "Quaternions have zero magnitude (nans may indicate a division by a zero magnitude)"
        def quaternion_normalization_check(q0, q1, q2, q3):
            qmag = jnp.linalg.norm(jnp.stack([q0, q1, q2, q3], axis=-1), axis=-1, keepdims=True).reshape(-1)
            # Quaternions should be normalized
            non_normalizard_quaternion_exists = jnp.any(jnp.logical_not(jnp.isclose(qmag, 1)))
            if non_normalizard_quaternion_exists:
                for i in range(num_items_in_batch):
                    if not jnp.isclose(qmag[i], 1):
                        print(f"i={i}, (q0,q1,q2,q3)=({q0[i]}, {q1[i]}, {q2[i]}, {q3[i]}) with magnitude {qmag[i]}")
            assert not non_normalizard_quaternion_exists, "Quaternions are not normalized to 1"

        # Check that we have the same number of states and actions
        num_states = batched_state.shape[0]
        num_actions = batched_action.shape[0]
        assert num_states == num_actions, f"Expected same number of states and actions, got {num_states} states and {num_actions} actions"
        num_items_in_batch = num_states
        assert batched_state.shape == (num_items_in_batch, self.state_size()), f"Expected state shape to be {self.state_size()}, got {batched_state.shape}"
        assert batched_action.shape == (num_items_in_batch, self.action_size()), f"Expected action shape to be {self.action_size()}, got {batched_action.shape}"

        # For convenience
        k = self.thrust_coef
        b = self.drag_yaw_coef
        kd = self.drag_force_coef

        # Unwrap the batched state and action 
        x, y, z        = batched_state[:, 0], batched_state[:, 1], batched_state[:, 2]
        q0, q1, q2, q3 = batched_state[:, 3], batched_state[:, 4], batched_state[:, 5], batched_state[:, 6]
        xd, yd, zd     = batched_state[:, 7], batched_state[:, 8], batched_state[:, 9]
        omx, omy, omz  = batched_state[:, 10], batched_state[:, 11], batched_state[:, 12]
        w1, w2, w3, w4 = batched_action[:, 0], batched_action[:, 1], batched_action[:, 2], batched_action[:, 3]

        # Ensure that quaternions aren't zero magnitude and are normalized
        quaternion_shape_check(q0, q1, q2, q3)
        quaternion_zero_magnitude_check(q0, q1, q2, q3)

        # Check for nans
        def nan_check(tensor, name):
            if jnp.any(jnp.isnan(tensor)):
                print(f"NaN found in {name}: {tensor}")
        for tensor, name in zip(
            [x, y, z, q0, q1, q2, q3, xd, yd, zd, omx, omy, omz, w1, w2, w3, w4], 
            ["x", "y", "z", "q0", "q1", "q2", "q3", "xd", "yd", "zd", "omx", "omy", "omz", "w1", "w2", "w3", "w4"]
        ):
            #print(f"Checking {name}, shape {tensor}")
            nan_check(tensor, name)

        # Normalize quaternions
        qmag = jnp.linalg.norm(jnp.stack([q0, q1, q2, q3], axis=-1), axis=-1, keepdims=True).reshape(-1)
        # We should have a magnitude for every quaternion
        q0, q1, q2, q3 = q0 / qmag, q1 / qmag, q2 / qmag, q3 / qmag
        # Check everything about the quaternions
        quaternion_shape_check(q0, q1, q2, q3)
        quaternion_zero_magnitude_check(q0, q1, q2, q3)
        quaternion_normalization_check(q0, q1, q2, q3)

        # Compute the control vector
        w1_sq, w2_sq, w3_sq, w4_sq = w1 ** 2, w2 ** 2, w3 ** 2, w4 ** 2
        r = self.diameter / 2
        ft = k * (w1_sq + w2_sq + w3_sq + w4_sq)
        tx = k * r * (w3_sq - w1_sq)
        ty = k * r * (w4_sq - w2_sq)
        tz = b * ((w2_sq + w4_sq) - (w1_sq + w3_sq))

        # Initialize batched_state_delta as zeros
        batched_state_delta = jnp.zeros_like(batched_state)

        # Positions change according to velocity
        batched_state_delta = batched_state_delta.at[:, 0].set(xd)
        batched_state_delta = batched_state_delta.at[:, 1].set(yd)
        batched_state_delta = batched_state_delta.at[:, 2].set(zd)

        # Quaternion change according to body rates
        quaternions = jnp.stack([q0, q1, q2, q3], axis=-1)
        angular_velocity_quaternions = jnp.stack([jnp.zeros_like(omx), omx, omy, omz], axis=-1)
        # Assert the shapes
        assert quaternions.shape == angular_velocity_quaternions.shape, f"Expected quaternions and angular_velocity_quaternions to be the same shape, got {quaternions.shape} and {angular_velocity_quaternions.shape}"
        qdot = quaternion_multiply(quaternions, angular_velocity_quaternions)
        batched_state_delta = batched_state_delta.at[:, 3].set(qdot[:, 0])
        batched_state_delta = batched_state_delta.at[:, 4].set(qdot[:, 1])
        batched_state_delta = batched_state_delta.at[:, 5].set(qdot[:, 2])
        batched_state_delta = batched_state_delta.at[:, 6].set(qdot[:, 3])

        # Velocities change according to forces and moments
        batched_state_delta = batched_state_delta.at[:, 7].set(-(ft / self.mass) * (2 * (q1 * q3 + q0 * q2)))
        batched_state_delta = batched_state_delta.at[:, 8].set(-(ft / self.mass) * (2 * (q2 * q3 - q0 * q1)))
        batched_state_delta = batched_state_delta.at[:, 9].set(self.g - (ft / self.mass) * (q0**2 - q1**2 - q2**2 + q3**2))

        # Body rates change according to moments of inertia and torques
        batched_state_delta = batched_state_delta.at[:, 10].set(((self.Iy - self.Iz) * omy * omz + tx) / self.Ix)
        batched_state_delta = batched_state_delta.at[:, 11].set(((self.Iz - self.Ix) * omx * omz + ty) / self.Iy)
        batched_state_delta = batched_state_delta.at[:, 12].set(((self.Ix - self.Iy) * omx * omy + tz) / self.Iz)
    
        return batched_state_delta
    
    def _state_delta(self, state, action):
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
        xd, yd, zd     = state[7],  state[8],  state[9]
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
        if states.ndim == 1:
            A,B = linearize_single(states, actions)
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