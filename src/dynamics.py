import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import os

import utils.general as general
import utils.geometric as geometric

class DynamicsQuadcopter3D:
    """
    This class computes the dynamics for a 3D quadcopter

    Initialize with the constants, and then call the step function 
    with the current state and action to get the next state
    
    We'll use the following state representation:
    state = [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    Some important stuff:
    - qw is the scalar part of the quaternion
    - +x is forward, +y is left, +z is up
    - '+' quadcopter configuration
    - w1, w2, w3, w4 are the angular velocities of the rotors, looking down from above:
    -- w1 is along the +x axis (forward)
    -- w2 is along the +y axis (left)
    -- w3 is along the -x axis (backward)
    -- w4 is along the -y axis (right)
    - during hover, to prevent unwanted yawing, w1 and w3 should be CCW and w2 and w4 should be CW (or vice versa)
    """
    def __init__(
        self,
        diameter,        # in metres
        mass,            # in kg
        Ix,              # moment of inertia about x axis
        Iy,              # moment of inertia about y axis
        Iz,              # moment of inertia about z axis
        g,               # acceleration due to gravity
        lift_coef,       # lift coefficient (relates rotor angular velocity to lift force)
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
        self.lift_coef = lift_coef
        self.thrust_coef = thrust_coef
        self.drag_coef = drag_coef
        self.dt = dt

    def state_size(self):
        return 13
    def action_size(self):
        return 4
    def state_plot_groups(self):
        return [3, 4, 3, 3]
    def action_plot_groups(self):
        return [4]
    def state_labels(self):
        return ["x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz"]
    def action_labels(self):
        return ["w1", "w2", "w3", "w4"]
    def action_ranges(self):
        magnitude_lo = 0
        magnitude_hi = 6
        return np.array([
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
            [-magnitude_lo, +magnitude_hi],
        ])

    def _ensure_states_and_actions_valid_and_batched(self, states_batch, actions_batch):
        """
        Ensure that the states and actions are valid and batched
        """
        states_batch = np.array(states_batch)
        actions_batch = np.array(actions_batch)
        is_batch = not (states_batch.ndim == 1 and actions_batch.ndim == 1)
        if not is_batch:
            states_batch = np.array([states_batch])
            actions_batch = np.array([actions_batch])
        # Assert that the states and actions are the right shape
        assert states_batch.shape[1] == self.state_size(), f"states_batch.shape[1] = {states_batch.shape[1]}, self.state_size() = {self.state_size()}, must be equal"
        assert actions_batch.shape[1] == self.action_size(), f"actions_batch.shape[1] = {actions_batch.shape[1]}, self.action_size() = {self.action_size()}, must be equal"
        assert states_batch.shape[0] == actions_batch.shape[0], f"states_batch.shape[0] = {states_batch.shape[0]}, actions_batch.shape[0] = {actions_batch.shape[0]}, must be equal"
        return states_batch, actions_batch, is_batch

    def _compute_second_derivatives(
        self,
        states_batch,
        actions_batch,
    ):
        """
        Uses the equations of motion to compute the second derivatives of the state
        given a batch of corresponding states and actions

        states_batch is of the shape (N, state_size) 
        actions_batch is of the shape (N, action_size)

        and we return the second derivatives for each state in the batch in a vectorized way
        """

        states_batch, actions_batch, is_batch = self._ensure_states_and_actions_valid_and_batched(states_batch, actions_batch)

        # Constants for convienience
        k = self.lift_coef
        b = self.thrust_coef
        d = self.drag_coef
        m = self.mass
        g = self.g
        Ixx = self.Ix
        Iyy = self.Iy
        Izz = self.Iz

        # Unpack the state and action elements (each N length vectors)
        # Don't use np split here, the shape is known to be (N, state_size) and (N, action_size)
        x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = states_batch.T
        w1, w2, w3, w4 = actions_batch.T

        # Convert to euler angles as a batch operation
        phi, theta, psi = geometric.quaternion_to_euler_angles_rad(qx, qy, qz, qw).T

        # These will be repeatedly used in the operations
        # ** is elementwise in numpy
        w1_sq = w1**2
        w2_sq = w2**2
        w3_sq = w3**2
        w4_sq = w4**2
        # so is +
        w_sq_sum  = w1_sq + w2_sq + w3_sq + w4_sq
        # and trig functions
        cos_theta = np.cos(theta)
        cos_phi   = np.cos(phi)
        cos_psi   = np.cos(psi)
        sin_theta = np.sin(theta)
        sin_phi   = np.sin(phi)
        sin_psi   = np.sin(psi)

        # The following are derived in the mathematica notebooks
        # Translation accelerations
        x_ddot = (-k * w_sq_sum * (cos_theta * cos_psi * sin_phi + sin_theta * sin_psi) + d * vx) / m
        y_ddot = (+k * w_sq_sum * (cos_theta * sin_phi * sin_psi - cos_psi * sin_theta) - d * vy) / m
        z_ddot = (k * w_sq_sum * cos_theta * cos_phi - d * vz) / m - g

        # Rotational accelerations
        theta_ddot = (+k * self.diameter * (w2_sq - w4_sq)) / Ixx
        phi_ddot   = (-k * self.diameter * (w1_sq - w3_sq)) / Iyy
        psi_ddot   = (b * ((w1_sq + w3_sq) - (w2_sq + w4_sq))) / Izz

        second_derivatives = np.column_stack([x_ddot, y_ddot, z_ddot, theta_ddot, phi_ddot, psi_ddot])

        if is_batch:
            return second_derivatives
        else:
            return second_derivatives[0]

    def _euler_method_propagate(
        self,
        states_batch,
        actions_batch,
    ):
        """
        Given a batch of states and actions, compute the next states
        using the Euler method
        """

        states_batch, actions_batch, is_batch = self._ensure_states_and_actions_valid_and_batched(states_batch, actions_batch)

        # Shapes are (N, state_size) and (N, action_size)
        states_batch = np.array(states_batch)
        actions_batch = np.array(actions_batch)

        # Unpack the states
        x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = states_batch.T

        # We have the second derivatives, so we can use the Euler method to update the state
        # by first updating the velocities and then updating the positions
        second_derivatives = self._compute_second_derivatives(states_batch, actions_batch)

        # Update the velocities as a vector
        new_first_derivatives = np.column_stack([vx, vy, vz, wx, wy, wz]) + np.array(second_derivatives) * self.dt

        # Update the positions
        new_positions = np.column_stack([x, y, z]) + new_first_derivatives[:, :3] * self.dt
        
        # When we get the delta in the zeroth derivates we need to convert
        # back to quaternions. That's because the delta that we've computed 
        # is a change in euler angles
        change_in_quaternion = geometric.euler_angles_rad_to_quaternion(*(new_first_derivatives[:, 3:] * self.dt).T)
        initial_quaternion   = np.column_stack([qx, qy, qz, qw])
        new_quaternion       = geometric.quaternion_multiply(initial_quaternion, change_in_quaternion)
        
        # Now we can assemble the new state
        new_states = np.column_stack([
            new_positions, 
            new_quaternion, 
            new_first_derivatives
        ])

        if is_batch:
            return new_states
        else:
            return new_states[0]

    def step(
        self,
        states_batch,          # current state
        actions_batch,         # control action
    ):
        """
        Given a state and a control action, compute the next state

        We'll use the Euler method (or RK4 - TODO), so the state needs to contain the
        generalised coordinates of the system and the velocities, the 
        generalised coordinates in this case being x, y, z, phi, theta, psi

        In actuality, we represent the rotations in the state as a quaternion, 
        but otherwise the state is as described above

        We use the equations of motion to compute the acceleration over some 
        time step, and then use that to update the velocities, and then use
        that to update the positions

        Small time steps are better for accuracy

        Designed to work in batches too
        """

        states_batch, actions_batch, is_batch = self._ensure_states_and_actions_valid_and_batched(states_batch, actions_batch)

        # Compute the next state using the Euler method
        new_states_batch = self._euler_method_propagate(states_batch, actions_batch)

        if is_batch:
            return new_states_batch
        else:
            return new_states_batch[0]


