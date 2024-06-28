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

    def _compute_second_derivatives(
        self,
        state,
        action,
    ):
        """
        Uses the equations of motion to compute the second derivatives of the state
        given the current state and action
        """

        # Unpack the state and action
        x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = state
        w1, w2, w3, w4 = action

        # Convert to euler angles
        phi, theta, psi = geometric.quaternion_to_euler_angles_rad(qx, qy, qz, qw)

        # Constants for convienience
        k = self.lift_coef
        b = self.thrust_coef
        d = self.drag_coef
        m = self.mass
        g = self.g
        Ixx = self.Ix
        Iyy = self.Iy
        Izz = self.Iz

        # The following are derived in the mathematica notebooks
        # Translational accelerations
        x_ddot = (-k * (w1**2 + w2**2 + w3**2 + w4**2) * (np.cos(theta) * np.cos(psi) * np.sin(phi) + np.sin(theta) * np.sin(psi)) + d * vx) / m
        y_ddot = (+k * (w1**2 + w2**2 + w3**2 + w4**2) * (np.cos(theta) * np.sin(phi) * np.sin(psi) - np.cos(psi) * np.sin(theta)) - d * vy) / m
        z_ddot = (-g * m + k * w1**2 * np.cos(theta) * np.cos(phi) + k * w2**2 * np.cos(theta) * np.cos(phi) + k * w3**2 * np.cos(theta) * np.cos(phi) + k * w4**2 * np.cos(theta) * np.cos(phi) - d * vz) / m

        # Rotational accelerations
        theta_ddot = (+k * self.diameter * (w2**2 - w4**2)) / Ixx
        phi_ddot   = (-k * self.diameter * (w1**2 - w3**2)) / Iyy
        psi_ddot   = (b * (w1**2 - w2**2 + w3**2 - w4**2)) / Izz

        return x_ddot, y_ddot, z_ddot, theta_ddot, phi_ddot, psi_ddot
    
    def _euler_method_propagate(
        self,
        state,
        action,
    ):
        # Unpack the state
        x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = state

        # We have the second derivatives, so we can use the Euler method to update the state
        # by first updating the velocities and then updating the positions
        second_derivatives = self._compute_second_derivatives(state, action)

        # Update the velocities as a vector
        new_first_derivatives = np.array([vx, vy, vz, wx, wy, wz]) + np.array(second_derivatives) * self.dt

        # Update the positions
        new_positions = np.array([x, y, z]) + new_first_derivatives[:3] * self.dt
        
        # When we get the delta in the zeroth derivates we need to convert
        # back to quaternions. That's because the delta that we've computed 
        # is a change in euler angles
        change_in_quaternion = geometric.euler_angles_rad_to_quaternion(*(new_first_derivatives[3:] * self.dt))
        initial_quaternion   = [qx, qy, qz, qw]
        new_quaternion       = geometric.quaternion_multiply(initial_quaternion, change_in_quaternion)
        
        # Now we can assemble the new state
        new_state = np.concatenate([
            new_positions, 
            new_quaternion, 
            new_first_derivatives
        ])

        return new_state  

    def step(
        self,
        state,          # current state
        action,         # control action
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
        """

        return self._euler_method_propagate(state, action)

    def save(self, folder):
        """
        Ensure the folder exists and then save oneself using pickle
        """
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "dynamics.pkl"), "wb") as f:
            pickle.dump(self, f)