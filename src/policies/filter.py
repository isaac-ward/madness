import numpy as np
import scipy as sp
# import control
from scipy.integrate import odeint 

# from pose_estimation.dynamics.dynamics_rot import *
# from pose_estimation.meas_gen_utils import *

from dynamics_jax import DynamicsQuadcopter3D

# def ssDef(x, dt):
#     A = None
#     B = None
#     C = None
#     Q = None
#     R = None

#     return A, B, C, Q, R

class ObservationModel:
    """
    Represents the observation model of the system.
    
    Attributes:
        h (function): Nonlinear measurement function.
        C (numpy.ndarray): Observation matrix (Jacobian of h).
    """
    def __init__(self, 
                 h = None,
                 C = None):
        """
        Initialize the observation model.

        Args:
            h (function, optional): Nonlinear measurement function.
            C (numpy.ndarray, optional): Observation matrix.
        """
        self.h = h
        self.C = C

class Filter:
    """
    Base class for state estimation filters.
    
    Attributes:
        mu (numpy.ndarray): Mean of the state estimate.
        Sig (numpy.ndarray): Covariance of the state estimate.
        Q (numpy.ndarray): Process noise covariance.
        R (numpy.ndarray): Measurement noise covariance.
        obs (ObservationModel): Observation model of the system.
        dyn (DynamicsQuadcopter3D): Dynamics model of the system.
        dt (float): Time step for dynamics propagation.
        rng_seed (int): Random seed for reproducibility.
    """
    def __init__(self, mu0, Sig0, Q, R, 
                 obs: ObservationModel,
                 dyn: DynamicsQuadcopter3D, 
                 rng_seed = 273):
        
        """
        Initialize the filter.

        Args:
            mu0 (numpy.ndarray): Initial state estimate.
            Sig0 (numpy.ndarray): Initial covariance estimate.
            Q (numpy.ndarray): Process noise covariance.
            R (numpy.ndarray): Measurement noise covariance.
            obs (ObservationModel): Observation model.
            dyn (DynamicsQuadcopter3D): Dynamics model.
            rng_seed (int, optional): Random seed for reproducibility. Default is 273.
        """
        
        self.mu = mu0
        self.Sig = Sig0
        self.Q = Q
        self.R = R
        self.obs = obs
        self.dyn = dyn
        self.dt = dyn.dt
        self.rng_seed = rng_seed

class EKF(Filter):
    """
    Extended Kalman Filter (EKF) for state estimation.

    Inherits from the base Filter class and implements the EKF-specific 
    predict, update, and step methods.
    """
    def __init__(self, mu0, Sig0, Q, R, 
                 obs: ObservationModel,
                 dyn: DynamicsQuadcopter3D, 
                 rng_seed=273):
        """
        Initialize the EKF.

        Args:
            mu0 (numpy.ndarray): Initial state estimate.
            Sig0 (numpy.ndarray): Initial covariance estimate.
            Q (numpy.ndarray): Process noise covariance.
            R (numpy.ndarray): Measurement noise covariance.
            obs (ObservationModel): Observation model.
            dyn (DynamicsQuadcopter3D): Dynamics model.
            rng_seed (int, optional): Random seed for reproducibility. Default is 273.
        """
        super().__init__(mu0, Sig0, Q, R, obs, dyn, rng_seed)

    def predict(self, u):
        """
        Perform the EKF prediction step.

        Args:
            u (numpy.ndarray): Control input.

        Returns:
            tuple: Predicted state mean and covariance.
        """
        A, _, _ = self.dyn.affinize(self.mu, u)
        A = np.array(A)

        # Propagate the state mean using the nonlinear dynamics.
        mu_plus = self.dyn.step(self.mu, u)
        # Propagate the state covariance using the linearized dynamics.
        Sig_plus = A @ self.Sig @ A.T + self.Q
        return mu_plus, Sig_plus

    def update(self, mu_plus, Sig_plus, ys):
        """
        Perform the EKF update step.

        Args:
            mu_plus (numpy.ndarray): Predicted state mean.
            Sig_plus (numpy.ndarray): Predicted state covariance.
            ys (numpy.ndarray): Measurement vector.

        Returns:
            tuple: Updated state mean and covariance.
        """

        # Compute the Kalman gain.
        K = Sig_plus @ self.obs.C.T @ np.linalg.inv(self.obs.C @ Sig_plus @ self.obs.C.T + self.R)

        # Compute the measurement prediction and residual.
        ym = self.obs.h(mu_plus)
        mu_plus_plus = mu_plus + K @ (ys - ym)

        # Update the covariance estimate.
        Sig_plus_plus = Sig_plus - K @ self.obs.C @ Sig_plus
        return mu_plus_plus, Sig_plus_plus
    
    def step(self, u, y):
        """
        Perform a single EKF step (predict + update).

        Args:
            u (numpy.ndarray): Control input.
            y (numpy.ndarray): Measurement vector.

        Returns:
            tuple: Updated state mean and covariance.
        """

        # Predict step.
        mu_tplus_t, Sig_tplus_t = self.predict(u)
        # Update step.
        mu_tplus_tplus, Sig_tplus_tplus = self.update(mu_tplus_t, Sig_tplus_t, y)

        # Store the updated state and covariance.
        self.mu = mu_tplus_tplus
        self.Sig = Sig_tplus_tplus
        return mu_tplus_tplus, Sig_tplus_tplus


# class MEKF(Filter):
#     def __init__(self, mu0, Sig0, Q, R, qref, 
#                  stateUpdate = linStateUpdate,
#                  measFunc = linMeasUpdate,
#                  ssMatfunc = ssDef,
#                  dt = 1,
#                  rng_seed = 273):
#         super().__init__(mu0, Sig0, Q, R, stateUpdate, measFunc, ssMatfunc, dt, rng_seed)
#         self.qref = qref

#     def step(self, u, y, I, qtol = 1e-4):
        
#         #### predict step ####

#         # nonlinear quat prop
#         qw = np.concatenate([self.qref, self.mu[3:]])
#         # print("q = ", self.qref)
#         # print("w = ", self.mu[3:])
#         qw = odeint(ode_qw, qw, [0,self.dt], args=(I, np.zeros((3,1))))[1]
#         q_tplus_t = qw[:4]

#         # linear state mean and cov prop
#         Phi, B, C = mekf_stm(self.mu, I, self.dt) 
#         ya = np.zeros((9,))
#         ya[3:] = y[4:]
#         if np.all(y[:4] == 0):
#             # print("changing C mat")
#             C[:3, :3] = np.zeros((3,3))
#             # y[:4] = np.zeros((3,))
#             # print(C)
#         else:
#             dq = q_mul(y[:4], q_conj(q_tplus_t))
#             ya[:3] = quat_to_mrp(dq)
        
#         mu_tplus_t = self.stateUpdate(self.mu, u, Phi, B, self.dt)
#         Sig_tplus_t = Phi @ self.Sig @ Phi.T + self.Q

#         #### update step ####

#         # kalman gain calc
#         K = Sig_tplus_t @ C.T @ np.linalg.inv(C @ Sig_tplus_t @ C.T + self.R)

#         # meas model
#         z = self.measFunc(mu_tplus_t, C, self.dt)

#         # state mean and cov update
#         mu_tplus_tplus = mu_tplus_t + K @ (ya - z)
#         self.Sig = Sig_tplus_t - K @ C @ Sig_tplus_t

#         #### reset step ####
#         self.qref = self.quatReset(mu_tplus_tplus, q_tplus_t)
#         self.mu = np.concatenate((np.zeros((3,)), mu_tplus_tplus[3:]))

#         qw = np.concatenate([self.qref, self.mu[3:]])

#         return mu_tplus_tplus, qw, self.Sig
    
#     def quatReset(self, mu_post, q_update):
#         # slice MRP from posterior mean
#         apvec = mu_post[:3]
#         ap = np.linalg.norm(apvec)
        
#         # compose delta q
#         dq = np.zeros((4))
#         dq[0] = 16 - ap**2
#         dq[1:] = 8*apvec.reshape((3,))   
#         dq *= 1/(16 + ap**2)

#         # perform quat multiplication for reset
#         q_reset = q_mul(dq, q_update)

#         return q_reset
    
    # def linquatUpdate(self, Aqq, Aqw, qw):
    #     # extract velocities from current prior
    #     Aq = np.block([Aqq, Aqw])
    #     q_update = Aq @ qw

    #     return q_update

    # def checkObsv(self, u):
    #     A, _, C, _, _ = self.ssMatFunc(self.mu, u, self.dt)
    #     O = control.obsv(A, C)
    #     r = np.linalg.matrix_rank(O)
    #     n = np.min(O.shape)
    #     is_observable = r == n

    #     return is_observable