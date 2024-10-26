import jax.numpy as jnp
import numpy as np
from dynamics_jax import DynamicsQuadcopter3D
import os
import shutil

class PolicyiLQR:
    def __init__(
        self,
        dynamics,
        Q,
        R,
        QN,
        x_track,
        u_track,
        eps=1e-3,
        max_iters=1000
    ):
        """
        Roll out a bunch of random actions and select the best one
        """
        self.dynamics = dynamics
        self.x_track = x_track
        self.u_track = u_track
        self.Q = Q
        self.R = R
        self.QN = QN
        self.log_folder = None
        self.eps = eps
        self.max_iters = max_iters

        # Solve iLQR
        self.x_bar,self.u_bar,self.Y,self.y = self.ilqr(
            x_track=self.x_track,
            u_track=self.u_track,
            quadrotor=self.dynamics,
            Q=self.Q,
            R=self.R,
            QN=self.QN,
            eps=self.eps,
            max_iters=self.max_iters
        )

    def enable_logging(
        self,
        run_folder,
    ):
        """
        Enable logging to a folder
        """
        self.log_folder = os.path.join(run_folder, "policy", "ilqr")

    def delete_logs(self):
        """
        Delete all logs
        """
        if self.log_folder is not None:
            shutil.rmtree(self.log_folder)
    
    def act(
        self,
        state_history,
        action_history,
        timestep,
    ):
        """
        """
        # Get the optimal action and other logging information
        x = state_history[-1]
        optimal_action = self.u_bar[timestep] + self.y[timestep] + self.Y[timestep] @ (x - self.x_bar[timestep])

        # ----------------------------------------------------------------
        # Logging from here on
        # ----------------------------------------------------------------

        # Log the state and action plans alongside the costs, 
        # if we're logging
        if self.log_folder is not None:
            pass

        return optimal_action

    # ----------------------------------------------------------------
    
    def ilqr(self,x_track,u_track,quadrotor:DynamicsQuadcopter3D,Q,R,QN,eps=1e-3,max_iters=1000):
        """
        Compute controls to track a given trajectory with iLQR. The iLQR tracking control law is described by
        the formula: u = u_bar + y + Y * (x - x_bar).
        This code is based on resources from Stanford AA203.
        Useful course notes can be found here: https://github.com/StanfordASL/AA203-Notes/blob/master/notes.pdf

        Parameters
        ----------
        x_track: numpy.ndarray
            Discrete state trajectory to track (dimensions N x n)
        u_track: numpy.ndarray
            Discrete initial control inputs to track trajectory (dimensions N x m)
        quadrotor: DynamicsQuadcopter3D
            Quadrotor dynamics object
        Q: numpy.ndarray
            The state cost matrix
        R: numpy.ndarray
            The control cost matrix
        QN: numpy.ndarray
            The terminal state cost matrix 
        eps: float
            Optional, convergence tolerance. Default 1e-3
        max_iters: int
            Optional, maximum allowable iterations of iLQR loop for convergence
        
        Returns
        -------
        x_bar: numpy.ndarray
            Discrete nominal state trajectory (dimensions N x n)
        u_bar: numpy.ndarray
            Discrete nominal control trajectory (dimensions N x m)
        Y: numpy.ndarray
            Discrete control gains for control law (dimensions N x m x n)
        y: numpy.ndarray
            Discrete control offset for control law (dimensions N x m)
        """
        # Check for a valid setup
        if max_iters <= 1:
            raise ValueError("Argument `max_iters` must be at least 1.")

        # Get state and control dimensions
        n = quadrotor.state_size()  # state dimension
        m = quadrotor.action_size()  # control dimension

        # Get total number of discrete control points on trajectory
        N = np.shape(u_track)[0]

        # Initialize control gains Y and offsets y
        Y = np.zeros((N, m, n))
        y = np.zeros((N, m))

        # Initialize the nominal trajectory deviations dx and du
        dx = np.zeros((N + 1, n))
        du = np.zeros((N, m))

        # Initialize the nominal trajectory x_bar and u_bar
        x_bar = np.zeros(np.shape(x_track))
        x_bar[0] = np.copy(x_track[0])
        u_bar = np.copy(u_track)

        # Step through each discrete point and create a dynamically feasible trajectory
        for _k in range(N):
            x_bar[_k+1] = np.array(quadrotor.step(x_bar[_k], u_bar[_k])) # Assert x_bar[k+1] = x_track[k+1]

        ## iLQR loop
        # Create variable to exit loop given convergence achieved
        converged = False

        # Limit iterations with max_iters
        # Algorithm sourced from section 3.1.2 in AA203 Course Notes
        for _i in range(max_iters):

            # Backwards Pass: 
            qN = QN@(x_bar[N]-x_track[-1])
            V = np.copy(QN)
            vbar = np.copy(qN)

            for _k in range(N-1,-1,-1):
                # Get Ak, Bk, and dk
                Ak,Bk = quadrotor.linearize(x_bar[_k],u_bar[_k])
                Ak,Bk = np.array(Ak),np.array(Bk)
                print("Ak: " + str(Ak))
                print("Bk: " + str(Bk))

                # Define cost functions
                print("x_bar: " + str(x_bar[_k]))
                print("x_track: " + str(x_track[_k]))
                qk = Q@(x_bar[_k]-x_track[_k])
                print("x_bar-x_track: " + str(x_bar[_k]-x_track[_k]))
                print("qk: " + str(qk))
                rk = R@u_bar[_k]
                print("rk: " + str(rk))
                
                # Define S
                reg = 1e-9 # term to help avoid singularities
                Su = rk + vbar.T@Bk
                print("Su: " +str(Su))
                Suu = R + Bk.T@V@Bk + reg*np.eye(m)
                print("Suu " + str(Suu))
                Sux = Bk.T@V@Ak
                print("Sux: " + str(Sux))

                # Define Y, y
                Y[_k] = -np.linalg.pinv(Suu)@Sux
                print("Y[k]: " + str(Y[_k]))
                y[_k] = -np.linalg.pinv(Suu)@Su
                print("y[k]: " + str(y[_k]))

                # Update V, vbar
                V = Q + Ak.T@V@Ak - Y[_k].T@Suu@Y[_k]
                print("V: " + str(V))
                vbar = qk + Ak.T@vbar + Sux.T@y[_k]
                print("vbar: " + str(vbar))

            # Forwards Pass
            u = np.zeros((N, m))
            x = np.zeros((N + 1, n))
            x[0] = np.copy(x_track[0])
            for _k in range(N):
                dx[_k] = x[_k] - x_bar[_k]
                print("dx: " + str(dx[_k]))
                du[_k] = y[_k] + Y[_k]@dx[_k]
                print("du: " + str(du[_k]))
                u[_k] = u_bar[_k] + du[_k]
                print("u: " + str(u[_k]))
                x[_k + 1] = np.array(quadrotor.step(x[_k],u[_k]))
                print("x: " + str(x[_k]))
            x_bar = np.copy(x)
            u_bar = np.copy(u)

            print("iLQR iteration: " + str(_i) + "\ndu: " + str(np.max(np.abs(du))) + "\n")

            if np.max(np.abs(du)) < eps:
                converged = True
                break

        # Verify solution found
        if not converged:
            raise RuntimeError("iLQR did not converge!")
        
        return x_bar, u_bar, Y, y