import jax.numpy as jnp
import src.dynamics_jax as dynamics
import os
import shutil

class PolicyiLQR:
    def __init__(
        self,
        state_size,
        action_size,
        dynamics,
        K,
        H,
        action_ranges,
        lambda_,
        map_,
        use_gpu_if_available=False,
    ):
        """
        Roll out a bunch of random actions and select the best one
        """
        self.state_size = state_size
        self.action_size = action_size
        self.dynamics = dynamics
        self.K = K
        self.H = H
        self.action_ranges = action_ranges

        # Lambda is the temperature of the softmax
        # infinity selects the best action plan, 0 selects uniformly
        self.lambda_ = lambda_

        # We need a map to plan paths against (e.g. collision checking)
        self.map_ = map_

        # Typically we'll sample about the previous best action plan
        self._previous_optimal_action_plan = jnp.zeros((H, action_size))

        # Will need a path to follow, but we want it to be
        # updated separately (changeable)
        self.path_xyz = None

        # Are we going to have logging? Defaultly no
        self.log_folder = None

        # If we're using a GPU, we'll need to move some things over
        self.use_gpu_if_available = use_gpu_if_available

        # Defaultly no logging
        self.log_folder = None

        # Defaultly no goal
        self.state_goal = None
    
    def update_state_goal(
        self,
        state_goal,
    ):
        """
        Update the path to follow
        """
        self.state_goal = state_goal

    def enable_logging(
        self,
        run_folder,
    ):
        """
        Enable logging to a folder
        """
        self.log_folder = os.path.join(run_folder, "policy", "mppi")

    def delete_logs(self):
        """
        Delete all logs
        """
        if self.log_folder is not None:
            shutil.rmtree(self.log_folder)

    # ----------------------------------------------------------------
    
    def ilqr(self,x_track,u_track,quadrotor:dynamics.DynamicsQuadcopter3D,Q,R,QN,eps=1e-3,max_iters=1000):
        """
        Compute controls to track a given trajectory with iLQR. The iLQR tracking control law is described by
        the formula: u = u_bar + y + Y * (x - x_bar).
        This code is based on resources from Stanford AA203.
        Useful course notes can be found here: https://github.com/StanfordASL/AA203-Notes/blob/master/notes.pdf

        Parameters
        ----------
        x_track: numpy.ndarray
            Discrete state trajectory to track (dimensions Nxn)
        u_track: numpy.ndarray
            Discrete initial control inputs to track trajectory (dimensions Nxm)
        quadrotor: dynamics.DynamicsQuadcopter3D
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
        n = jnp.shape(x_track)[1]  # state dimension
        m = jnp.shape(u_track)[1]  # control dimension

        # Get total number of discrete control points on trajectory
        N = jnp.shape(u_track)[0]

        # Initialize control gains Y and offsets y
        Y = jnp.zeros((N, m, n))
        y = jnp.zeros((N, m))

        # Initialize the nominal trajectory x_bar and u_bar
        x_bar = jnp.zeros(jnp.shape(x_track))
        x_bar[0] = jnp.copy(x_track[0])
        u_bar = jnp.copy(u_track)

        # Initialize the nominal trajectory deviations dx and du
        dx = jnp.zeros((N + 1, n))
        du = jnp.zeros((N, m))

        # Initialize the time steps
        dt = jnp.zeros(u_track.shape[0])

        # Step through each discrete point and create a dynamically feasible trajectory
        for _k in range(N):
            dt[_k] = jnp.linalg.norm(x_track[_k+1,[0,2]]-x_track[_k,[0,2]])/jnp.linalg.norm(x_track[_k,[1,3]])
            x_bar[_k + 1] = quadrotor.dynamics_true_no_disturbances(x_bar[_k], u_bar[_k], dt=dt[_k])

        ## iLQR loop
        # Create variable to exit loop given convergence achieved
        converged = False

        # Limit iterations with max_iters
        # Algorithm sourced from section 3.1.2 in AA203 Course Notes
        for _i in range(max_iters):

            # Backwards Pass: 
            qN = QN@(x_bar[N]-x_track[-1])
            V = jnp.copy(QN)
            vbar = jnp.copy(qN)

            for _k in range(N-1,-1,-1):
                # Get Ak, Bk, and dk
                Ak,Bk = quadrotor.linearize(x_bar[_k],u_bar[_k],dt[_k])

                # Define cost functions
                qk = Q@(x_bar[_k]-x_track[_k])
                rk = R@u_bar[_k]
                
                # Define S
                reg = 1e-9 # term to help avoid singularities
                Su = rk + vbar.T@Bk
                Suu = R + Bk.T@V@Bk + reg*jnp.eye(m)
                Sux = Bk.T@V@Ak

                # Define Y, y
                Y[_k] = -jnp.linalg.pinv(Suu)@Sux
                y[_k] = -jnp.linalg.pinv(Suu)@Su

                # Update V, vbar
                V = Q + Ak.T@V@Ak - Y[_k].T@Suu@Y[_k]
                vbar = qk + Ak.T@vbar + Sux.T@y[_k]

            # Forwards Pass
            u = jnp.zeros((N, m))
            x = jnp.zeros((N + 1, n))
            x[0] = jnp.copy(x_track[0])
            for _k in range(N):
                dx[_k] = x[_k] - x_bar[_k]
                du[_k] = y[_k] + Y[_k]@dx[_k]
                u[_k] = u_bar[_k] + du[_k]
                dt[_k] = jnp.linalg.norm(x_bar[_k+1,[0,2]]-x_bar[_k,[0,2]])/jnp.linalg.norm(x_bar[_k,[1,3]])
                x[_k + 1] = quadrotor.dynamics_true_no_disturbances(x[_k], u[_k], dt=dt[_k])
            x_bar = jnp.copy(x)
            u_bar = jnp.copy(u)

            print("iLQR iteration: " + str(_i) + "\ndu: " + str(jnp.max(jnp.abs(du))) + "\n")

            if jnp.max(jnp.abs(du)) < eps:
                converged = True
                break

        # Verify solution found
        if not converged:
            raise RuntimeError("iLQR did not converge!")
        
        return x_bar, u_bar, Y, y