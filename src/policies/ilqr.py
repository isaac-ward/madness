import jax.numpy as jnp
import numpy as np
from dynamics_jax import DynamicsQuadcopter3D
import os
import shutil

class PolicyiLQR:
    def __init__(
        self,
        dynamics:DynamicsQuadcopter3D,
        Q,
        R,
        QN,
        x_track,
        u_track,
        eps=1e-2,
        max_iters=1000,
        verbose=False
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
        self.verbose = verbose

        # Solve iLQR
        # self.x_bar,self.u_bar,self.Y,self.y = self.ilqr(
        #     x_track=self.x_track,
        #     u_track=self.u_track,
        #     quadrotor=self.dynamics,
        #     Q=self.Q,
        #     R=self.R,
        #     QN=self.QN,
        #     eps=self.eps,
        #     max_iters=self.max_iters
        # )

        # Solve AL-iLQR
        # TODO Add SDF constraints???
        self.x_bar,self.u_bar,self.Y,self.y = self.al_ilqr(
            x_track=self.x_track,
            u_track=self.u_track,
            dyn=self.dynamics,
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
        timestep
    ):
        """
        """
        # Get the optimal action and other logging information
        x = state_history[-1]
        optimal_action = self.u_bar[timestep] + self.y[timestep] + self.Y[timestep] @ (x - self.x_bar[timestep])

        # Cap the action range
        u_upper = np.array(self.dynamics.action_ranges())[:,1]
        u_lower = np.array(self.dynamics.action_ranges())[:,0]
        optimal_action = np.clip(optimal_action, u_lower, u_upper) # Restrict action with limits

        if self.verbose:
            print("u: " + str(optimal_action))

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
        u_upper = np.array(self.dynamics.action_ranges())[:,1]
        u_lower = np.array(self.dynamics.action_ranges())[:,0]

        # Check for a valid setup
        if max_iters <= 1:
            raise ValueError("Argument `max_iters` must be at least 1.")

        # Get state and control dimensions
        n = quadrotor.state_size()  # state dimension
        m = quadrotor.action_size()  # control dimension

        # Get total number of discrete control points on trajectory
        N = np.shape(u_track)[0]

        # Initialize control gains Y and offsets y
        Kk = np.zeros((N, m, n))
        dk = np.zeros((N, m))

        # Initialize the nominal trajectory deviations dx and du
        dx = np.zeros((N + 1, n))
        du = np.zeros((N, m))

        # Initialize the nominal trajectory x_bar and u_bar
        x_bar = np.zeros_like(x_track)
        x_bar[0] = np.copy(x_track[0])
        u_bar = np.copy(u_track)

        # Step through each discrete point and create a dynamically feasible trajectory
        # A_init, B_init = quadrotor.linearize(x_track[:-1],u_track)
        # A_init,B_init = np.array(A_init),np.array(B_init)
        # for k in range(N):
        #     x_bar[k+1] = A_init[k] @ x_bar[k] + B_init[k] @ u_bar[k]
        #     x_bar[k+1,3:7] /= np.linalg.norm(x_bar[k+1,3:7])
        
        # assert not np.any(np.isnan(A_init)), "A contains NaN values from tracked trajectory"
        # assert not np.any(np.isnan(B_init)), "B contains NaN values from tracked trajectory"
        # assert not np.any(np.isnan(x_bar)), "x_bar contains NaN values from tracked trajectory"
        # assert not np.any(np.isnan(u_bar)), "u_bar contains NaN values from tracked trajectory"
        # assert not np.any(np.isinf(A_init)), "A contains inf values from tracked trajectory"
        # assert not np.any(np.isinf(B_init)), "B contains inf values from tracked trajectory"
        # assert not np.any(np.isinf(x_bar)), "x_bar contains inf values from tracked trajectory"
        # assert not np.any(np.isinf(u_bar)), "u_bar contains inf values from tracked trajectory"
        

        for _k in range(N):
            x_bar[_k+1] = np.array(quadrotor.step(x_bar[_k], u_bar[_k]))
        print("x_bar: " + str(x_bar))
        # x_bar = np.copy(x_track)

        # Last cost
        J_last =  np.inf
        J = self.cost_function(x_bar,x_track,u_bar)

        # Regularization factor
        ρinit = 0
        ρ = ρinit
        ρmax = 1e-6
        ρinc = 1e-7

        # Get linearized jacobians
        A_total,B_total = quadrotor.linearize(x_bar[:-1],u_bar)
        A_total,B_total = np.array(A_total),np.array(B_total)

        assert not np.any(np.isnan(A_total)), "A contains NaN values from tracked trajectory"
        assert not np.any(np.isnan(B_total)), "B contains NaN values from tracked trajectory"
        assert not np.any(np.isinf(A_total)), "A contains inf values from tracked trajectory"
        assert not np.any(np.isinf(B_total)), "B contains inf values from tracked trajectory"

        ## iLQR loop
        # Create variable to exit loop given convergence achieved
        converged = False

        # Limit iterations with max_iters
        for _i in range(max_iters):
            # Backwards Pass: 
            # Build cost function gradients / Hessians at N
            lN_x = QN @ (x_bar[-1] - x_track[-1])
            lN_xx = np.copy(QN)

            # Calc cost to go at N
            p = np.copy(lN_x)
            P = np.copy(lN_xx)

            for _k in range(N-1,-1,-1):
                # Build cost function gradients / Hessians at kth step
                lk_x = Q @ (x_bar[_k] - x_track[_k])
                lk_u = R @ u_bar[_k]
                lk_xx = np.copy(Q)
                lk_uu = np.copy(R)
                lk_ux = 0

                # Get A, B
                A = np.copy(A_total[_k])
                B = np.copy(B_total[_k])

                # Build gradients / Hessians of action value function
                Q_xx = lk_xx + A.T @ P @ A
                Q_uu = lk_uu + B.T @ P @ B
                Q_ux = lk_ux + B.T @ P @ A
                Q_xu = np.copy(Q_ux.T)
                Q_x = lk_x + A.T @ p
                Q_u = lk_u + B.T @ p

                # Check if positive definite
                incrementing = True
                # Q_uu_reg = np.eye(np.shape(Q_uu)[0])*ρmax + Q_uu
                while incrementing:
                    Q_uu_reg = np.eye(np.shape(Q_uu)[0])*ρ + Q_uu
                    print("k: " + str(_k) + " / " + str(N-1))
                    print("Q_uu: " + str(Q_uu_reg))
                    print("eigs: " + str(np.linalg.eigvals(Q_uu_reg)))
                    print("A: " + str(A))
                    print("B: " + str(B))
                    if not np.all(np.linalg.eigvals(Q_uu_reg) > 0):
                        ρ += ρinc
                        if ρ > ρmax:
                            raise Exception("Hit maximum limit for regularization ρ = " + str(ρ))
                    else:
                        incrementing = False
                
                # Calc control gains
                inv_gain = -np.linalg.pinv(Q_uu_reg)
                Kk[_k] = inv_gain @ Q_ux
                dk[_k] = inv_gain @ Q_u
                P = Q_xx + Kk[_k].T @ Q_uu @ Kk[_k] + Kk[_k].T @ Q_ux + Q_xu @ Kk[_k]
                p = Q_x + Kk[_k].T @ Q_uu @ dk[_k] + Kk[_k].T @ Q_u + Q_xu @ dk[_k]
                # P = Q_xx + Kk[_k].T @ Q_uu @ Kk[_k] - Kk[_k].T @ Q_ux - Q_xu @ Kk[_k]
                # p = Q_x + Kk[_k].T @ Q_uu @ dk[_k] + Kk[_k].T @ Q_u - Q_xu @ dk[_k]
                # P = Q_xx + A.T@P@A - Kk[_k].T@Q_uu@Kk[_k]
                # p = Q_x + A.T@p + Q_ux.T@dk[_k]
                # assert not np.any(np.isnan(Kk[_k])), "Kk contains NaN values from tracked trajectory"
                # assert not np.any(np.isnan(dk[_k])), "dk contains NaN values from tracked trajectory"
                # assert not np.any(np.isinf(Kk[_k])), "Kk contains inf values from tracked trajectory"
                # assert not np.any(np.isinf(dk[_k])), "dk contains inf values from tracked trajectory"

            # Forwards Pass
            u = np.zeros((N, m))
            x = np.zeros((N + 1, n))
            x[0] = np.copy(x_track[0])
            for _k in range(N):
                dx[_k] = x[_k] - x_bar[_k]
                # print("x: " + str(np.linalg.norm(x[_k,3:7])))
                # print("x_bar: " + str(np.linalg.norm(x_bar[_k,3:7])))
                # print("dx: " + str(np.linalg.norm(dx[_k,3:7])))
                du[_k] = dk[_k] + Kk[_k] @ dx[_k] 
                u[_k] = u_bar[_k] + du[_k]
                u[_k] = np.clip(u[_k], u_lower, u_upper) # Restrict action with limits
                # print("u " + str(u[_k]))
                # print("x " + str(x[_k]))
                x[_k + 1] = np.array(quadrotor.step(x[_k],u[_k]))
            # assert not np.any(np.isnan(dx)), "dx contains NaN values " + str(dx)
            # assert not np.any(np.isnan(du)), "du contains NaN values " + str(du)
            x_bar = np.copy(x)
            u_bar = np.copy(u)
            # New cost
            J_last = np.copy(J)
            J = self.cost_function(x_bar,x_track,u_bar)
            improve = abs(J_last - J)

            print("iLQR iteration: " + str(_i) + "\nCost Improvement: " + str(J_last - J))
            print("J = " + str(J))
            print("J_last = " + str(J_last))

            if improve < eps and _i > 2:
                converged = True
                break

        # Verify solution found
        if not converged:
            raise RuntimeError("iLQR did not converge!")
        
        return x_bar, u_bar, Kk, dk
    
    def al_ilqr(
            self,
            x_track,
            u_track,
            dyn:DynamicsQuadcopter3D,
            Q,
            R,
            QN,
            eps=1e-2,
            max_iters=1000
    ):
        """
        """
        # Check for a valid setup
        if max_iters <= 1:
            raise ValueError("Argument `max_iters` must be at least 1.")

        # Get state and control dimensions
        n = dyn.state_size()        # state dimension
        m = dyn.action_size()       # control dimension
        N = np.shape(x_track)[0]    # timesteps
        
        # Create state and control vectors
        x = np.zeros((N,n))
        x[0] = np.copy(x_track[0])
        u = np.copy(u_track)

        # Propagate state with controls (dynamically feasible initial trajectory)
        for _k in range(0,N-1):
            x[_k+1] = dyn.step(x[_k],u[_k])
        
        # Create last iteration vectors
        x_last = np.copy(x)
        u_last = np.copy(u)
        J_last = self.cost_function(x,x_track,u)
        Kk = np.zeros((N-1,m,n))
        dk = np.zeros((N-1,m))

        # Create penalty matrix and lagrange multiplier
        c = self.constraints(dyn,x,u)
        μ = 1e-4 # Initial penalty
        φμ = 1.1 # Penalty scaling
        λ = np.zeros_like(c) # Lagrange multipler
        λ = self.update_lagrange_multiplier(λ,μ,dyn,x,u)
        Iμ = self.update_penalty_matrix(μ,λ,dyn,x,u) # Penalty matrix

        # Create iLQR loop
        converged = False
        iters = 1
        while (not converged) and (iters < max_iters):
            # Backward Pass
            print("Back Pass: " + str(iters) + " / " + str(max_iters))
            Kk, dk, deltaV = self.ilqr_backpass(
                x=x,
                x_track=x_track,
                u=u,
                dyn=dyn,
                Q=Q,
                R=R,
                QN=QN,
                λ=λ, 
                Iμ=Iμ
            )

            # Forward Pass
            print("Forward Pass: " + str(iters) + " / " + str(max_iters))
            x, u, J = self.ilqr_forwardpass(
                x_last=x_last,
                x_track=x_track,
                u_last=u_last,
                dyn=dyn,
                Kk=Kk,
                dk=dk,
                J_last=J_last,
                deltaV=deltaV
            )

            # Check convergence
            print("Cost Improvement: " + str(J_last - J))
            if abs(J - J_last) < eps:
                x_last = np.copy(x)
                u_last = np.copy(u)
                converged = True
            else:
                x_last = np.copy(x)
                u_last = np.copy(u)
                J_last = np.copy(J)
            
            # Iterate 
            μ += φμ*μ
            λ = self.update_lagrange_multiplier(λ,μ,dyn,x,u)
            Iμ = self.update_penalty_matrix(μ,λ,dyn,x,u)

            # Increment iter
            iters += 1
        
        return x_last, u_last, Kk, dk
    
    def ilqr_backpass(
            self,
            x:np.ndarray,
            x_track:np.ndarray,
            u:np.ndarray,
            dyn:DynamicsQuadcopter3D,
            Q:np.ndarray,
            R:np.ndarray,
            QN:np.ndarray,
            λ:np.ndarray,
            Iμ:np.ndarray
    ):
        """
        """
        # Get N time steps
        N = np.shape(x)[0]
        n = dyn.state_size()        # state dimension
        m = dyn.action_size()       # control dimension

        # Control gains
        Kk = np.zeros((N-1,m,n))
        dk = np.zeros((N-1,m))
        deltaV = np.zeros((N-1,2))

        # Build cost function gradients / Hessians at N
        lN_x = QN @ (x[-1] - x_track[-1])
        lN_u = 0
        lN_xx = QN
        lN_uu = 0
        lN_xu = 0
        lN_ux = 0

        # Build constraint gradients / Hessians at N
        c = self.constraints(dyn,x,u)
        cN_x = np.zeros_like(c[-1])

        # Calc cost to go at N
        p = lN_x + cN_x.T @ (λ[-1] + Iμ[-1] @ c[-1])
        P = lN_xx + cN_x.T @ Iμ[-1] @ cN_x

        # Regularization factor
        ρinit = 0
        ρ = ρinit
        ρmax = 1e-6
        ρinc = 1e-7

        # Get linearized jacobians
        A_total,B_total = dyn.linearize(x[:-1],u)
        A_total,B_total = np.array(A_total),np.array(B_total)

        # TODO vectorize?
        for _k in range(N-2,-1,-1):
            # Build cost function gradients / Hessians at kth step
            lk_x = Q @ (x[_k] - x_track[_k])
            lk_u = R @ u[_k]
            lk_xx = Q
            lk_uu = R
            lk_ux = 0

            # Build constraint gradients / Hessians at kth step
            ck_x = np.zeros_like(c[_k])
            ck_u = np.array([1,1,1,1,-1,-1,-1,-1])

            # Get A, B
            A = np.copy(A_total[_k])
            B = np.copy(B_total[_k])

            # Build gradients / Hessians of action value function
            Q_xx = lk_xx + A.T @ P @ A + ck_x.T @ Iμ[_k] @ ck_x
            Q_uu = lk_uu + B.T @ P @ B + ck_u.T @ Iμ[_k] @ ck_u
            Q_ux = lk_ux + B.T @ P @ A + ck_u.T @ Iμ[_k] @ ck_x
            Q_xu = np.copy(Q_ux.T)
            Q_x = lk_x + A.T @ p + ck_x.T @ (λ[_k] + Iμ[_k] @ c[_k])
            Q_u = lk_u + B.T @ p + ck_u.T @ (λ[_k] + Iμ[_k] @ c[_k])

            # Check if positive definite
            incrementing = True
            # Q_uu_reg = np.eye(np.shape(Q_uu)[0],np.shape(Q_uu)[1])*ρmax + Q_uu
            while incrementing:
                Q_uu_reg = np.eye(np.shape(Q_uu)[0],np.shape(Q_uu)[1])*ρ + Q_uu
                try:
                    # Attempt Cholesky decomposition
                    np.linalg.cholesky(Q_uu_reg)
                    # If successful, Q_uu_reg is positive definite
                    incrementing = False
                except np.linalg.LinAlgError:
                    ρ += ρinc
                    if ρ > ρmax:
                        raise Exception("Hit maximum limit for regularization: " + str(ρ))
            
            # Calc control gains
            inv_gain = -np.linalg.pinv(Q_uu_reg)
            assert not np.any(np.isnan(inv_gain)), "inv_gain contains NaN values"
            assert not np.any(np.isnan(Q_ux)), "Q_ux contains NaN values"
            assert not np.any(np.isnan(Q_u)), "Q_u contains NaN values"
            Kk[_k] = inv_gain @ Q_ux
            dk[_k] = inv_gain @ Q_u
            deltaV[_k] = np.array([dk[_k].T @ Q_u,0.5*dk[_k].T @ Q_uu @ dk[_k]])
            P = Q_xx + Kk[_k].T @ Q_uu @ Kk[_k] + Kk[_k].T @ Q_ux + Q_xu @ Kk[_k]
            p = Q_x + Kk[_k].T @ Q_uu @ dk[_k] + Kk[_k].T @ Q_u + Q_xu @ dk[_k]
        
        return Kk, dk, deltaV

    def ilqr_forwardpass(
            self,
            x_last:np.ndarray,
            x_track:np.ndarray,
            u_last:np.ndarray,
            dyn:DynamicsQuadcopter3D,
            Kk:np.ndarray,
            dk:np.ndarray,
            J_last:float,
            deltaV:np.ndarray,
            max_iters=50
    ):
        """
        """
        assert not np.any(np.isnan(Kk)), "inv_gain contains NaN values"
        assert not np.any(np.isnan(dk)), "Q_ux contains NaN values"
        assert not np.any(np.isinf(Kk)), "inv_gain contains inf values"
        assert not np.any(np.isinf(dk)), "Q_ux contains inf values"
        # Get N time steps, (n,m) dimensions
        N = np.shape(x_last)[0]
        n = dyn.state_size()
        m = dyn.action_size()

        # Create state and control vectors
        x = np.zeros((N,n))
        x[0] = np.copy(x_last[0])
        u = np.zeros((N-1,m))

        # Create deviation vectors
        dx = np.zeros((N,n))
        du = np.zeros((N-1,m))

        # Initialize line search parameters
        α = 1
        γ = 0.5
        β1 = 1e-4
        β2 = 10
        break_line_search = False
        iteration_count = 0

        # upper and lower clips
        u_upper = np.array(dyn.action_ranges())[:,1]
        u_lower = np.array(dyn.action_ranges())[:,0]

        while not break_line_search:
            iteration_count += 1
            print("Iteration Count: " + str(iteration_count))
            # Propagate trajectory
            for _k in range(0,N-1):
                assert not np.any(np.isnan(α)), "NaN detected in α " + str(α)
                assert not np.any(np.isnan(dk[_k])), "NaN detected in dk["+str(_k)+"] " + str(dk[_k])
                assert not np.any(np.isnan(Kk[_k])), "NaN detected in Kk["+str(_k)+"] " + str(Kk[_k])
                assert not np.any(np.isnan(x[_k])), "NaN detected in x["+str(_k)+"] " + str(x[_k])
                assert not np.any(np.isnan(x_last[_k])), "NaN detected in x_last["+str(_k)+"] " + str(x_last[_k])

                # Calc deviations at k
                dx[_k] = x[_k] - x_last[_k]             # Calc state deviation
                du[_k] = α * dk[_k] + Kk[_k] @ dx[_k]   # Calc control deviation using iLQR
                #print("α: " + str(α))
                #print("dk: " + str(dk[_k]))
                #print("Kk: " + str(Kk[_k]))
                #print("dx: " + str(dx[_k]))

                # Calc control and next state
                u[_k] = u_last[_k] + du[_k]             # Calc new control
                u[_k] = np.clip(u[_k], u_lower, u_upper)
                x[_k+1] = dyn.step(x[_k],u[_k])         # Calc next state
                #print("u_last: " + str(u_last[_k]))
                #print("du: " + str(du[_k]))
                assert not np.any(np.isnan(x[_k+1])), "NaN detected in x["+str(_k+1)+"] x=" + str(x[_k]) + " u=" + str(u[_k])
            
            # Check that line search conditions satisfied
            J = self.cost_function(x,x_track,u) # Calc cost function for this run

            z = (J - J_last) / np.sum([α * deltaV[_k,0] + (α**2) * deltaV[_k,1] for _k in range(0,np.shape(deltaV)[0])])
            print("z: " + str(z))
            if (z >= β1) and (z <= β2):
                # If value within line search range, return values
                break_line_search = True
            else:
                # If values not within line search range, increment alpha and rerun
                α = γ * α
                if iteration_count >= max_iters:
                    raise Exception("Max iterations reached for iLQR Forward Pass. z: " + str(z))

        return x, u, J
    
    def cost_function(
            self,
            x,
            x_track,
            u,
    ):
        """
        """
        # Get N time steps
        N = np.shape(x)[0]

        # Get cost function matrices
        Q = self.Q
        QN = self.QN
        R = self.R

        # Check for NaN or Inf values in matrices
        assert not np.any(np.isnan(Q)), "NaN detected in Q"
        assert not np.any(np.isnan(R)), "NaN detected in R"
        assert not np.any(np.isinf(Q)), "Inf detected in Q"
        assert not np.any(np.isinf(R)), "Inf detected in R"

        # Create cost function
        J = 0

        # Add final cost
        J += 0.5 * (x[-1] - x_track[-1]).T @ QN @ (x[-1] - x_track[-1])

        # Add cost at each step
        for _k in range(0,N-1):
            assert not np.any(np.isnan(x[_k] - x_track[_k])), "NaN detected in x[" + str(_k) + "] - x_track[" + str(_k) + "]" + str(x[_k] - x_track[_k])
            assert not np.any(np.isnan(u[_k])), "NaN detected in u[" + str(_k) + "] " + str(u[_k])
            assert not np.any(np.isinf(x[_k] - x_track[_k])), "Inf detected in x[" + str(_k) + "] - x_track[" + str(_k) + "]" + str(x[_k] - x_track[_k])
            assert not np.any(np.isinf(u[_k])), "Inf detected in u[" + str(_k) + "]" + str(u[_k])
            J += 0.5*((x[_k] - x_track[_k]).T @ Q @ (x[_k] - x_track[_k]) + u[_k].T @ R @ u[_k])
        
        return J
    
    def update_lagrange_multiplier(
            self,
            λ,
            μ,
            dyn:DynamicsQuadcopter3D,
            x:np.ndarray,
            u:np.ndarray
    ):
        """
        """
        # Get timesteps (N)
        N = np.shape(x)[0]

        # Get constraints
        c = self.constraints(
            dyn=dyn,
            x=x,
            u=u
        )
        
        # Update lagrange multiplier (λ)
        λ += np.maximum(0,λ+μ*c)
        
        return λ
    
    def update_penalty_matrix(
            self,
            μ,
            λ,
            dyn:DynamicsQuadcopter3D,
            x:np.ndarray,
            u:np.ndarray
    ):
        """
        """
        # Get constraints
        c = self.constraints(
            dyn=dyn,
            x=x,
            u=u
        )

        # Create penalty matrix
        Iμ_vect = (c > 0) * μ # TODO fix this

        # Update penalty matrix
        Iμ_vect[(c < 0) & (λ == 0)] = 0

        # Create diagonal penalty matrix
        n,m = np.shape(Iμ_vect) # n timesteps, m constraints
        Iμ = np.zeros((n,m,m))
        for _i in range(n):
            Iμ[_i] = np.diag(Iμ_vect[_i])
        
        return Iμ

    def constraints(
            self,
            dyn:DynamicsQuadcopter3D,
            x:np.ndarray,
            u:np.ndarray
    ):
        """
        """
        # Get timesteps (N)
        N = np.shape(x)[0]

        # Get upper and lower control bounds
        u_upper = np.array(dyn.action_ranges())[:,1]
        u_lower = np.array(dyn.action_ranges())[:,0]

        # Create constraint vector (N x i)
        # note: i = number of constraints
        ck = np.array([[u[_k]-u_upper,u_lower-u[_k]] for _k in range(0,N-1)])
        cN = np.zeros(np.shape(ck[0]))
        c = np.concatenate((ck, [cN]), axis=0)
        c = c.reshape(c.shape[0], -1)

        return c