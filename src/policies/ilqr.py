import jax.numpy as jnp
import numpy as np
from dynamics import DynamicsQuadcopter3D
import os
import shutil
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class PolicyALiLQR:
    """
    Class that solves for and executes an AL-iLQR trajectory tracking policy.

    Parameters
    ----------
    dynamics: DynamicsQuadcopter3D
        Class describing the nonlinear dynamics of a system
    Q: np.ndarray
        The state cost matrix (real, symmetric, positive semi-definite matrix) dimensions n x n
    R: np.ndarray
        The control cost matrix (real, symmetric, positive-definite matrix) dimensions m x m
    QN: np.ndarray
        The terminal state cost matrix (real, symmetric, positive semi-definite matrix) dimensions n x n
    W: np.ndarray
        The continuity cost matrix dimensions m x m
    x_track: np.ndarray
        An array of discrete points making up a state trajectory we wish to track
    u_track: np.ndarray
        An array of discrete controls which is our "best guess" for executing the x_track trajectory
    segments: int
        Number of segments to divide the AL-iLQR problem into, default is a single segment (aka solving 1 problem)
    eps: float
        The convergence criteria for AL-iLQR. If the cost improvement is less than this threshold for any iteration,
        consider it converged. Default is 1e-2
    max_iters: int
        The maximum number of iterations for the AL-iLQR solver
    verbose: boolean
        Print status messages to terminal while solving tracking problem
    """
    def __init__(
        self,
        dynamics:DynamicsQuadcopter3D,
        Q:np.ndarray,
        R:np.ndarray,
        QN:np.ndarray,
        W:np.ndarray,
        x_track:np.ndarray,
        u_track:np.ndarray,
        segments=1,
        eps=1e-2,
        max_iters=1000,
        verbose=False,
        run_folder=None,
    ):
        """
        Initialization function for PolicyiLQR class
        """
        # Store system dynamics
        self.dynamics = dynamics        # System dynamics

        # Store trajectory to track
        self.x_track = x_track          # State trajectory to track
        self.u_track = u_track          # Open loop control inputs for x_track

        # Store iLQR cost matrices
        self.Q = Q                      # State cost matrix
        self.R = R                      # Control cost matrix
        self.QN = QN                    # Terminal state cost matrix
        self.W = W                      # Continuity cost matrix
        
        # AL-iLQR parameters
        self.eps = eps                  # Convergence criteria
        self.max_iters = max_iters      # Maximum allowable iterations
        self.segments = segments        # Segments for AL-iLQR problem

        # Class variables
        self.run_folder = run_folder    # Run folder for logging
        self.log_folder = None          # Log folder for logging
        self.verbose = verbose          # Verbose option for class operations

        # Internal diagnostic variables
        self.state_error = []           # State error
        self.cost = []                  # Costs - a list of lists where each list is a segment
        self.compute_time = []          # Solve time for each segment
        self.seg_length = []            # Numer of steps per segment
        self.constraint_violations = [] # Constraint violations

        # Verify inputs are valid
        if self.max_iters <= 1:
            raise ValueError("Argument `max_iters` must be at least 1")
        
        # Enable logging if desired
        if self.run_folder != None:
            self.enable_logging(self.run_folder)

        # Solve segmented AL-iLQR
        self.x_bar,self.u_bar,self.Kk,self.dk = self.segmented_al_ilqr(
            x_track=self.x_track,
            u_track=self.u_track,
            segments=self.segments,
        )
    
    def _print(
            self, 
            *args
    ):
        """
        Function to print messages to terminal when verbose option is enabled
        """
        if self.verbose:
            print(*args)

    def enable_logging(
        self,
        run_folder,
    ):
        """
        Enable logging to a folder
        """
        self.log_folder = os.path.join(run_folder, "policy", "al_ilqr")
        # Make subfolders and exists ok
        os.makedirs(self.log_folder, exist_ok=True)

    def delete_logs(
            self
    ):
        """
        Function to delete all logs
        """
        if self.log_folder is not None:
            shutil.rmtree(self.log_folder)
    
    def generate_logs(
            self,
    ):
        """
        Function to generate log files on the performance of AL-iLQR
        """
        # Get class variables
        dyn = self.dynamics         # System dynamics

        # Get state and control dimensions
        N = np.shape(self.x_track)[0]
        n = dyn.state_size()
        m = dyn.action_size()

        # TODO Plot iLQR nominal trajectory (each segment with different colors)
        # v.plot_environment_from_objects(
        #         map_=map_,
        #         sdfs=sdfs,
        #         path_xyz=path_xyz,
        #         path_xyz_smooth=path_xyz_smooth,
        #         path_xyz_cvx=x[:,:3],
        #         path_propagated=propagated_traj_path,
        #         path_al_ilqr=None,
        #         save_filename=f"environment_{indx}",
        #     )

        # Plot compute time and length per segment
        fig, axs = plt.subplots(1,2,figsize=(12, 18))
        fig.suptitle("Solve Time for Each AL-iLQR Segment")
        for i, ax in enumerate(axs.flatten()):
            if i == 0:
                ax.plot(self.compute_time)
                ax.set_xlabel("Segment Number")
                ax.set_ylabel("Compute Time (s)")
                ax.grid()
            elif i == 1:
                ax.plot(self.seg_length)
                ax.set_xlabel("Segment Number")
                ax.set_ylabel("Number of Discrete Points")
                ax.grid()
        
        # Adjust figure spacing
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save state error plots
        plt.savefig(os.path.join(self.log_folder,'segment_info.png'))
        plt.close()

        # TODO Plot nominal trajectory improvement per iteration

        # Plot state errors
        state_err_rows = -(n // -2)
        fig, axs = plt.subplots(state_err_rows, 2, figsize=(12, 18))
        fig.suptitle("State Error with Respect to Tracked Trajectory")

        state_error_array = np.array(self.state_error)

        for i, ax in enumerate(axs.flatten()):
            if i < state_error_array.shape[1]: # Ensure you don't exceed the number of states
                ax.plot(state_error_array[:, i])
                ax.set_xlabel("Timestep")
                ax.set_ylabel(str(self.dynamics.state_labels()[i]) + "-State Error")
                ax.grid()
            else:
                ax.axis('off') # Hide unused subplots
        
        # Adjust figure spacing
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save state error plots
        plt.savefig(os.path.join(self.log_folder,'state_error_evolution.png'))
        plt.close()

        # Plot cost evolution
        cost_labels = ["AL-iLQR Total Cost","iLQR Terminal Cost","AL Terminal Cost","iLQR Tracking Cost","AL Tracking Cost","Continuity Cost"]
        for _j in range(len(self.cost)):
            cost_array = np.array(self.cost[_j])
            cost_rows = -(len(cost_labels) // -2)
            fig, axs = plt.subplots(cost_rows, 2, figsize=(12, 18))
            fig.suptitle("Cost Evolution for Each AL-iLQR Iteration (Segment "+str(_j)+")")

            for i, ax in enumerate(axs.flatten()):
                if i < len(cost_labels): # Ensure you don't exceed the number of cost components
                    ax.plot(cost_array[:, i])
                    ax.set_xlabel("AL-iLQR Iteration")
                    ax.set_ylabel(cost_labels[i])
                    ax.grid()
                else:
                    ax.axis('off') # Hide unused subplots
            
            # Adjust figure spacing
            fig.subplots_adjust(hspace=0.4, wspace=0.3)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            plt.savefig(os.path.join(self.log_folder,'segment_'+str(_j)+'_cost_evolution.png'))
            plt.close()

            # Save cost evolution data to csv
            np.savetxt(os.path.join(self.log_folder,'segment_'+str(_j)+'_cost_evolution.csv'), cost_array, delimiter=",", fmt="%.6f", header=",".join(cost_labels), comments="")
    
    def act(
        self,
        state_history:np.ndarray,
        action_history:np.ndarray,
        timestep_1_base:int,
    ):
        """
        Function to execute iLQR control
        """
        # Get the optimal action and other logging information
        timestep = timestep_1_base - 1
        x = state_history[-1]
        try:
            optimal_action = self.u_bar[timestep] + self.dk[timestep] + self.Kk[timestep] @ (x - self.x_bar[timestep])
            
            # Store state error
            self.state_error.append((x - self.x_track[timestep]))
        except:
            optimal_action = self.u_bar[-1] + self.dk[-1] + self.Kk[-1] @ (x - self.x_bar[-1])

            # Store state error
            self.state_error.append((x - self.x_track[-1]))

        

        # Store constraint violations
        # c = self.constraints(x,optimal_action) # Need one-off constraint checking
        # self.constraint_violations.append()

        # Cap the action range
        u_upper = np.array(self.dynamics.action_ranges())[:,1]
        u_lower = np.array(self.dynamics.action_ranges())[:,0]
        optimal_action = np.clip(optimal_action, u_lower, u_upper) # Restrict action with limits

        # TODO Save constraint violations

        # Print controls executed at what timestep
        # self._print("Timestep: " + str(timestep_1_base) + "/" + str(self.dk.shape[0]))
        # self._print("u: " + str(optimal_action))

        # Log the state and action plans alongside the costs, 
        # if we're logging
        # TODO
        if self.log_folder is not None:
            pass

        return optimal_action

    
    def segmented_al_ilqr(
            self,
            x_track:np.ndarray,
            u_track:np.ndarray,
            segments=1,
    ):
        """
        Compute closed loop control policy using AL-iLQR to track a given trajectory. This function allows for solving the
        AL-iLQR problem in segments, which will ideally aid in convergence and reduce the importance of a "good" initial 
        u_track guess. 
        
        The iLQR tracking control law is described by the formula: u = u_nominal + y + Y * (x - x_nominal).

        Parameters
        ----------
        x_track: np.ndarray
            An array of discrete points making up a state trajectory we wish to track
        u_track: np.ndarray
            An array of discrete controls which is our "best guess" for executing the x_track trajectory
        segments: int
            Number of segments to divide the AL-iLQR problem into, default is a single segment (aka solving 1 problem)
        
        Returns
        -------
        x_nominal: np.ndarray
            Nominal state trajectory for AL-iLQR
        u_nominal: np.ndarray
            Nominal control inputs for AL-iLQR
        Kk: np.ndarray
            Feedback control gains
        dk: np.ndarray
            Feedforward control gains
        """
        # Get class variables
        dyn = self.dynamics         # System dynamics

        # Get state and control dimensions
        N = np.shape(x_track)[0]
        n = dyn.state_size()
        m = dyn.action_size()

        # Create control variables
        x_nominal = np.zeros_like(x_track)
        u_nominal = np.zeros_like(u_track)
        Kk = np.zeros((N-1,m,n))
        dk = np.zeros((N-1,m))

        # Create segmenting variables
        x_start = np.copy(x_track[0])   # state to start at when propagating u_track
        segment_size = N // segments    # number of states in each segment
        u_continuity = None             # last control input from prior segment

        # Loop through each AL-iLQR problem segment
        for _segs in range(segments):
            # Start timer
            start_time = time.time()

            # Get start and end indices
            start_step = int(_segs * segment_size)
            end_step = int(min((start_step + segment_size), N - 1))
            self.seg_length.append(end_step-start_step)
            
            # Populate continuity and starting state if prior segment exists
            if _segs != 0:
                #x_start = np.copy(x_track[start_step])
                x_start = np.copy(x_nominal[start_step]) # Start at last state of previous segment
                #u_continuity = np.copy(u_nominal[start_step-1])
            
            # Print status if verbose
            self._print("\nRunning AL-iLQR over segment " + str(_segs + 1) + "/" + str(segments) + " (" + str(start_step)
                        + "," + str(end_step) + ")")
            self._print("---------------------------------------------------------")

            # Run AL-iLQR over segment
            x_nominal_seg, u_nominal_seg, Kk_seg, dk_seg = self.al_ilqr(
                x_start=x_start,
                x_track=x_track[start_step:end_step + 1],
                u_track=u_track[start_step:end_step],
            )
            #     u_continuity=u_continuity
            # )
            
            # Store solution segments
            x_nominal[start_step:end_step + 1] = np.copy(x_nominal_seg)
            u_nominal[start_step:end_step] = np.copy(u_nominal_seg)
            Kk[start_step:end_step] = np.copy(Kk_seg)
            dk[start_step:end_step] = np.copy(dk_seg)

            # End timer
            end_time = time.time()
            self.compute_time.append(end_time-start_time)
        
        # Return control sequence
        return x_nominal, u_nominal, Kk, dk
    
    def al_ilqr(
            self,
            x_start:np.ndarray,
            x_track:np.ndarray,
            u_track:np.ndarray,
            u_continuity=None
    ):
        """
        AL-iLQR algorithm to solve the tracking problem. Implementation based off of implementation outlined 
        in "AL-iLQR Tutorial" by Brian Jackson. Given a state trajectory to track and a "best guess" on the
        control inputs required to execute this trajectory, solve for a closed loop iLQR controller that will
        track the state trajectory. 
        
        The iLQR tracking control law is described by the formula: u = u_nominal + y + Y * (x - x_nominal).

        Parameters
        ----------
        x_start: np.ndarray
            The initial state to propagate u_track over
        x_track: np.ndarray
            An array of discrete points making up a state trajectory we wish to track
        u_track: np.ndarray
            An array of discrete controls which is our "best guess" for executing the x_track trajectory
        u_continuity: np.ndarray
            The final action from the last segment of the tracking problem, default to None
        """
        # Get class variables
        eps = self.eps              # Convergence criteria
        max_iters = self.max_iters  # Maximum AL-iLQR iterations
        dyn = self.dynamics         # System dynamics

        # Get state and control dimensions
        N = np.shape(x_track)[0]    # timesteps
        n = dyn.state_size()        # state dimension
        m = dyn.action_size()       # control dimension
        
        # Create state and control vectors
        x = np.zeros((N,n))     # State trajectory
        x[0] = np.copy(x_start)
        u = np.copy(u_track)    # Control inputs

        # Propagate state with controls (dynamically feasible initial trajectory)
        for _k in range(0,N-1):
            x[_k+1] = dyn.step(x[_k],u[_k])

        # Get constraints
        c,_,_ = self.constraints(x,u)

        # Create penalty matrix and lagrange multiplier
        μ = 1e-4                                                        # Initial penalty
        φμ = 2.                                                         # Penalty scaling
        λ = self.update_lagrange_multiplier(μ,np.zeros_like(c),x,u)     # Lagrange multipler
        Iμ = self.update_penalty_matrix(μ,λ,x,u)                        # Penalty matrix

        # Create variables to store results from the previous iteration
        self.cost.append([])
        x_last = np.copy(x)                                             # Last iteration trajectory
        u_last = np.copy(u)                                             # Last iteration control inputs
        J_last = self.cost_function(x,x_track,u,λ,Iμ,u_continuity,1)    # Last iteration cost

        # Control gains
        Kk = np.zeros((N-1,m,n))    # Feedback gain
        dk = np.zeros((N-1,m))      # Feedforward gain

        # Create regularization variables
        ρmult_init = 1.                 # Initial regularization multiplier 
        ρmult = np.copy(ρmult_init)     # Regularization multiplier
        ρscaling = 10.                  # Regularization scaling variable

        # Create AL-iLQR loop variables
        converged = False   # Convergence variable
        iters = 1           # Current iteration
        forward_err = 0     # Line search convergence failure

        pbar = tqdm(range(max_iters), desc="Backward/forward pass", total=max_iters)
        
        # AL-iLQR loop
        while (not converged) and (iters < max_iters):
            pbar.update(1)

            # Execute backward Pass
            #self._print("Backward Pass: " + str(iters) + "/" + str(max_iters))
            Kk, dk, deltaV = self.al_ilqr_backpass(
                x=x,
                x_track=x_track,
                u=u,
                λ=λ, 
                Iμ=Iμ,
                u_continuity=u_continuity,
                ρmult=ρmult
            )

            # Execute forward Pass
            #self._print("Forward Pass: " + str(iters) + "/" + str(max_iters))
            forward_err = 0
            x, u, J, forward_err = self.al_ilqr_forwardpass(
                x_last=x_last,
                x_track=x_track,
                u_last=u_last,
                Kk=Kk,
                dk=dk,
                J_last=J_last,
                deltaV=deltaV,
                λ=λ,
                Iμ=Iμ,
                u_continuity=u_continuity,
            )

            # If unable to complete line search, increase backward pass regularization
            if forward_err:
                # Scale up regularization multiplier
                ρmult *= ρscaling
                self._print("**Warning** Line Search convergence failed, increasing regularization term (ρ) by " + str(ρmult) + "x")
                continue
            elif ρmult > ρmult_init:
                # Scale down regularization multiplier
                ρmult = ρmult / ρscaling
            
            # Log cost for this iteration
            J = self.cost_function(
                x=x,
                x_track=x_track,
                u=u,
                λ=λ,
                Iμ=Iμ,
                u_continuity=u_continuity,
                iter_cost=1
            )

            # Check AL-iLQR convergence
            #self._print("   Cost Improvement: " + str(J_last - J) + "\n")
            change_in_cost = J - J_last
            cost_improvement = -change_in_cost
            pbar.set_description(f"Backward/forward pass | cost_improvement={cost_improvement:.4f} | J_last={J_last:.4f} -> J={J:.4f}")
   
            if ((cost_improvement < eps) and (cost_improvement >= 0)):# or (cost_improvement < 0):
                x_last = np.copy(x)
                u_last = np.copy(u)
                converged = True
            else:
                x_last = np.copy(x)
                u_last = np.copy(u)
                J_last = np.copy(J)

            if abs(cost_improvement) > 1e9:
                cost_labels = ["AL-iLQR Total Cost","iLQR Terminal Cost","AL Terminal Cost","iLQR Tracking Cost","AL Tracking Cost","Continuity Cost"]
                cost_matrix = np.array(self.cost[-1])
                for _z in range(len(cost_labels)):
                    print(cost_labels[_z] + ": " + str(cost_matrix[-1,_z]))
            
            # Update AL variables
            c,_,_ = self.constraints(x,u)
            if np.max(c > 0):
                μ = φμ*μ
            λ = self.update_lagrange_multiplier(μ,λ,x,u)
            Iμ = self.update_penalty_matrix(μ,λ,x,u)
            for _i in range(c.shape[0]):
                if (λ[_i] + 0.5*c[_i].T @ Iμ[_i]).T @ c[_i] < 0:
                    print("------------------")
                    print("λ: \n" + str(λ[_i]))
                    print("c: \n" + str(c[_i]))
                    print("Iμ: \n" + str(Iμ[_i]))

            # Increment iteration variable
            iters += 1

        pbar.close()
        
        return x_last, u_last, Kk, dk
    
    def al_ilqr_backpass(
            self,
            x:np.ndarray,
            x_track:np.ndarray,
            u:np.ndarray,
            λ:np.ndarray,
            Iμ:np.ndarray,
            u_continuity=None,
            ρmult=1,
    ):
        """
        The AL-iLQR backward pass algorithm

        The purpose of the AL-iLQR backward pass is to compute the locally optimal control policy characterized 
        by the feedback (Kk) and feedforward (dk) gains. First, you solve Riccati-like equations starting from 
        the terminal point in the trajectory and iterating backwards (hence the name backwards pass). With these 
        equations solved, we can solve for the gains.

        Parameters
        ----------
        x: np.ndarray
            State trajectory for current iteration
        x_track: np.ndarray
            An array of discrete points making up a state trajectory we wish to track
        u: np.ndarray
            An array of discrete controls which is our "best guess" for executing the x_track trajectory
        λ: np.ndarray
            Lagrange multipliers
        Iμ: np.ndarray
            Penalty matrix at each timestep
        u_continuity: np.ndarray
            The final action from the last segment of the tracking problem, default to None
        ρmult: int
            Scaling factor for regularization
        
        Returns
        -------
        Kk: np.ndarray
            Feedback gain
        dk: np.ndarray
            Feedforward gain
        deltaV: np.ndarray
            Expected cost improvement for this backward pass
        """
        # Get class variables
        Q = self.Q          # State cost matrix
        R = self.R          # Control cost matrix
        QN = self.QN        # Terminal state cost matrix
        W = self.W          # Continuity cost matrix
        dyn = self.dynamics # System dynamics

        # Get state and control dimensions
        N = np.shape(x_track)[0]    # timesteps
        n = dyn.state_size()        # state dimension
        m = dyn.action_size()       # control dimension

        # Control gains
        Kk = np.zeros((N-1,m,n))    # Feedback gain
        dk = np.zeros((N-1,m))      # Feedforward gain
        deltaV = np.zeros((N-1,2))  # Expected cost improvement

        # Build cost function gradients / Hessians at N
        lN_x = QN @ (x[-1] - x_track[-1])   # Terminal iLQR cost (dx)
        lN_xx = QN                          # Terminal iLQR cost (dxdx)

        # Build constraint gradients / Hessians at N
        c, c_x, c_u = self.constraints(x,u)       # Constraints

        # Calc cost to go at N
        p = lN_x + c_x[-1].T @ (λ[-1] + Iμ[-1] @ c[-1])
        P = lN_xx + c_x[-1].T @ Iμ[-1] @ c_x[-1]

        # Regularization factor
        ρmax = 1e4          # Maximum regularization
        ρinc = 2            # Regularization scaling factor
        ρinit = 1e-9
        ρ = max(ρmult * ρinit,ρinit)  # Regularization factor

        # Verify regularization is ok
        if ρ > ρmax:
            raise Exception("Hit maximum limit for regularization (ρ = " + str(ρ) + ")")

        # Get linearized jacobians
        A_total,B_total = dyn.linearize(x[:-1],u)
        A_total,B_total = np.array(A_total),np.array(B_total)

        # Step through each timestep
        for _k in range(N-2,-1,-1):
            # Build cost function gradients / Hessians at kth step
            lk_x = Q @ (x[_k] - x_track[_k])    # iLQR cost at k (dx)
            lk_u = R @ u[_k]                    # iLQR cost at k (du)
            lk_xx = Q                           # iLQR cost at k (dxdx)
            lk_uu = R                           # iLQR cost at k (dudu)
            lk_ux = 0                           # iLQR cost at k (dudx)

            # Check for continuity considerations
            if (u_continuity is not None and u_continuity.any()) and (_k == 0):
                lk_u += W @ (u_continuity - u[0])
                lk_uu += W

            # Get A, B jacobians at k
            A = np.copy(A_total[_k]) 
            B = np.copy(B_total[_k]) 

            # Build gradients / Hessians of action value function
            Q_xx = lk_xx + A.T @ P @ A + c_x[_k].T @ Iμ[_k] @ c_x[_k]
            Q_uu = lk_uu + B.T @ P @ B + c_u[_k].T @ Iμ[_k] @ c_u[_k]
            Q_ux = lk_ux + B.T @ P @ A + c_u[_k].T @ Iμ[_k] @ c_x[_k]
            Q_xu = np.copy(Q_ux.T)
            Q_x = lk_x + A.T @ p + c_x[_k].T @ (λ[_k] + Iμ[_k] @ c[_k])
            Q_u = lk_u + B.T @ p + c_u[_k].T @ (λ[_k] + Iμ[_k] @ c[_k])

            # Check Q_uu positive definite
            incrementing = True
            while incrementing:
                Q_uu_reg = np.eye(np.shape(Q_uu)[0],np.shape(Q_uu)[1])*ρ + Q_uu
                try:
                    # Attempt Cholesky decomposition
                    # np.linalg.cholesky(Q_uu_reg)
                    np.linalg.pinv(Q_uu_reg)
                    # If successful, Q_uu_reg is positive definite
                    incrementing = False
                except np.linalg.LinAlgError:
                    ρ *= ρinc
                    self._print("**Warning** Increasing regularization term (ρ = " + str(ρ) + ")")
                    _k = N-2
                    if ρ > ρmax:
                        raise Exception("Hit maximum limit for regularization (ρ = " + str(ρ) + ")")
            
            # Calc control gains
            inv_gain = -1 * np.linalg.pinv(Q_uu_reg)
            assert not np.any(np.isnan(inv_gain)), "inv_gain contains NaN values"
            assert not np.any(np.isnan(Q_ux)), "Q_ux contains NaN values"
            assert not np.any(np.isnan(Q_u)), "Q_u contains NaN values"
            Kk[_k] = inv_gain @ Q_ux
            dk[_k] = inv_gain @ Q_u
            deltaV[_k] = np.array([dk[_k].T @ Q_u,0.5*dk[_k].T @ Q_uu @ dk[_k]])
            P = Q_xx + Kk[_k].T @ Q_uu @ Kk[_k] + Kk[_k].T @ Q_ux + Q_xu @ Kk[_k]
            p = Q_x + Kk[_k].T @ Q_uu @ dk[_k] + Kk[_k].T @ Q_u + Q_xu @ dk[_k]
        
        return Kk, dk, deltaV

    def al_ilqr_forwardpass(
            self,
            x_last:np.ndarray,
            x_track:np.ndarray,
            u_last:np.ndarray,
            Kk:np.ndarray,
            dk:np.ndarray,
            J_last:float,
            deltaV:np.ndarray,
            λ:np.ndarray,
            Iμ:np.ndarray,
            max_iters=50,
            u_continuity=None
    ):
        """
        The AL-iLQR forward pass algorithm. 
        
        The forward pass applies the control gains calculated in the backward pass to determine the new nominal 
        trajectory. Evaluate the cost of this new trajectory and compare it to the previous trajectory. Check if 
        the trajectory has improved sufficiently and scale with a line search (adjusting the feedforward gains).

        Parameters
        ----------
        x_last: np.ndarray
            Last AL-iLQR iteration state trajectory
        x_track: np.ndarray
            An array of discrete points making up a state trajectory we wish to track
        u_last: np.ndarray
            Last AL-iLQR iteration control inputs
        Kk: np.ndarray
            Feedback gain
        dk: np.ndarray
            Feedforward gain
        J_last: float
            Last AL-iLQR iteration cost
        deltaV: np.ndarray
            Expected cost improvement for this backward pass
        λ: np.ndarray
            Lagrange multipliers
        Iμ: np.ndarray
            Penalty matrix at each timestep
        max_iters: int
            Maxium iterations for linesearch, default to 50
        u_continuity: np.ndarray
            The final action from the last segment of the tracking problem, default to None
        
        Returns
        -------
        x: np.ndarray
            New state trajectory
        u: np.ndarray
            New control inputs
        J: float
            New cost of trajectory
        forward_err: int
            0 if line search converged, 1 if line search fails to converge
        """
        # Get class variables
        dyn = self.dynamics # System dynamics
    
        # Get state and control dimensions
        N = np.shape(x_track)[0]    # timesteps
        n = dyn.state_size()        # state dimension
        m = dyn.action_size()       # control dimension

        # Create state and control vectors
        x = np.zeros((N,n))
        x[0] = np.copy(x_last[0])
        u = np.zeros((N-1,m))

        # Create deviation vectors
        dx = np.zeros((N,n))
        du = np.zeros((N-1,m))

        # Initialize line search parameters
        αinit = 1                   # Initial line search scale value
        α = np.copy(αinit)          # Line search scaling value
        γ = 0.5                     # Scale the line search scale value
        β1 = 1e-5                   # Lower line search bound
        β2 = 1e3                     # Upper line search bound
        break_line_search = False   # Line search completed
        iteration_count = 0         # Total iterations of line search
        forward_err = 0             # 1 if line search fails to converge

        # Upper and Lower control bounds TODO debug AL
        u_upper = np.array(dyn.action_ranges())[:,1]#*5
        u_lower = np.array(dyn.action_ranges())[:,0]#*5

        pbar = tqdm(range(max_iters), desc="Optimizing alpha", total=max_iters, leave=False)

        pbar = tqdm(range(max_iters), desc="Optimizing alpha", total=max_iters, leave=False)

        # Perform line search
        while not break_line_search:
            # Increment iteration
            iteration_count += 1
            pbar.update(1)

            # Propagate trajectory with new control gains
            for _k in range(0,N-1):
                # Calculate deviations at kth step
                dx[_k] = x[_k] - x_last[_k]             # State deviation
                du[_k] = α * dk[_k] + Kk[_k] @ dx[_k]   # Control deviation

                # Calc new control and next state
                u[_k] = u_last[_k] + du[_k]                 # New control
                u[_k] = np.clip(u[_k], u_lower, u_upper)    # Clip control to feasible range (prevents dynamics from blowing up)
                x[_k+1] = dyn.step(x[_k],u[_k])             # Next state

                # Debugging assert
                assert not np.any(np.isnan(x[_k+1])), "NaN detected in x["+str(_k+1)+"] x=" + str(x[_k]) + " u=" + str(u[_k])
            
            # Calculate cost of this AL-iLQR solution
            J = self.cost_function(
                x=x,
                x_track=x_track,
                u=u,
                λ=λ,
                Iμ=Iμ,
                u_continuity=u_continuity
            )

            # Calculate ratio of improvement to predicted improvement
            z = (J_last - J) / (-1 * np.sum([α * deltaV[_k,0] + (α**2) * deltaV[_k,1] for _k in range(0,N-1)]))

            # Evaluate line search
            if (z >= β1) and (z <= β2):
                break_line_search = True
            else:
                # If values not within line search range, increment alpha and loop
                α = γ * α
                # If max iterations hit, report error
                if iteration_count >= max_iters:
                    self._print("**Warning** Max iterations reached for iLQR Forward Pass (z = " + str(z) + ")")
                    forward_err = 1
                    break_line_search = True

        pbar.close()

        return x, u, J, forward_err
    
    def cost_function(
            self,
            x:np.ndarray,
            x_track:np.ndarray,
            u:np.ndarray,
            λ:np.ndarray,
            Iμ:np.ndarray,
            u_continuity=None,
            iter_cost=0,
    ):
        """
        For the AL-iLQR tracking problem, compute the total cost.

        Parameters
        ----------
        x: np.ndarray
            State trajectory
        x_track: np.ndarray
            An array of discrete points making up a state trajectory we wish to track
        u: np.ndarray
            Control inputs
        λ: np.ndarray
            Lagrange multipliers
        Iμ: np.ndarray
            Penalty matrix at each timestep
        u_continuity: np.ndarray
            The final action from the last segment of the tracking problem, default to None
        
        Returns
        -------
        J: float
            The total cost of this solution of the AL-iLQR problem
        """
        # Get class variables
        Q = self.Q      # State cost matrix
        R = self.R      # Control cost matrix
        QN = self.QN    # Terminal state cost matrix
        W = self.W      # Continuity cost matrix

        # Get state and control dimensions
        N = np.shape(x_track)[0] # timesteps

        # Get AL constraint costs
        c,_,_ = self.constraints(x,u)

        # Create total cost function
        J = 0

        # Add final iLQR cost
        J_iLQR_terminal = 0.5 * (x[-1] - x_track[-1]).T @ QN @ (x[-1] - x_track[-1])
        J += J_iLQR_terminal

        # Add final AL cost
        J_AL_terminal = (λ[-1] + 0.5*c[-1] @ Iμ[-1]).T @ c[-1]
        J += J_AL_terminal

        # Add AL-iLQR cost at each step
        J_iLQR_tracking = 0
        J_AL_tracking = 0
        for _k in range(0,N-1):
            # iLQR cost
            J_iLQR_tracking += 0.5*((x[_k] - x_track[_k]).T @ Q @ (x[_k] - x_track[_k]) + u[_k].T @ R @ u[_k])
            
            # AL cost
            J_AL_tracking += (λ[_k] + 0.5*c[_k].T @ Iμ[_k]).T @ c[_k]
        J += J_iLQR_tracking
        J += J_AL_tracking
        
        # Add continuity cost
        J_continuity = 0
        if u_continuity is not None and u_continuity.any():
            J_continuity = 0.5 * (u_continuity - u[0]).T @ W @  (u_continuity - u[0])
        J += J_continuity
        
        # Add to diagnostic variables
        if iter_cost:
            self.cost[-1].append([J,J_iLQR_terminal,J_AL_terminal,J_iLQR_tracking,J_AL_tracking,J_continuity])

        return J
    
    def update_lagrange_multiplier(
            self,
            μ:float,
            λ:np.ndarray,
            x:np.ndarray,
            u:np.ndarray
    ):
        """
        For the AL-iLQR tracking problem, compute the Lagrange multipliers at each timestep.

        Parameters
        ----------
        μ: float
            Penalty term
        λ: np.ndarray
            Lagrange multipliers
        x: np.ndarray
            State trajectory
        u: np.ndarray
            Control inputs

        Returns
        -------
        λ: np.ndarray
            Lagrange multipliers (timesteps) x (number of constraints)
        """
        # Get constraints
        c,_,_ = self.constraints(x,u)
        
        # Update lagrange multiplier (λ)
        λ = np.maximum(0,λ+μ*c)
        
        # If constraint satisfied, don't blow up
        λ[c <= 0] = 0
        
        return λ
    
    def update_penalty_matrix(
            self,
            μ:float,
            λ:np.ndarray,
            x:np.ndarray,
            u:np.ndarray
    ):
        """
        For the AL-iLQR tracking problem, compute the penalty matrix at each timestep.

        Parameters
        ----------
        μ: float
            Penalty term
        λ: np.ndarray
            Lagrange multipliers
        x: np.ndarray
            State trajectory
        u: np.ndarray
            Control inputs

        Returns
        -------
        Iμ: np.ndarray
            Penalty matrix at each timestep
        """
        # Get constraints
        c,_,_ = self.constraints(x,u)

        # Create penalty matrix
        Iμ_vect = (c > 0) * μ

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
            x:np.ndarray,
            u:np.ndarray
    ):
        """
        Given a state and control trajectory, compute the constraints for the AL-iLQR tracking problem.

        Parameters
        ----------
        x: np.ndarray
            State trajectory
        u: np.ndarray
            Control inputs

        Returns
        -------
        c: np.ndarray
            Constraint vector at each timestep (timestep) x (number of constraints)
        c_x: np.ndarray
            Constraint vector gradient wrt state at each timestep (timestep) x (number of constraints)
        c_u: np.ndarray
            Constraint vector gradient wrt control at each timestep (timestep) x (number of constraints)
        """
        # Get class variables
        dyn = self.dynamics # System dynamics

        # Get state and control dimensions
        N = np.shape(x)[0] # timesteps

        # Get upper and lower control bounds
        u_upper = np.array(dyn.action_ranges())[:,1]
        u_lower = np.array(dyn.action_ranges())[:,0]

        # Create constraint vector (N x i)
        # note: i = number of constraints
        ck = np.array([[u[_k]-u_upper,u_lower-u[_k]] for _k in range(0,N-1)])
        cN = np.zeros(np.shape(ck[0]))
        c = np.concatenate((ck, [cN]), axis=0)
        c = c.reshape(c.shape[0], -1)

        # Calculate first derivative of constraints wrt x
        ck_x = np.array([np.zeros_like(ck[0]) for _k in range(0,N-1)])
        cN_x = np.zeros(np.shape(ck_x[0]))
        c_x = np.concatenate((ck_x, [cN_x]), axis=0)
        c_x = c_x.reshape(c_x.shape[0], -1)

        # Calculate first derivative of constraints wrt u
        ck_u = np.array([[[1,1,1,1],[-1,-1,-1,-1]] for _k in range(0,N-1)])
        cN_u = np.zeros(np.shape(ck_u[0]))
        c_u = np.concatenate((ck_u, [cN_u]), axis=0)
        c_u = c_u.reshape(c_u.shape[0], -1)

        return c, c_x, c_u