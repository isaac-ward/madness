# import numpy as np
import cvxpy as cvx
import cupy as cp
import numpy as np
from tqdm import tqdm

import utils.geometric
from utils.general import log_softmax, gradient_log_softmax, Cacher
import policies.costs
import policies.samplers

from dynamics_jax import DynamicsQuadcopter3D
from sdf import Environment_SDF, SDF_Types

class Trajectory:
    def __init__(
                self,
                state=None,
                action=None
            ):
        self.state = state
        self.action = action

class SCPSolver:
    def __init__(
            self,
            K,
            dynamics: DynamicsQuadcopter3D,
            trajInit: Trajectory,
            sdf:Environment_SDF,
            cost_tol = 1e-3,
            maxiter = 50.,
            sig = 10.,
            eps_dyn = 1.,
            eps_sdf = 1e-4,
            eps_rot = 1.,
            eta = 1.,
            eta_vec = np.array([1e-3, 10]),
            rho_vec = np.array([0, 0.1, 0.7]),
            beta_vec = np.array([2,2]),
            pull_from_cache=False
    ):
        self.K = K
        self.dynamics = dynamics
        self.sdf = sdf

        self.dt = dynamics.dt
        self.cost_tol = cost_tol
        self.maxiter = maxiter
        self.sig = sig
        self.eps_dyn = eps_dyn
        self.eps_sdf = eps_sdf
        self.eps_rot = eps_rot
        self.eta = eta
        self.eta_vec = eta_vec
        self.rho_vec = rho_vec
        self.beta_vec = beta_vec

        self.nu = self.dynamics.action_size()
        self.nx = self.dynamics.state_size()
        self.nss = len(self.sdf.sdf_list)

        E = np.eye(self.nx)
        # E = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2, 1, 1, 1, 1 ,1, 1])
        self.E = E

        self.action = cvx.Variable((self.K,self.nu))
        self.state = cvx.Variable((self.K + 1, self.nx))
        self.slack_sdf = cvx.Variable((self.K + 1, self.nss))
        self.slack_dyn = cvx.Variable((self.K, self.nx))
        # self.slack_quat = cvx.Variable(self.K+1)
        self.action_prev = trajInit.action
        self.state_prev = trajInit.state
        
        A, B, C = self.dynamics.affinize(self.state_prev[:-1], self.action_prev)
        A, B, C = np.array(A), np.array(B), np.array(C)

        propState = np.copy(trajInit.state)
        for i in range(K):  # Runs for all actions
            propState[i+1, :] = dynamics.step(propState[i, :], trajInit.action[i, :])

        # Compute affine components
        A_prop = np.array([A[i] @ propState[i, :] for i in range(K)])
        B_action = np.array([B[i] @ trajInit.action[i, :] for i in range(K)])
        residual = propState[1:, :] - (A_prop + B_action + C)

        self.slack_dyn_prev = cvx.Variable(self.K, self.nss)
        self.slack_dyn_prev.value = np.array([np.linalg.inv(self.E) @ residual[i, :] for i in range(K)])

        self.slack_sdf_prev = self.sdf.sdf_values(self.state_prev[:,:3])
        self.sdf = sdf
        self.constraints = []
        self.cost = np.inf
        self.rho_inc = 1
        self.slack_inc = 1

        self.pull_from_cache = pull_from_cache

    def dyn_constraints(
        self,
    ):
        
        A, B, C = self.dynamics.affinize(self.state_prev[:-1], self.action_prev)
        A, B, C = np.array(A),np.array(B),np.array(C)
        # E = np.eye(self.dynamics.state_size())
        # Dynamic feasibility constraints
        self.constraints += [ self.state[k+1] == A[k,:,:]@self.state[k] + B[k,:,:]@self.action[k] + C[k,:] + self.E@self.slack_dyn[k] for k in range(self.K) ]
        # Following two lines are for trust region constraints for state and action
        # self.constraints += [ cvx.norm_inf(self.state[k] - self.state_prev[k]) <= self.rho*self.rho_inc for k in range(self.K+1)]
        # self.constraints += [ cvx.norm_inf(self.action[k] - self.action_prev[k]) <= self.rho*self.rho_inc for k in range(self.K)]
        # self.constraints += [ cvx.norm_inf(self.slack_sdf[k] - self.slack_sdf_prev[k]) <= self.rho*self.rho_inc for k in range(self.K + 1)]

        self.constraints += [ cvx.norm_inf(self.state[:-1] - self.state_prev[:-1]) + cvx.norm_inf(self.action - self.action_prev) + cvx.norm_inf(self.slack_sdf - self.slack_sdf_prev,axis=1) < self.eta ]

        # self.constraints += [ cvx.norm(self.state[k, 3:7]) - 1 <= self.slack_quat[k] for k in range(self.K+1) ]

        # bouond on dynamics slack variable
        # slack_bound = self.slack_region*self.slack_inc
        #print(slack_bound)
        # self.constraints += [ cvx.norm( self.slack_dyn, p=1 ) <= slack_bound ]

        # return slack_bound
    
    def sdf_constraints(
            self
    ):
        
        # slack sdf prev is going to be a matrix (num_timesteps, num_sdfs)
        # G's shape is the same
        G = gradient_log_softmax(self.sig, self.slack_sdf_prev)
        # # affine part of the assembled matrix form of the constraints
        L0 = log_softmax(self.sig, self.slack_sdf_prev)

        self.constraints += [ cvx.diag( G @ (self.slack_sdf - self.slack_sdf_prev).T) + L0 >= 0 ]


        for i in range(self.nss):
            c = self.sdf.sdf_list[i].center_metres_xyz

            match self.sdf.sdf_list[i].sdf_type:
                case 0:
                    r = self.sdf.sdf_list[i].radius_metres
                    self.constraints += [ self.slack_sdf[k,i] <= 1 - (1/r)*cvx.norm2(self.state[k,:3] - c) for k in range(self.K + 1) ]
                case 1:
                    # NOT TESTED
                    s = self.sdf.sdf_list[i].diagonal_metres
                    self.constraints += [ self.slack_sdf[k,i] <= 1 - cvx.norm_inf( (self.state[k,:3] - c)/s ) for k in range(self.K + 1) ]
                    


    def boundary_constraints(
            self,
            state_goal,
            state_history,
    ):
        
        action_ranges = np.array(self.dynamics.action_ranges())

        self.constraints += [self.state[0] == state_history[-1]]
        self.constraints += [self.state[-1] == state_goal]
        self.constraints += [self.action[k] <= action_ranges[:,1] for k in range(self.K)]
        self.constraints += [self.action[k] >= action_ranges[:,0] for k in range(self.K)]

    def update_constraints(
            self,
            state_goal,
            state_history
            ):
        
        self.constraints = []

        self.dyn_constraints()
        self.sdf_constraints()
        self.boundary_constraints(state_goal, state_history)

        # return slack_bound

    def update_objective(
        self,
        state_goal
    ):
        # Get the action ranges for normalization
        ranges = self.dynamics.action_ranges()
        upper = ranges[:,1]
        action_upper_norm = np.square( np.linalg.norm(upper) )

        # Compute the action cost
        # action_cost = cvx.sum( [ cvx.square( cvx.norm(self.action[k], p=2)/norm_fac ) for k in range(self.K) ] ) / self.K
        # action_norms = cvx.norm(self.action, p=2, axis=1)  # Compute L2 norms for each action (axis=1 for rows)
        # action_cost = cvx.sum_squares(action_norms) / (action_upper_norm * self.K)  # Sum of squared normalized actions
        action_cost = cvx.square( cvx.norm(self.action, p='fro') ) / action_upper_norm

        # Compute cost for rotation rate
        rotation_cost = self.eps_rot * cvx.square( cvx.norm(self.state[:,-3:], p='fro') )

        # Compute virtual control running cost
        virtual_cost = self.eps_dyn * cvx.norm(self.E @ (self.slack_dyn).T, p=1)
        # virtual_cost = self.eps_dyn * cvx.norm(self.slack_dyn, p=1)

        # Compute the distance cost
        # Frobenius norm for position only
        distance_cost = cvx.square( cvx.norm(state_goal[np.newaxis,:3] - self.state[:,:3], p='fro') ) # TODO position only?

                # Compute the terminal cost
        #terminal_cost =  -self.eps_sdf*cvx.sum( self.slack_sdf ) + self.eps_dyn*cvx.norm( self.slack_dyn, p=1 ) #+ self.eps_quat*cvx.norm( self.slack_quat, p=1 )
        terminal_cost = -self.eps_sdf * cvx.sum(self.slack_sdf)
        
        # Final objective is a weighted sum of the above
        running_cost = action_cost + rotation_cost + virtual_cost # + distance_cost
        bolza_cost = running_cost + terminal_cost
        self.objective = bolza_cost
        # self.objective = terminal_cost + virtual_cost

        return action_cost, rotation_cost, virtual_cost, distance_cost, terminal_cost, bolza_cost

    def solve(
            self,
            state_goal,
            state_history,
            return_information=False,
            verbose=True,
    ):
        
        # Check if results cached for this
        computation_inputs_state = (
            state_goal,
            state_history,
            self.K,
            self.action_prev,
            self.state_prev,
            self.sdf.computation_inputs,
            self.cost_tol,
            self.maxiter,
            self.sig,
            self.eps_dyn,
            self.eps_sdf,
            self.eta,
            self.slack_region,
            "state"
        )
        computation_inputs_action = (
            state_goal,
            state_history,
            self.K,
            self.action_prev,
            self.state_prev,
            self.sdf.computation_inputs,
            self.cost_tol,
            self.maxiter,
            self.sig,
            self.eps_dyn,
            self.eps_sdf,
            self.eta,
            self.slack_region,
            "action"
        )

        log_action_cost = []
        log_rotation_cost = []
        log_virtual_cost = []
        log_distance_cost = []
        log_terminal_cost = []
        log_bolza_cost = []
        log_slack_bound = []
        log_solver_details = []

        # Add a progress bar   
        pbar = tqdm(total=self.maxiter, desc=f"Running SCP for max {self.maxiter} iterations")

        cacher_state = Cacher(computation_inputs_state)
        cacher_action = Cacher(computation_inputs_action)
        optimal_action_history = None
        optimal_state_history = None

        if self.pull_from_cache and cacher_state.exists() and cacher_action.exists():
            optimal_action_history = cacher_action.load()
            optimal_state_history = cacher_state.load()
        else:
            ii = 0
            while ii < self.maxiter:
                pbar.update(1)
                if verbose: print("SCP Iteration: ", ii)
                ii += 1
                slack_bound = self.update_constraints(state_goal, state_history)
                log_slack_bound.append(slack_bound)
                action_cost, rotation_cost, virtual_cost, distance_cost, terminal_cost, bolza_cost = self.update_objective(state_goal)
                prob = cvx.Problem(cvx.Minimize(self.objective), self.constraints)
                if verbose: print("Attempting to solve the problem")
                try:
                    clarabel_options = {
                        "tol_rel_gap": 1e-6,
                        "tol_abs_gap": 1e-6
                    }
                    prob.solve(solver=cvx.CLARABEL)# ,**clarabel_options)
                except:
                    prob.solve(solver=cvx.SCS)
                if verbose: print("Solver: " + str(prob.solver_stats.solver_name))
                if verbose: print("Problem Status: ", prob.status)
                if verbose: print("Cost: " + str(prob.value))
                cost = prob.value

                # Store everything
                log_action_cost.append(action_cost.value)
                log_rotation_cost.append(rotation_cost.value)
                log_virtual_cost.append(virtual_cost.value)
                log_distance_cost.append(distance_cost.value)
                log_terminal_cost.append(terminal_cost.value)
                log_bolza_cost.append(bolza_cost.value)
                log_solver_details.append((prob.solver_stats.solver_name, prob.status))
    
                delta_cost = prob.value - self.cost
                if np.abs(delta_cost) < self.cost_tol:
                    break

                if not(prob.status == cvx.OPTIMAL or prob.status == cvx.OPTIMAL_INACCURATE):
                    # print("look. we tried and now we are here. what can we do?")
                    self.state.value = np.copy(self.state_prev)
                    self.action.value =  np.copy(self.action_prev)
                    self.slack_sdf.value = np.copy(self.slack_sdf_prev)
                    self.rho_inc += 1
                    self.slack_inc *= 2
                    continue
                
                # print("we made it this far boys. let's pass it on")
                self.slack_region = np.linalg.norm(self.slack_dyn.value, ord=1)
                if verbose: print("Norm of slack_dyn: ", self.slack_region)
                self.cost = np.copy(prob.value)
                self.state_prev = np.copy(self.state.value)
                self.action_prev = np.copy(self.action.value)
                self.slack_sdf_prev = np.copy(self.slack_sdf.value)
                self.rho_inc = 1
                self.slack_inc = 1


            optimal_action_history = np.copy(self.action.value)
            optimal_state_history = np.copy(self.state.value)

            cacher_action.save(optimal_action_history)
            cacher_state.save(optimal_state_history)

        if return_information:
            return optimal_action_history, optimal_state_history, (log_action_cost, log_rotation_cost, log_virtual_cost, log_distance_cost, log_terminal_cost, log_bolza_cost, log_slack_bound), self.slack_dyn_prev.value, log_solver_details, prob
        else:
            return optimal_action_history, optimal_state_history
        
class SCvxSolver:
    """
    Just trying to simplify this problem
    """
    def __init__(
            self,
            dynamics:DynamicsQuadcopter3D,
            x_traj_init:np.ndarray,
            u_traj_init:np.ndarray,
            x_start:np.ndarray,
            x_goal:np.ndarray,
            sdf:Environment_SDF,
            sig=50,
            eps=1e-3,
            eps_ss=1e-4,
            verbose=False,
            pull_from_cache=False
    ):
        self.dynamics = dynamics
        self.x_traj_init = x_traj_init
        self.u_traj_init = u_traj_init
        self.x_start = x_start
        self.x_goal = x_goal
        self.sdf = sdf
        self.sig = sig
        self.eps = eps
        self.eps_ss = eps_ss
        self.verbose = verbose
        self.pull_from_cache = pull_from_cache

        self.N = np.shape(self.x_traj_init)[0]
        self.n = self.dynamics.state_size()
        self.m = self.dynamics.action_size()
        self.nss = len(self.sdf.sdf_list)
    
    def _print(
            self, 
            *args
    ):
        """
        Function to print messages to terminal when verbose option is enabled
        """
        if self.verbose:
            print(*args)
    
    def boundary_constraints(
            self,
            x:cvx.Variable,
    ):
        """
        Boundary constraints for state trajectory

        Equations 38f and 38g in paper.
        """
        # Create constraint list
        constraints = []

        # Add state boundary constraints
        constraints += [x[0] == self.x_start]
        constraints += [x[-1] == self.x_goal]

        return constraints
    
    def set_constraints(
            self,
            x:cvx.Variable,
            u:cvx.Variable,
    ):
        """
        Set constraints for both state and action trajectories.

        Equations 38c and 38d in paper.
        """
        # Create constraint list
        constraints = []

        # Add state set contraints
        # TODO add free final time constraint
        # TODO add velocity constraints
        # v_upper = np.array([1,1,1])
        # v_lower = -1 * w_upper
        # constraints += [x[_k,6:9] <= w_upper for _k in range(self.N)]
        # constraints += [x[_k,6:9] >= w_lower for _k in range(self.N)]
        # Add angular velocity constraints TODO future proof pls
        w_upper = np.array([0.5,0.5,0.5])
        w_lower = -1 * w_upper
        constraints += [x[_k,9:12] <= w_upper for _k in range(self.N)]
        constraints += [x[_k,9:12] >= w_lower for _k in range(self.N)]

        # Add control set constraints
        u_upper = np.array(self.dynamics.action_ranges())[:,1]
        u_lower = np.array(self.dynamics.action_ranges())[:,0]
        constraints += [u[k] <= u_upper for k in range(self.N - 1)]
        constraints += [u[k] >= u_lower for k in range(self.N - 1)]

        return constraints
    
    def sdf_constraints(
            self,
            x:cvx.Variable,
            x_prev:np.ndarray,
            slack_sdf:cvx.Variable
    ):
        """
        """
        # Create constraint list
        constraints = []

        # slack sdf prev is going to be a matrix (num_timesteps, num_sdfs)
        slack_sdf_prev = self.sdf.sdf_values(x_prev[:,:3])

        # G's shape is the same
        G = gradient_log_softmax(self.sig, slack_sdf_prev)

        # # affine part of the assembled matrix form of the constraints
        L0 = log_softmax(self.sig, slack_sdf_prev)
        
        constraints += [cvx.diag( G @ (slack_sdf - slack_sdf_prev).T) + L0 >= 0]

        for i in range(self.nss):
            c = self.sdf.sdf_list[i].center_metres_xyz

            match self.sdf.sdf_list[i].sdf_type:
                case 0:
                    r = self.sdf.sdf_list[i].radius_metres
                    constraints += [slack_sdf[k,i] <= 1 - (1/r)*cvx.norm2(x[k,:3] - c) for k in range(self.N)]
                case 1:
                    # NOT TESTED
                    s = self.sdf.sdf_list[i].diagonal_metres
                    constraints += [slack_sdf[k,i] <= 1 - cvx.norm_inf((x[k,:3] - c)/s ) for k in range(self.N)]
        
        return constraints
    
    def dynamic_constraints(
        self,
        x:cvx.Variable,
        u:cvx.Variable,
        nu:cvx.Variable,
        x_prev:np.ndarray,
        u_prev:np.ndarray,
        nu_max:float,
    ):
        """
        Dynamic constraints for states and controls

        Equations 36b
        """
        # Create constraint list
        constraints = []

        # Propagate states with model
        # x_prop = np.zeros_like(x_prev)
        # x_prop[0] = np.copy(x_prev[0])
        # for _k in range(self.N - 1):
        #     x_prop[_k + 1] = self.dynamics.step(x_prop[_k], u_prev[_k])

        # Get affinized dynamics
        A, B, C = self.dynamics.affinize(x_prev[:-1],u_prev)
        A, B, C = np.array(A),np.array(B),np.array(C)

        # Create virtual control term (recommended to be identity matrix)
        E = np.eye(self.n)#np.zeros_like(A[0])

        # Dynamic feasibility constraint (Equation 46a)
        constraints += [x[k+1] == A[k] @ x[k] + B[k] @ u[k] + C[k] + E @ nu[k] for k in range(self.N - 1)]

        constraints += [cvx.max(cvx.abs(nu)) <= nu_max]

        return constraints
    
    def trust_region_constraints(
            self,
            x:cvx.Variable,
            u:cvx.Variable,
            x_prev:np.ndarray,
            u_prev:np.ndarray,
            αx,
            αu,
            η,
    ):
        """
        Add trust region constraint to handle artificial unboundedness
        """
        # Create constraint list
        constraints = []

        # Add trust region constraint (Equation 45)
        constraints += [αx*cvx.norm2(x[_k] - x_prev[_k]) + αu*cvx.norm2(u[_k] - u_prev[_k]) <= η for _k in range(self.N - 1)]

        return constraints
    
    def objective_update(
            self,
            x:cvx.Variable,
            u:cvx.Variable,
            slack_sdf:cvx.Variable,
            nu:cvx.Variable,
            λ,
    ):
        """
        Update the objective function of the SCvx SCP problem.
        # TODO normalize?
        """
        # Define objective function
        objective = 0

        # Add control effort cost
        u_upper = np.array(self.dynamics.action_ranges())[:,1]
        u_lower = np.array(self.dynamics.action_ranges())[:,0]
        control_max = np.max([np.linalg.norm(u_upper),np.linalg.norm(u_lower)])**2 * (self.N - 1)
        control_objective = cvx.sum([(cvx.norm2(u[_k]))**2 for _k in range(self.N - 1)]) / control_max
        objective += control_objective

        # Add goal distance cost
        distance_max = (cvx.norm2(self.x_start - self.x_goal))**2 * (self.N - 1)
        distance_objective = cvx.sum([(cvx.norm2(x[_k] - self.x_goal))**2 for _k in range(self.N - 1)]) / distance_max
        objective += distance_objective

        # Add virtual control cost
        virtual_control_objective = λ * cvx.sum([cvx.norm2(nu[_k])**2 for _k in range(self.N - 1)])
        objective += virtual_control_objective

        # Add sdf terminal cost
        sdf_objective = -self.eps_ss * cvx.sum(slack_sdf)
        objective += sdf_objective

        return objective, control_objective, distance_objective, virtual_control_objective, sdf_objective
    
    def solve_failed(
            self,
            η,
            λ,
            virt_max,
            failed,
    ):
        """
        """
        λscale = 2.
        λmax = 1e6
        λmin = 1e-3
        ηscale = 2.
        ηmax = 10
        ηmin = 1e-3
        virt_scale = 2
        if failed:
            # λ /= λscale
            η *= ηscale
            virt_max *= virt_scale
        else:
            # if λ < λmax:
            #     λ *= λscale
            if (η > 1) and (virt_max > 1):
                η /= ηscale
                virt_max /= virt_scale

        return η, λ, virt_max


    def solve(
            self,
            max_iters=30,
            plot_progress_helper=None,
            return_information=False,
    ):
        """
        The Successive Convexification (SCvx) solver is outlined in "Convex Optimization for 
        Trajectory Generation" by Malyuta et al.

        SCvx is an applied methodology for solving Sequential Convex Programs (SCPs). SCP is a 
        framework to apply convex optimization techniques to solve an inherently non-convex problem.
        It works by solving a sequence of convex subproblems (which are approximations of the original
        non-convex problem). By the end, the user should have a fairly accurate local optimal solution
        for the problem.

        SCvx solves the SCP by employing
        1. Virtual control variables (slack variables)
        2. Adaptive trust regions

        TODO this shit is ass and slow as fuck need to better tune and scale (plus need better update rules)
        - αx
        - αu
        - ηinit
        - λinit
        - nu_max
        """
        # Check if results cached for this
        computation_inputs_state = (
            self.x_traj_init,
            self.u_traj_init,
            self.x_start,
            self.x_goal,
            self.sdf.computation_inputs,
            self.sig,
            self.eps,
            self.eps_ss,
            max_iters,
            "state"
        )
        computation_inputs_action = (
            self.x_traj_init,
            self.u_traj_init,
            self.x_start,
            self.x_goal,
            self.sdf.computation_inputs,
            self.sig,
            self.eps,
            self.eps_ss,
            max_iters,
            "action"
        )

        # Get state and action cachers
        cacher_state = Cacher(computation_inputs_state)
        cacher_action = Cacher(computation_inputs_action)

        # Create variables for state and action histories and logging
        optimal_action_history = None
        optimal_state_history = None
        logs_per_iter = []

        # Check if cache available
        if self.pull_from_cache and cacher_state.exists() and cacher_action.exists():
            optimal_action_history = cacher_action.load()
            optimal_state_history = cacher_state.load()
        else:
            # Define previous trajectory
            x_prev = np.copy(self.x_traj_init)
            u_prev = np.copy(self.u_traj_init)
            J_prev = np.inf

            # Define trust region parameters
            αx = 1.
            αu = 0
            ηinit = 1.
            η = np.copy(ηinit)
            
            # Define virtual control penalty
            λinit = 30.
            λ = np.copy(λinit)
            nu_max = 1.

            # Define SCvx convergence variables
            iters = 1
            converged = False

            # SCvx loop
            while (iters <= max_iters) and (not converged):
                # Print Info
                self._print("SCvx Iteration " + str(iters))
                self._print("   λ: " + str(λ))
                self._print("   η: " + str(η))
                self._print("   nu_max: " + str(nu_max))

                # Create convex variables
                x = cvx.Variable((self.N,self.n))
                u = cvx.Variable((self.N - 1,self.m))
                nu = cvx.Variable((self.N - 1,self.n))
                slack_sdf = cvx.Variable((self.N,self.nss))

                # Get problem constraints
                constraints = []
                constraints += self.boundary_constraints(x)
                # constraints += [x[0] == self.x_start]
                # if nu_max <= 1:
                #     constraints += [x[-1] == self.x_goal]
                constraints += self.dynamic_constraints(x,u,nu,x_prev,u_prev,nu_max)
                constraints += self.set_constraints(x,u)
                constraints += self.trust_region_constraints(x,u,x_prev,u_prev,αx,αu,η)
                constraints += self.sdf_constraints(x,x_prev,slack_sdf)

                # Get problem objective
                objective, control_objective, distance_objective, virtual_control_objective, sdf_objective = self.objective_update(x,u,slack_sdf,nu,λ)

                # Solve problem
                prob = cvx.Problem(cvx.Minimize(objective),constraints)
                try:
                    prob.solve(solver=cvx.CLARABEL)
                except:
                    η, λ, nu_max = self.solve_failed(η,λ,nu_max,1)
                    continue
                if not(prob.status == cvx.OPTIMAL or prob.status == cvx.OPTIMAL_INACCURATE):
                    η, λ, nu_max = self.solve_failed(η,λ,nu_max,1)
                    continue
                else:
                    η, λ, nu_max = self.solve_failed(η,λ,nu_max,0)
                J = prob.value

                # Check convergence criteria
                no_change = np.allclose(x.value,x_prev)
                if ((abs(J_prev - J) < self.eps) and (np.max(np.abs(nu.value)) < self.eps)) or no_change: # TODO don't be stupid
                    converged = True
                
                # Display improvement
                self._print("   Cost improvement: " + str(J_prev - J))
                self._print("   Max Virtual Control: " + str(np.max(np.abs(nu.value))))
                self._print("   Control Objective: " + str(control_objective.value))
                self._print("   Distance Objective: " + str(distance_objective.value))
                self._print("   Virtual Control Objective: " + str(virtual_control_objective.value))
                self._print("   SDF Objective: " + str(sdf_objective.value))

                # Penalize virtual control
                # if (np.linalg.norm(nu.value,np.inf)) > self.eps and (λ < λmax):
                #     λ *= λscale
                #     η /= ηscale

                # Store new previous trajectory
                x_prev = np.copy(x.value)
                u_prev = np.copy(u.value)
                J_prev = np.copy(J)

                # 
                log_per_iter = {"x":x_prev,
                                "u":u_prev,
                                "nu":nu.value,
                                "J":J_prev,
                                "u_cost":control_objective.value,
                                "x_cost":distance_objective.value,
                                "nu_cost":virtual_control_objective.value,
                                "sdf_cost":sdf_objective.value,
                                }
                logs_per_iter.append(log_per_iter)

                if plot_progress_helper is not None:
                    plot_progress_helper(x_prev,u_prev,iters,logs_per_iter)

                # Add iters
                iters += 1
            
            # Save state trajectory and actions
            optimal_state_history = np.copy(x.value)
            optimal_action_history = np.copy(u.value)

            # Cache
            cacher_action.save(optimal_action_history)
            cacher_state.save(optimal_state_history)

        return optimal_state_history,optimal_action_history,logs_per_iter

# TODO
class PolicyConvex:
    def __init__(
            self,
            dynamics,
            sdf_nearest,
            K,
            solverClass = SCPSolver,
            ):
        
        self.solver = solverClass(K, dynamics, )
        self.dynamics = dynamics
        
        self.sdf = sdf_nearest
        self.state_goal = None

    def update_state_goal(
        self,
        state_goal,
    ):
        """
        Update the path to follow
        """
        self.state_goal = state_goal

    def update_sdf(
            self,
            sdf_nearest
    ):
        self.sdf = sdf_nearest
        

    def new_traj(
            self,
            state_goal,
            sdf_nearest
    ):
        self.update_state_goal(state_goal)
        self.update_sdf(sdf_nearest)

    def act(
            self,
            state_history,
            action_history
    ):
    
        optimal_action_history, optimal_state_history = self.solver.solve(state_history, action_history)
        return optimal_action_history[0], optimal_state_history[0]
    

# -----------------------------------------------------------------------------------------------------------------------
