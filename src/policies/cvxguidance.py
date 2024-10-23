# import numpy as np
import cvxpy as cvx
import cupy as cp
import numpy as np

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
            eps_sdf = 1.,
            rho = 1.,
            slack_region = 1.,
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
        self.rho = rho
        self.slack_region = slack_region

        self.nu = self.dynamics.action_size()
        self.nx = self.dynamics.state_size()
        self.nss = len(self.sdf.sdf_list)

        self.action = cvx.Variable((self.K,self.nu))
        self.state = cvx.Variable((self.K + 1, self.nx))
        self.slack_sdf = cvx.Variable((self.K + 1, self.nss))
        self.slack_dyn = cvx.Variable((self.K, self.nx))
        self.action_prev = trajInit.action
        self.state_prev = trajInit.state
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
        self.constraints += [ self.state[k+1] == A[k,:,:]@self.state[k] + B[k,:,:]@self.action[k] + C[k,:] + self.slack_dyn[k] for k in range(self.K) ]
        self.constraints += [ cvx.norm_inf(self.state[k] - self.state_prev[k]) <= self.rho*self.rho_inc for k in range(self.K+1)]
        self.constraints += [ cvx.norm_inf(self.action[k] - self.action_prev[k]) <= self.rho*self.rho_inc for k in range(self.K)]

        slack_bound = self.slack_region*self.slack_inc
        
        print(slack_bound)

        self.constraints += [ cvx.norm( self.slack_dyn, p='fro' ) <= slack_bound ]
    
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

    def update_objective(
        self,
        state_goal
    ):
        terminal_cost =  -self.eps_sdf*cvx.sum( self.slack_sdf ) + self.eps_dyn*cvx.square( cvx.norm(self.slack_dyn, p='fro') )

        action_cost = cvx.square( cvx.norm(self.action, p='fro') )
        distance_cost = cvx.square( cvx.norm(state_goal[np.newaxis,:3] - self.state[:,:3], p='fro') ) # TODO position only?
        
        bolza_sum = action_cost + distance_cost

        self.objective = bolza_sum + terminal_cost

    def solve(
            self,
            state_goal,
            state_history
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
            self.rho,
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
            self.rho,
            self.slack_region,
            "action"
        )

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
                print("SCP Iteration: ", ii)
                ii += 1
                self.update_constraints(state_goal, state_history)
                self.update_objective(state_goal)
                prob = cvx.Problem(cvx.Minimize(self.objective), self.constraints)
                print("Attempting to solve the problem")
                try:
                    prob.solve(solver=cvx.CLARABEL)
                except:
                    prob.solve(solver=cvx.SCS)
                print("Solver: " + str(prob.solver_stats.solver_name))
                print("Problem Status: ", prob.status)
                print("Cost: " + str(prob.value))
    
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
                self.slack_region = np.linalg.norm(self.slack_dyn.value, ord='fro')
                print("Norm of slack_dyn: ", self.slack_region)
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

        return optimal_action_history, optimal_state_history

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
