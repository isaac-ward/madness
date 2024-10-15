# import numpy as np
import cvxpy as cvx
import cupy as cp
import numpy as np

import utils.geometric
from utils.general import log_softmax, gradient_log_softmax
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
            horizon = 1,
            sig = 50,
            eps = 1
    ):
        self.K = K
        self.dynamics = dynamics
        self.sdf = sdf
        # self.state_goal = state_goal

        self.dt = dynamics.dt
        self.horizon = horizon
        self.sig = sig
        self.eps = eps

        self.nu = self.dynamics.action_size()
        self.nx = self.dynamics.state_size()

        self.action = cvx.Variable((self.K,self.nu))
        self.state = cvx.Variable((self.K + 1, self.nx))
        self.slack_sdf = cvx.Variable((self.K + 1, len(self.sdf.sdf_list)))
        self.slack_dyn = cvx.Variable((self.K+1, self.nx))
        # self.alpha = cvx.Variable(1)
        self.action.value = trajInit.action
        self.state.value = trajInit.state
        self.slack_sdf.value = self.sdf.sdf_values(self.state.value[:,:3])
        self.sdf = sdf
        self.constraints = []

        self.step_count = 1

    def dyn_constraints(
            self,
    ):
        A, B, C = self.dynamics.affinize(self.state.value[:-1], self.action.value)
        A, B, C = np.array(A),np.array(B),np.array(C)
        print("A_0 = ", A[0,:,:])
        print("B_0  =", B[50,:,:])
        self.constraints += [ self.state[k+1] == A[k,:,:]@self.state[k] + B[k,:,:]@self.action[k] + C[k,:] + self.slack_dyn[k,:] for k in range(self.K) ]

    def sdf_constraints(
            self
    ):
        
        delta_0 = self.slack_sdf.value
        G = gradient_log_softmax(self.sig, self.slack_sdf.value)
        g0 = log_softmax(self.sig, delta_0)

        print("G.shape = ", G.shape)
        print("g0.shape = ", g0.shape)

        self.constraints += [ cvx.diag( G @ (self.slack_sdf - delta_0).T ) + g0 >= 0 ]

        for i in range(len(self.sdf.sdf_list)):
            c = self.sdf.sdf_list[i].center_metres_xyz

            match self.sdf.sdf_list[i].sdf_type:
                case 0:
                    r = self.sdf.sdf_list[i].radius_metres
                    self.constraints += [ self.slack_sdf[:,i] <= 1 - (1/r)*cvx.norm(self.state[:,:3] - c[np.newaxis,:]) ]
                case 1:
                    # NOT TESTED
                    s = self.sdf.sdf_list[i].diagonal_metres
                    self.constraints += [ self.slack_sdf[:,i] <= 1 - cvx.norm_inf( (self.state[:,:3] - c)/s )]
                    


    def boundary_constraints(
            self,
            state_goal,
            state_history,
    ):
        
        action_ranges = np.array(self.dynamics.action_ranges())

        self.constraints += [self.state[0] == state_history[-1]]
        self.constraints += [self.state[-1] == state_goal]
        # self.constraints += [self.action[k] <= action_ranges[:,1] for k in range(self.K)]
        # self.constraints += [self.action[k] >= action_ranges[:,0] for k in range(self.K)]

    def update_constraints(
            self,
            state_goal,
            state_history
            ):
        
        self.dyn_constraints()
        # self.sdf_constraints()
        self.boundary_constraints(state_goal, state_history)

    def update_objective(
        self,
        state_goal
    ):
        terminal_cost = -self.eps*cvx.sum( self.slack_sdf )

        action_cost = cvx.sum([ cvx.sum( [ cvx.square(self.action[k,u]) for u in range(self.nu) ] ) for k in range(self.K) ])

        distance_cost = cvx.sum( [ cvx.square( cvx.norm(state_goal - self.state[k]) ) for k in range(self.K+1) ] )
        
        bolza_sum = action_cost # + distance_cost

        self.objective = 0 # bolza_sum # + terminal_cost

    def solve(
            self,
            state_goal,
            state_history
    ):
        print("count: ", self.step_count)
        print("count mod horizon: ", self.step_count%self.horizon)
        if self.step_count%self.horizon == 0:
            self.update_constraints(state_goal, state_history)
            self.update_objective(state_goal)
            prob = cvx.Problem(cvx.Minimize(self.objective), self.constraints)
            print("attempting to solve the problem")
            prob.solve()
            print("Problem Status: ", prob.status)
            
            self.step_count = 1

        optimal_action_history = self.action.value[self.step_count-1:]
        optimal_state_history = self.state.value[self.step_count-1:]

        self.step_count += 1

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