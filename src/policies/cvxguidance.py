# import numpy as np
import cvxpy as cvx
import cupy as cp
import jax.numpy as jnp



import utils.geometric
import utils.general
import policies.costs
import policies.samplers

from src.dynamics_jax import DynamicsQuadcopter3D
from src.standard import SDFSphere # CORRECT THIS
from src.standard import Astar # CORRECT THIS

# TODO
# import sdf.sdf write and import sdf class that holds sphere locations and sizes (plus the field?)

class SCPSolver:
    def __init__(
            self,
            K,
            dynamics: DynamicsQuadcopter3D,
            state_goal,
            trajInit = Astar,
            horizon = 1
    ):
        self.K = K
        self.dynamics = dynamics
        self.state_goal = state_goal
        
        self.dt = dynamics.dt
        self.horizon = horizon

        self.nu = self.dynamics.action_size()
        self.nx = self.dynamics.state_size()


        self.action = cvx.Variable((self.K,self.nu))
        self.state = cvx.Variable((self.K + 1, self.nx))
        self.action.value, self.state.value = Astar()
        self.constraints = []
        self.objective = cvx.sum( [ cvx.sum( [ cvx.square(self.action[k,u]) for u in range(self.nu) ] ) for k in range(self.K) ] )

        self.step_count = 1

    def dyn_constraints(
            self,
            state_history,
            action_history
    ):
        A, B, C = self.dynamics.affinize(state_history[-1], action_history[-1])
        self.constraints += [ self.state[k+1] == A@self.state[k] + B@self.action[k] + C for k in range(self.K) ]

    def sdf_constraints(
            self
    ):
        self.constraints += []

    def boundary_constraints(
            self,
            state_history,
    ):
        self.constraints += [self.state[0] == state_history[-1]]
        self.constraints += [self.state[self.K] == self.state_goal]
        self.constraints += [self.state[k] <= self.dynamics.action_ranges[:,1] for k in range(self.K)]
        self.constraints += [self.state[k] >= self.dynamics.action_ranges[:,0] for k in range(self.K)]

    def update_constraints(
            self,
            state_history, 
            action_history):
        
        self.dyn_constraints(state_history, action_history)
        self.sdf_constraints()
        self.boundary_constraints(state_history)

    def solve(
            self,
            state_history,
            action_history
    ):
        if self.step_count%self.horizon == 0:
            
            self.update_constraints(state_history, action_history)
            prob = cvx.Problem(cvx.Minimize(self.objective), self.constraints)
            prob.solve()
            
            self.step_count = 1

        optimal_action_history = self.action.value[self.step_count-1:]
        optimal_state_history = self.state.value[self.step_count-1:]

        self.step_count += 1

        return optimal_action_history, optimal_state_history


class PolicyConvex:
    def __init__(
            self,
            dynamics,
            sdf_nearest,
            K,
            solverClass = SCPSolver,
            ):
        
        self.solver = solverClass(K, dynamics,)
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