import numpy as np
import cvxpy as cvx
import cupy as cp


import utils.geometric
import utils.general
import policies.costs
import policies.samplers

# TODO
# import sdf.sdf write and import sdf class that holds sphere locations and sizes (plus the field?)

class ConvexSolver:
    def __init__(
            self,
            dt,
            K,
            dynamics,
            state_history,
            state_goal
    ):
        self.dynamics = dynamics
        self.K = K
        self.dt = dt
        self.action = cvx.Variable((self.K,2))
        self.state = cvx.Variable(self.K + 1, self.dynamics.state_size())
        self.constraints = []
        self.objective = cvx.sum( [ cvx.square(self.action[i,1]) + cvx.square(self.action[i,2]) for i in range(K) ] )

    def dyn_constraints(
            self
    ):
        self.constraints += []

    def sdf_constraints(
            self
    ):
        self.constraints += []

    def boundary_constraints(
            self
    ):
        self.constraints += []

    def solve(
            self
    ):
        prob = cvx.Problem(cvx.Minimize(self.objective), self.constraints)
        prob.solve()

        optimal_action_history = self.action.value
        optimal_state_history = self.state.value

        return optimal_action_history, optimal_state_history


class PolicyConvex:
    def __init__(
            self,
            dynamics,
            sdf_nearest
            ):
        
        self.solver = ConvexSolver
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
            state_history
    ):
    
        optimal_action_history, optimal_state_history = self.solver.solve()
        return optimal_action_history[0], optimal_state_history[0]