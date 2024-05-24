import numpy as np

class MPPI:
    """
    Implementation of model predictive path integral control
    """

    def __init__(
        self, 
        dynamics_fn, 
        control_bounds_lower,
        control_bounds_upper,
        K,
        H,
        lambda_,
    ):

        """
        dynamics_fn: function that takes in a state and action and returns the next state
        control_bounds_lower: lower bounds of the action space
        control_bounds_upper: upper bounds of the action space
        K: number of samples to take
        H: length of the horizon
        lambda_: temperature parameter - if infinity, then we just take the single best action, if 0, then we take a uniform distribution over explored action sequences
        
        """
        self.dynamics_fn = dynamics_fn
        self.control_bounds_lower = control_bounds_lower
        self.control_bounds_upper = control_bounds_upper
        self.K = K
        self.H = H
        self.lambda_ = lambda_

        # We'll use the bounds to create a distribution that we can sample from. Initiall
        # we have now reward information so we'll just sample uniformly
        self.action_distribution = lambda: np.random.uniform(
            low=self.control_bounds_lower,
            high=self.control_bounds_upper,
            size=(self.K, self.H)
        )

    def rollout(self, x0, U):
        """
        Rollout the dynamics function from an initial state with a sequence of actions,
        and return the resulting trajectory of states
        """
        x = x0
        X = [x]
        # That the u here is a sequence of controls, u_0, u_1, ..., u_{H-1}
        for u in U:
            x = self.dynamics_fn(x, u)
            X.append(x)
        return np.array(X)
    
    def score(self, X, nominal_xy_positions):
        """
        Score a trajectory of states
        """
        
        # Optimize two things here:
        # - the positions that we go through should overlap with the nominal positions (use
        #   path comparison implementation for this
        # - the distance we travel should be maximized

    def optimal_control_sequence(self, x0):
        """
        Optimize the control sequence to minimize the cost of the trajectory
        """
        
        # Start by sampling K, H long control sequences
        Us = self.action_distribution()

        # Rollout each of the K control sequences
        Xs = []
        for U in Us:
            X = self.rollout(x0, U)
            Xs.append(X)
        
        # Score each of the trajectories
        scores = []
        for X in Xs:
            score = self.score(X)
            scores.append(score)

        # Take the best
        # TODO MPPI weighted with softmax
        best_index = np.argmax(scores)
        best_U = Us[best_index]
        return best_U


