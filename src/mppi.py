import numpy as np

from path import Path

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
        nominal_xy_positions,
        map,
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
        self.nominal_xy_path = Path(nominal_xy_positions)
        self.map = map

    def sample_action_sequence(self):
        # This is actually the MPOPI simulation - we'll sample a sequence
        # of actions out to the horizon 
        action_sequence = np.random.uniform(
            low=self.control_bounds_lower,
            high=self.control_bounds_upper,
            size=(self.H, len(self.control_bounds_lower))
        )
        return action_sequence

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
    
    def score(self, X):
        """
        Score a trajectory of states
        """
        
        # Optimize two things here:
        # - the positions that we go through should overlap with the nominal positions (use
        #   path comparison implementation for this
        # - the distance to the goal should be minimized

        # Recall that x and y at 0 and 2
        actual_xy_positions = X[:,[0,2]]
        actual_xy_positions = actual_xy_positions.reshape(-1, 2)
        actual_path = Path(actual_xy_positions)

        # If we hit a wall get negative infinity
        if self.map.does_path_hit_boundary(actual_path):
            return -np.inf

        # How much does the actual path deviate from the nominal path?
        deviation = self.nominal_xy_path.deviation_from_path(actual_path)

        # Calculate the distance to the end point
        distance_to_end = np.linalg.norm(actual_xy_positions[-1] - self.nominal_xy_path.path_metres[-1])

        # We're going to maximize this score, so we need to negate 
        # these terms
        deviation = -deviation
        distance_to_end = -distance_to_end

        # Score is a weighted sum of the two
        score = 10 * deviation + 0.5 * distance_to_end

        return score

    def optimal_control_sequence(self, x0, return_scored_rollouts=False):
        """
        Optimize the control sequence to minimize the cost of the trajectory
        """
        
        # Start by sampling K, H long control sequences
        Us = np.array([ self.sample_action_sequence() for _ in range(self.K) ])

        # TODO this should all be vectorized

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

        # TODO MPPI weighted with softmax

        # Take the best
        best_index = np.argmax(scores)
        best_U = Us[best_index]
        if return_scored_rollouts:
            return best_U, (Xs, scores)
        else:   
            return best_U


