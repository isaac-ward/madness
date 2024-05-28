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
        lambda_: temperature parameter:
            - if lambda_ << 0 then the smallest control sequence scores are emphasized in the weights 
            - if lambda_ = 0 then the weights are uniform (all equal)
            - if lambda_ >> 0 then the largest control sequence scores are emphasized in the weights
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
    
    def score(self, X, U):
        """
        Score a trajectory of states and controls
        """
        
        # Optimize some things here:
        # - the positions that we go through should overlap with the nominal positions (use
        #   path comparison implementation for this
        # - the distance to the goal should be minimized
        # - we shouldn't go too fast or too slow
        # - we should avoid obstacles
        # - we should not put in too much control effort
        # - stay upright

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
        
        # Calculate the length of the path. We actually want to favor paths
        # that are ~1m long, so we'll punish paths that are too short
        # or too long. If we have a lookahead of 2 seconds, then
        # a 1m long path represents a speed of desired_length/lookahead=0.5m/s
        length = actual_path.length_along_path()
        # TODO put in self.H and self.dt to compute this exactly
        desired_length = 0.4
        length_deviation = np.abs(length - desired_length) # aka speed
        # TODO could also look at vx, vy

        # Stay upright
        angle = X[:,4]
        angle_deviation = np.linalg.norm(angle) 
        # Don't have a high angular velocity
        angular_velocity = X[:,5]
        angular_velocity_deviation = np.linalg.norm(angular_velocity)

        # Control effort should be minimized
        control_effort = np.linalg.norm(U)

        # We're going to maximize this score, so we need to negate 
        # these terms. Now if deviation is large, it's a worse score,
        # and if distance to end is large, it's a worse score, etc.
        deviation        *= -1
        distance_to_end  *= -1
        length_deviation *= -1
        angle_deviation  *= -1
        angular_velocity_deviation *= -1
        control_effort   *= -1

        # Score is a weighted sum
        # Notes from playing around:
        # - the distance_to_end is important to emphasize, otherwise
        #   we'll go backwards along the path, even sometimes 
        #   crashing into the wall in the wrong direction from the 
        #   start
        # - the length_deviation is important to emphasize, otherwise
        #   we'll go too fast and crash
        score = 10 * deviation + 25 * distance_to_end + 30 * length_deviation + 4 * angle_deviation + 1 * angular_velocity_deviation + 0.25 * control_effort

        return score

    def optimal_control_sequence(self, x0, return_scored_rollouts=False):
        """
        Optimize the control sequence to minimize the cost of the trajectory
        """
        
        # TODO implement some form of adaptive importance sampling

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
        for i, _ in enumerate(Xs):
            score = self.score(Xs[i], Us[i])
            scores.append(score)

        # MPPI weighted with softmax + lambda
        weights = np.exp(self.lambda_ * np.array(scores))
        weights /= np.sum(weights)

        # Get the optimal by multiplying the weights by the control sequences
        # and summing
        #opt_U = np.sum(weights[:,np.newaxis,np.newaxis] * Us, axis=0)
        # Take the best control sequence
        opt_U = Us[np.argmax(scores)]

        # Return the optimal control sequence
        if return_scored_rollouts:
            return opt_U, (Xs, scores)
        else:   
            return opt_U


